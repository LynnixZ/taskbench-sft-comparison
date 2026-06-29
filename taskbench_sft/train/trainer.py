"""Train one SFT run for a single mode.

Both SFT runs (Full-JSON and Trajectory) share the same base checkpoint, the same
train/validation sample IDs, the same seed, optimizer, LoRA rank, epochs, and
batch strategy. The only intended difference is the prompt and assistant target.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from taskbench_sft.config import ExperimentConfig
from taskbench_sft.logging_utils import get_logger
from taskbench_sft.schema import GoldSample, Mode, ToolCatalog
from taskbench_sft.tokenization import IGNORE_INDEX
from taskbench_sft.train.checkpoint_select import CommonScoreCallback
from taskbench_sft.train.dataset import DataCollatorForCausalSFT, SupervisedDataset
from taskbench_sft.train.model import load_model, load_tokenizer

logger = get_logger(__name__)


def _dataset_token_stats(ds: SupervisedDataset) -> Dict[str, int]:
    input_tokens = 0
    target_tokens = 0
    total_tokens = 0
    for ex in ds.examples:
        total = len(ex["input_ids"])
        tgt = sum(1 for x in ex["labels"] if x != IGNORE_INDEX)
        total_tokens += total
        target_tokens += tgt
        input_tokens += total - tgt
    return {
        "train_examples": len(ds.examples),
        "input_tokens": input_tokens,
        "assistant_target_tokens": target_tokens,
        "total_processed_tokens": total_tokens,
    }


def train_mode(
    mode: Mode,
    train_samples: Sequence[GoldSample],
    val_samples: Sequence[GoldSample],
    catalogs: Dict[str, ToolCatalog],
    cfg: ExperimentConfig,
    output_dir: str | Path,
    excluded_ids: Optional[set] = None,
    wandb_run: Any = None,
) -> Dict[str, Any]:
    """Run SFT for one mode; returns a compute-fairness + checkpoint summary."""
    import torch
    from transformers import Trainer, TrainingArguments

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, for_training=True)

    train_ds = SupervisedDataset(train_samples, catalogs, tokenizer, mode, cfg, excluded_ids)
    # Optionally cap the val set used for eval_loss (keeps smoke tests fast).
    val_for_loss = list(val_samples)
    if cfg.eval.max_val_samples and len(val_for_loss) > cfg.eval.max_val_samples:
        val_for_loss = val_for_loss[: cfg.eval.max_val_samples]
    val_ds = SupervisedDataset(val_for_loss, catalogs, tokenizer, mode, cfg, excluded_ids)
    collator = DataCollatorForCausalSFT(tokenizer)
    token_stats = _dataset_token_stats(train_ds)

    args = TrainingArguments(
        output_dir=str(output_dir / "hf_trainer"),
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.scheduler,
        max_grad_norm=cfg.training.max_grad_norm,
        optim=cfg.training.optim,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        logging_steps=cfg.training.logging_steps,
        # eval & save must use the same strategy for load_best_model_at_end; for
        # "steps" we save on the eval cadence so the best checkpoint is captured.
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.eval_strategy,
        save_steps=cfg.training.eval_steps if cfg.training.eval_strategy == "steps" else cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_steps=cfg.training.max_steps if cfg.training.max_steps else -1,
        seed=cfg.training.seed,
        # HF Trainer's W&B callback reuses the already-active run (started by the
        # orchestrator with the stable run id), so train/* metrics land there.
        report_to=["wandb"] if (wandb_run is not None and wandb_run.enabled) else [],
        remove_unused_columns=False,
    )

    rs = getattr(cfg.training, "rule_smoothing", None)
    use_rule = bool(rs and rs.enabled and mode == Mode.TRAJECTORY)
    if use_rule:
        from taskbench_sft.train.rule_smoothing import rule_smoothing_loss

        class RuleSmoothingTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                soft = inputs.pop("soft_targets", None)
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss = rule_smoothing_loss(outputs.logits, labels, soft)
                return (loss, outputs) if return_outputs else loss

        TrainerCls: Any = RuleSmoothingTrainer
        logger.info("Rule-aware label smoothing ENABLED (alpha_max=%s, max_lag=%s)",
                    rs.alpha_max, rs.max_lag)
    else:
        TrainerCls = Trainer

    trainer = TrainerCls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    score_cb = CommonScoreCallback(
        trainer, tokenizer, val_samples, catalogs, mode, cfg, output_dir, wandb_run=wandb_run
    )
    trainer.add_callback(score_cb)

    # Early stopping on eval_loss (patience counts evals = epochs when
    # eval_strategy="epoch"): stop after N consecutive non-improving evals.
    if cfg.training.early_stopping_patience:
        from transformers import EarlyStoppingCallback

        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience,
                early_stopping_threshold=cfg.training.early_stopping_threshold,
            )
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    wall_clock = time.time() - t0

    # best_by_loss is the loaded best model (load_best_model_at_end on eval_loss).
    trainer.save_model(str(output_dir / "best_by_loss"))
    tokenizer.save_pretrained(str(output_dir / "best_by_loss"))
    # last_checkpoint snapshot.
    trainer.save_model(str(output_dir / "last_checkpoint"))
    tokenizer.save_pretrained(str(output_dir / "last_checkpoint"))

    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )
    summary = {
        "mode": mode.value,
        "compute": {
            **token_stats,
            "optimization_steps": int(trainer.state.global_step),
            "epochs": cfg.training.epochs,
            "wall_clock_seconds": round(wall_clock, 2),
            "peak_gpu_memory_gib": round(peak_mem, 3),
            "train_loss": float(train_result.training_loss),
        },
        "checkpoints": {
            "best_by_loss": str(output_dir / "best_by_loss"),
            "best_by_common_score": str(output_dir / "best_by_common_score"),
            "last_checkpoint": str(output_dir / "last_checkpoint"),
        },
        "best_common_score": score_cb.best_score,
    }
    with open(output_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Training [%s] done in %.1fs; %s", mode.value, wall_clock, summary["compute"])

    # Release the training model/optimizer from the GPU before the caller loads
    # a separate model for inference (avoids stacking two copies -> OOM).
    import gc

    del trainer, model, score_cb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
