"""Generation-based validation scoring + checkpoint-selection callback.

Checkpoints are NOT selected by validation loss alone. We additionally compute
generation-based validation metrics (sequence exact match, node F1, edge F1, NED,
parse validity) and a configurable ``validation_common_score``. We persist three
checkpoints: ``best_by_loss``, ``best_by_common_score``, and ``last_checkpoint``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from transformers import TrainerCallback

from taskbench_sft.config import ExperimentConfig
from taskbench_sft.eval.evaluator import evaluate_predictions
from taskbench_sft.eval.score import common_score
from taskbench_sft.infer.generate import _build_prompt_text, _max_new_tokens
from taskbench_sft.logging_utils import get_logger
from taskbench_sft.prompts.builder import build_messages
from taskbench_sft.schema import GoldSample, Mode, ToolCatalog

logger = get_logger(__name__)


def generate_val_predictions(
    model: Any,
    tokenizer: Any,
    val_samples: Sequence[GoldSample],
    catalogs: Dict[str, ToolCatalog],
    mode: Mode,
    cfg: ExperimentConfig,
) -> List[Dict[str, str]]:
    """Greedy-generate raw responses on (a capped subset of) validation samples."""
    import torch

    cap = cfg.eval.max_val_eval_samples
    subset = list(val_samples)[:cap] if cap and cap > 0 else list(val_samples)
    max_new = _max_new_tokens(mode, cfg)
    was_training = model.training
    model.eval()
    use_cache_prev = getattr(model.config, "use_cache", None)
    model.config.use_cache = True
    device = next(model.parameters()).device
    preds: List[Dict[str, str]] = []
    with torch.no_grad():
        for s in subset:
            messages = build_messages(s, mode, catalogs[s.domain], cfg.prompt)
            prompt_text = _build_prompt_text(tokenizer, messages, cfg)
            enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen = out[0][enc["input_ids"].shape[1]:]
            preds.append(
                {"sample_id": s.id, "raw_response": tokenizer.decode(gen, skip_special_tokens=True)}
            )
    if use_cache_prev is not None:
        model.config.use_cache = use_cache_prev
    if was_training:
        model.train()
    return preds


def compute_val_common_score(
    model: Any,
    tokenizer: Any,
    val_samples: Sequence[GoldSample],
    catalogs: Dict[str, ToolCatalog],
    mode: Mode,
    cfg: ExperimentConfig,
) -> Dict[str, float]:
    """Generate on validation and return overall common metrics + common_score."""
    preds = generate_val_predictions(model, tokenizer, val_samples, catalogs, mode, cfg)
    golds_by_id = {s.id: s for s in val_samples}
    report = evaluate_predictions(preds, golds_by_id, catalogs, mode, cfg)
    overall = report["groups"]["overall"]["overall"]
    score = common_score(overall, cfg.checkpoint_selection)
    overall = dict(overall)
    overall.pop("full_json_specific", None)
    overall["validation_common_score"] = score
    return overall


class CommonScoreCallback(TrainerCallback):
    """On each evaluation, compute the generation-based common score and persist
    ``best_by_common_score`` when it improves."""

    def __init__(
        self,
        trainer_ref: Any,
        tokenizer: Any,
        val_samples: Sequence[GoldSample],
        catalogs: Dict[str, ToolCatalog],
        mode: Mode,
        cfg: ExperimentConfig,
        output_dir: str | Path,
        wandb_run: Any = None,
    ) -> None:
        self.trainer_ref = trainer_ref
        self.tokenizer = tokenizer
        self.val_samples = val_samples
        self.catalogs = catalogs
        self.mode = mode
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.wandb_run = wandb_run
        self.best_score = float("-inf")
        self.history: List[Dict[str, Any]] = []

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: ANN001
        model = kwargs.get("model") or self.trainer_ref.model
        metrics = compute_val_common_score(
            model, self.tokenizer, self.val_samples, self.catalogs, self.mode, self.cfg
        )
        score = metrics["validation_common_score"]
        record = {"step": int(state.global_step), **metrics}
        self.history.append(record)
        # Log generation-based eval metrics to W&B under the eval/* namespace.
        if self.wandb_run is not None and self.wandb_run.enabled:
            self.wandb_run.log(
                {
                    "eval/node_f1": metrics.get("node_f1", 0.0),
                    "eval/edge_f1": metrics.get("edge_f1", 0.0),
                    "eval/trajectory_exact_match": metrics.get("trajectory_exact_match", 0.0),
                    "eval/ned": metrics.get("ned", 0.0),
                    "eval/parse_valid_rate": metrics.get("parse_valid_rate", 0.0),
                    "eval/schema_valid_rate": metrics.get("schema_valid_rate", 0.0),
                    "eval/invalid_tool_rate": metrics.get("hallucinated_tool_rate", 0.0),
                    "eval/validation_common_score": score,
                    "train/global_step": int(state.global_step),
                },
                step=int(state.global_step),
            )
        logger.info(
            "[val gen] step=%d common_score=%.4f node_f1=%.4f edge_f1=%.4f seq_em=%.4f parse=%.4f",
            state.global_step, score, metrics.get("node_f1", 0.0),
            metrics.get("edge_f1", 0.0), metrics.get("sequence_exact_match", 0.0),
            metrics.get("parse_valid_rate", 0.0),
        )
        if score > self.best_score:
            self.best_score = score
            best_dir = self.output_dir / "best_by_common_score"
            self.trainer_ref.save_model(str(best_dir))
            self.tokenizer.save_pretrained(str(best_dir))
            with open(best_dir / "val_common_score.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            logger.info("New best_by_common_score=%.4f saved to %s", score, best_dir)
        # Persist the running history each eval.
        with open(self.output_dir / "val_common_score_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        return control
