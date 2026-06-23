"""Unattended GPU smoke test orchestrator with W&B monitoring.

Runs the four settings (Base/SFT x Full-JSON/Trajectory) on a tiny fixed-seed
Node+Chain split, with:

* a token pre-flight that sets ``max_seq_length`` from the actual data,
* one independent, resumable W&B run per setting (stable ids),
* live inference-progress + per-split (Node/Chain) metric logging,
* greedy decoding, QLoRA SFT (a few steps), no output repair,
* a comparison table + machine-readable summary,

and is safe to re-run (idempotent: completed settings are skipped).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from taskbench_sft.config import ExperimentConfig
from taskbench_sft.logging_utils import get_logger
from taskbench_sft.schema import GoldSample, Mode, ToolCatalog
from taskbench_sft.wandb_utils import WandbRun, init_run

logger = get_logger(__name__)

# (setting slug, Mode, is_sft)
_SETTINGS = [
    ("base-full-json", Mode.FULL_JSON, False),
    ("base-trajectory", Mode.TRAJECTORY, False),
    ("sft-full-json", Mode.FULL_JSON, True),
    ("sft-trajectory", Mode.TRAJECTORY, True),
]


def _gpu_name() -> Optional[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _setting_tags(cfg: ExperimentConfig, mode: Mode, is_sft: bool) -> List[str]:
    tags = list(cfg.wandb.base_tags)
    tags.append("full-json" if mode == Mode.FULL_JSON else "trajectory")
    tags.append("sft" if is_sft else "base")
    return tags


def _wandb_config(
    cfg: ExperimentConfig,
    mode: Mode,
    setting: str,
    split_hash: Optional[str],
    counts: Dict[str, int],
    git_commit: Optional[str],
) -> Dict[str, Any]:
    return {
        "model_name": cfg.model.name,
        "model_revision": cfg.model.revision,
        "mode": mode.value,
        "training_setting": setting,
        "seed": cfg.training.seed,
        "train_sample_count": counts["train"],
        "validation_sample_count": counts["val"],
        "test_node_count": counts["test_node"],
        "test_chain_count": counts["test_chain"],
        "max_steps": cfg.training.max_steps,
        "batch_size": cfg.training.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "lora_rank": cfg.lora.r,
        "quantization": cfg.training.method,
        "max_seq_length": cfg.tokenization.max_seq_length,
        "split_manifest_hash": split_hash,
        "git_commit": git_commit,
        "GPU_name": _gpu_name(),
    }


def _inference_progress_cb(wrun: WandbRun, mode: Mode):
    state = {"lat": [], "out": []}

    def cb(done: int, total: int, record: Dict[str, Any]) -> None:
        state["lat"].append(record.get("latency_seconds", 0.0))
        state["out"].append(record.get("output_tokens", 0))
        if wrun.enabled and (done % 2 == 0 or done == total):
            wrun.log(
                {
                    "inference/completed_samples": done,
                    "inference/total_samples": total,
                    "inference/progress": done / max(total, 1),
                    "inference/average_latency": sum(state["lat"]) / len(state["lat"]),
                    "inference/output_tokens": sum(state["out"]) / len(state["out"]),
                }
            )

    return cb


def _log_eval(wrun: WandbRun, mode: Mode, report: Dict[str, Any]) -> Dict[str, Any]:
    groups = report["groups"]
    overall = groups["overall"]["overall"]
    node = groups.get("topology", {}).get("single")
    chain = groups.get("topology", {}).get("chain")
    payload: Dict[str, Any] = {
        "inference/parse_valid_rate": overall["parse_valid_rate"],
        "inference/schema_valid_rate": overall["schema_valid_rate"],
        "inference/invalid_tool_rate": overall["hallucinated_tool_rate"],
    }
    if node:
        payload.update(
            {
                "test_node/node_f1": node["node_f1"],
                "test_node/exact_match": node["trajectory_exact_match"],
                "test_node/parse_valid_rate": node["parse_valid_rate"],
            }
        )
    if chain:
        payload.update(
            {
                "test_chain/node_f1": chain["node_f1"],
                "test_chain/edge_f1": chain["edge_f1"],
                "test_chain/ned": chain["ned"],
                "test_chain/exact_match": chain["trajectory_exact_match"],
                "test_chain/parse_valid_rate": chain["parse_valid_rate"],
            }
        )
    if wrun.enabled:
        wrun.log(payload)
        wrun.update_summary(payload)
    return overall


def run_gpu_smoke(
    cfg: ExperimentConfig,
    experiment_run_id: str,
    train_n: int = 24,
    val_n: int = 6,
    test_node_n: int = 4,
    test_chain_n: int = 4,
) -> Dict[str, Any]:
    """Run the 4-setting GPU smoke test end to end."""
    from taskbench_sft.data.loader import file_sha256
    from taskbench_sft.data.prepare import load_all_samples, load_catalogs
    from taskbench_sft.data.split import load_split_file, make_smoke_split, write_split
    from taskbench_sft.eval.evaluator import evaluate_predictions, load_predictions, write_report
    from taskbench_sft.infer.generate import run_inference
    from taskbench_sft.manifest import _git_commit
    from taskbench_sft.reports.compare import build_comparison_table
    from taskbench_sft.reports.token_length import (
        compute_token_length_report,
        write_token_length_report,
    )
    from taskbench_sft.train.model import load_for_inference, load_tokenizer
    from taskbench_sft.train.trainer import train_mode

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    git_commit = _git_commit()

    catalogs = load_catalogs(cfg.data)

    # ---- 1. Tiny Node+Chain split (fixed seed) ----
    split_dir = Path(cfg.split.out_dir)
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.exists():
        usable = [
            s for s in load_all_samples(cfg.data)
            if s.is_usable and s.topology.value in cfg.data.include_topologies
        ]
        tr, va, te, used_seed = make_smoke_split(usable, cfg.split, train_n, val_n, test_node_n, test_chain_n)
        write_split(tr, va, te, cfg.split, used_seed)
    split_hash = file_sha256(manifest_path)

    train_samples = load_split_file(split_dir / "train.jsonl")
    val_samples = load_split_file(split_dir / "validation.jsonl")
    test_node = load_split_file(split_dir / "test_node.jsonl")
    test_chain = load_split_file(split_dir / "test_chain.jsonl")
    test_all = load_split_file(split_dir / "test_all.jsonl")
    golds_by_id = {s.id: s for s in test_all}
    counts = {
        "train": len(train_samples),
        "val": len(val_samples),
        "test_node": len(test_node),
        "test_chain": len(test_chain),
    }
    logger.info("Smoke split sizes: %s", counts)

    # ---- 2. Token pre-flight -> set max_seq_length from data ----
    tokenizer = load_tokenizer(cfg)
    report = compute_token_length_report(tokenizer, train_samples + val_samples, catalogs, cfg)
    write_token_length_report(report, cfg.tokenization.report_path)
    full_max = report["full_json"]["total_tokens"]["max"]
    # Round up to a multiple of 128 with headroom, capped at the model context.
    needed = int(math.ceil((full_max + 64) / 128.0) * 128)
    model_ctx = getattr(tokenizer, "model_max_length", 4096)
    if not isinstance(model_ctx, int) or model_ctx > 100000:
        model_ctx = 4096
    cfg = cfg.merged_with({"tokenization": {"max_seq_length": min(needed, model_ctx)}})
    excluded = set(report.get("shared_excluded_ids", []))
    logger.info(
        "Token pre-flight: full_json max total=%d -> max_seq_length=%d (excluded=%d)",
        int(full_max), cfg.tokenization.max_seq_length, len(excluded),
    )

    # ---- 3. Run each setting ----
    metric_reports: Dict[str, Path] = {}
    summaries: Dict[str, Any] = {}
    base_model = None

    for slug, mode, is_sft in _SETTINGS:
        run_name = f"{slug}"
        run_id = f"{experiment_run_id}-{slug}"
        run_dir = out_root / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"

        if metrics_path.exists():
            logger.info("[%s] already complete (metrics.json present); skipping", slug)
            metric_reports[run_name] = metrics_path
            continue

        wrun = init_run(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=run_name,
            run_id=run_id,
            tags=_setting_tags(cfg, mode, is_sft),
            config=_wandb_config(cfg, mode, slug, split_hash, counts, git_commit),
            requested_mode=cfg.wandb.mode,
            log_model=cfg.wandb.log_model,
            dir=str(run_dir),
        )
        try:
            t0 = time.time()
            if is_sft:
                summary = train_mode(
                    mode, train_samples, val_samples, catalogs, cfg, run_dir, excluded, wandb_run=wrun
                )
                summaries[slug] = summary
                adapter = run_dir / "best_by_common_score"
                if not adapter.exists():
                    adapter = run_dir / "best_by_loss"
                model = load_for_inference(cfg, adapter_dir=adapter)
                checkpoint_label = str(adapter)
            else:
                if base_model is None:
                    base_model = load_for_inference(cfg, adapter_dir=None)
                model = base_model
                checkpoint_label = cfg.model.name

            pred_path = run_dir / "predictions_test_all.jsonl"
            run_inference(
                model, tokenizer, test_all, catalogs, mode, cfg, pred_path,
                checkpoint_label=checkpoint_label,
                progress_cb=_inference_progress_cb(wrun, mode),
            )
            report_obj = evaluate_predictions(
                load_predictions(pred_path), golds_by_id, catalogs, mode, cfg
            )
            write_report(report_obj, metrics_path)
            overall = _log_eval(wrun, mode, report_obj)
            wrun.update_summary({"runtime_seconds": round(time.time() - t0, 1)})
            metric_reports[run_name] = metrics_path
            logger.info(
                "[%s] node_f1=%.3f edge_f1=%.3f ned=%.3f traj_em=%.3f parse=%.3f (%.0fs)",
                slug, overall["node_f1"], overall["edge_f1"], overall["ned"],
                overall["trajectory_exact_match"], overall["parse_valid_rate"], time.time() - t0,
            )
            if is_sft:
                del model
        finally:
            wrun.finish()

    # ---- 4. Comparison table + summary ----
    report_args = [f"{name}={path}" for name, path in metric_reports.items()]
    table = build_comparison_table(report_args, str(out_root / "comparison.md"))
    summary = {
        "experiment_run_id": experiment_run_id,
        "git_commit": git_commit,
        "gpu": _gpu_name(),
        "split_manifest_hash": split_hash,
        "counts": counts,
        "max_seq_length": cfg.tokenization.max_seq_length,
        "settings": list(metric_reports.keys()),
        "train_summaries": summaries,
    }
    with open(out_root / "smoke_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("GPU smoke complete. Comparison:\n%s", table)
    print(table)
    return summary
