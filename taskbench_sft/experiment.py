"""Orchestrate the 4-run experiment matrix.

| Run             | Model     | Training | Prompt     |
| --------------- | --------- | -------- | ---------- |
| Base-Full-JSON  | Base LLM  | No SFT   | Full JSON  |
| Base-Trajectory | Base LLM  | No SFT   | Trajectory |
| SFT-Full-JSON   | Same base | SFT      | Full JSON  |
| SFT-Trajectory  | Same base | SFT      | Trajectory |

Both SFT runs share the same base checkpoint, train/validation sample IDs, seed,
optimizer, LoRA rank, epochs, and batch strategy; only the prompt + assistant
target differ.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from taskbench_sft.config import ExperimentConfig
from taskbench_sft.logging_utils import get_logger
from taskbench_sft.schema import Mode

logger = get_logger(__name__)

_SMOKE_OVERRIDES: Dict[str, Any] = {
    "training": {
        "method": "lora",
        "epochs": 1,
        "max_steps": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "eval_steps": 2,
        "save_steps": 2,
        "bf16": False,
        "gradient_checkpointing": False,
        "logging_steps": 1,
    },
    "tokenization": {"max_seq_length": 1024},
    "eval": {"max_val_eval_samples": 4, "compute_rouge": True},
    "inference": {"full_json_max_new_tokens": 256, "trajectory_max_new_tokens": 64},
    "split": {"out_dir": "artifacts/splits_smoke"},
}


def _ensure_split(cfg: ExperimentConfig) -> Dict[str, Path]:
    from taskbench_sft.data.prepare import load_all_samples
    from taskbench_sft.data.split import make_split, write_split

    base = Path(cfg.split.out_dir)
    manifest = base / "split_manifest.json"
    if not manifest.exists():
        samples = load_all_samples(cfg.data)
        usable = [s for s in samples if s.is_usable and s.topology.value in cfg.data.include_topologies]
        train, val, test, used_seed = make_split(usable, cfg.split)
        write_split(train, val, test, cfg.split, used_seed)
    return {
        "train": base / "train.jsonl",
        "validation": base / "validation.jsonl",
        "test_node": base / "test_node.jsonl",
        "test_chain": base / "test_chain.jsonl",
        "test_all": base / "test_all.jsonl",
        "manifest": manifest,
    }


def _ensure_token_report(cfg: ExperimentConfig, paths: Dict[str, Path], catalogs) -> None:
    from taskbench_sft.data.split import load_split_file
    from taskbench_sft.reports.token_length import (
        compute_token_length_report,
        write_token_length_report,
    )
    from taskbench_sft.train.model import load_tokenizer

    report_path = Path(cfg.tokenization.report_path)
    if report_path.exists():
        return
    tokenizer = load_tokenizer(cfg)
    samples = load_split_file(paths["train"]) + load_split_file(paths["validation"])
    report = compute_token_length_report(tokenizer, samples, catalogs, cfg)
    write_token_length_report(report, report_path)


def _excluded_ids(cfg: ExperimentConfig) -> set:
    path = Path(cfg.tokenization.report_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("shared_excluded_ids", []))
    return set()


def _evaluate_run(cfg: ExperimentConfig, run_name: str, mode: Mode, pred_path: Path, golds_by_id, catalogs) -> Path:
    from taskbench_sft.eval.evaluator import evaluate_predictions, load_predictions, write_report

    preds = load_predictions(pred_path)
    report = evaluate_predictions(preds, golds_by_id, catalogs, mode, cfg)
    out = Path(cfg.output_dir) / run_name / "metrics.json"
    write_report(report, out)
    overall = report["groups"]["overall"]["overall"]
    logger.info(
        "[%s] node_f1=%.3f edge_f1=%.3f ned=%.3f traj_em=%.3f halluc=%.3f parse=%.3f",
        run_name, overall["node_f1"], overall["edge_f1"], overall["ned"],
        overall["trajectory_exact_match"], overall["hallucinated_tool_rate"],
        overall["parse_valid_rate"],
    )
    return out


def run_matrix(cfg: ExperimentConfig, smoke: bool = False) -> Dict[str, Any]:
    """Run all four experiments end to end and build the comparison table."""
    from taskbench_sft.data.prepare import load_catalogs
    from taskbench_sft.data.split import load_split_file
    from taskbench_sft.infer.generate import run_inference
    from taskbench_sft.manifest import build_run_manifest, write_run_manifest
    from taskbench_sft.reports.compare import build_comparison_table
    from taskbench_sft.train.model import load_for_inference, load_tokenizer
    from taskbench_sft.train.trainer import train_mode

    if smoke:
        cfg = cfg.merged_with(_SMOKE_OVERRIDES)
        logger.info("SMOKE MODE: tiny configuration")

    catalogs = load_catalogs(cfg.data)
    paths = _ensure_split(cfg)
    _ensure_token_report(cfg, paths, catalogs)
    excluded = _excluded_ids(cfg)

    train_samples = load_split_file(paths["train"])
    val_samples = load_split_file(paths["validation"])
    test_samples = load_split_file(paths["test_all"])
    if smoke:
        train_samples = train_samples[:32]
        val_samples = val_samples[:8]
        test_samples = test_samples[:16]
    golds_by_id = {s.id: s for s in test_samples}

    tokenizer = load_tokenizer(cfg)
    out_root = Path(cfg.output_dir)
    metric_reports: Dict[str, Path] = {}

    # ---- Baseline runs (no SFT): one base model, both prompts ----
    base_model = load_for_inference(cfg, adapter_dir=None)
    for mode, run_name in [(Mode.FULL_JSON, "Base-Full-JSON"), (Mode.TRAJECTORY, "Base-Trajectory")]:
        pred_path = out_root / run_name / "predictions_test_all.jsonl"
        run_inference(base_model, tokenizer, test_samples, catalogs, mode, cfg, pred_path,
                      checkpoint_label=cfg.model.name)
        metric_reports[run_name] = _evaluate_run(cfg, run_name, mode, pred_path, golds_by_id, catalogs)
        write_run_manifest(build_run_manifest(cfg, run_name, paths["manifest"], {"mode": mode.value, "sft": False}),
                           out_root / run_name)
    del base_model

    # ---- SFT runs: train each mode, infer with its adapter ----
    for mode, run_name in [(Mode.FULL_JSON, "SFT-Full-JSON"), (Mode.TRAJECTORY, "SFT-Trajectory")]:
        run_dir = out_root / run_name
        summary = train_mode(mode, train_samples, val_samples, catalogs, cfg, run_dir, excluded)
        adapter = run_dir / "best_by_common_score"
        if not adapter.exists():
            adapter = run_dir / "best_by_loss"
        model = load_for_inference(cfg, adapter_dir=adapter)
        pred_path = run_dir / "predictions_test_all.jsonl"
        run_inference(model, tokenizer, test_samples, catalogs, mode, cfg, pred_path,
                      checkpoint_label=str(adapter))
        metric_reports[run_name] = _evaluate_run(cfg, run_name, mode, pred_path, golds_by_id, catalogs)
        write_run_manifest(
            build_run_manifest(cfg, run_name, paths["manifest"], {"mode": mode.value, "sft": True, "train_summary": summary}),
            run_dir,
        )
        del model

    # ---- Comparison table ----
    report_args = [f"{name}={path}" for name, path in metric_reports.items()]
    table = build_comparison_table(report_args, str(out_root / "comparison.md"))
    logger.info("Comparison table:\n%s", table)
    print(table)
    return {"metric_reports": {k: str(v) for k, v in metric_reports.items()}}
