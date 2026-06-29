"""Command-line interface tying the whole pipeline together.

Subcommands (run ``python -m taskbench_sft.cli <cmd> --help``):

    stats         Compute dataset statistics from the downloaded data.
    split         Build the stratified train/val/test split + manifest.
    token-report  Token-length report (needs the model tokenizer).
    train         SFT one mode (full_json | trajectory).
    infer         Generate predictions for a run (base or SFT).
    evaluate      Compute grouped metrics from a predictions file.
    compare       Build the final comparison table across runs.
    run-matrix    Orchestrate the 4-run matrix end to end.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from taskbench_sft.config import ExperimentConfig, load_config
from taskbench_sft.logging_utils import configure_logging, get_logger
from taskbench_sft.schema import Mode

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _coerce(value: str) -> Any:
    """Coerce a CLI string to bool/int/float/None/JSON, else leave as str."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _parse_set(pairs: List[str]) -> Dict[str, Any]:
    """Parse ``--set a.b.c=value`` pairs into a nested override dict."""
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got: {pair}")
        key, _, value = pair.partition("=")
        node = out
        keys = key.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _coerce(value)
    return out


def _load_cfg(args: argparse.Namespace) -> ExperimentConfig:
    overrides: Dict[str, Any] = {}
    if getattr(args, "seed", None) is not None:
        overrides.setdefault("split", {})["seed"] = args.seed
        overrides.setdefault("training", {})["seed"] = args.seed
    if getattr(args, "model_name", None):
        overrides.setdefault("model", {})["name"] = args.model_name
    if getattr(args, "max_steps", None) is not None:
        overrides.setdefault("training", {})["max_steps"] = args.max_steps
    cfg = load_config(getattr(args, "config", None), overrides or None)
    set_over = _parse_set(getattr(args, "set", None) or [])
    if set_over:
        cfg = cfg.merged_with(set_over)
    return cfg


def _split_paths(cfg: ExperimentConfig) -> Dict[str, Path]:
    base = Path(cfg.split.out_dir)
    return {
        "train": base / "train.jsonl",
        "validation": base / "validation.jsonl",
        "test_node": base / "test_node.jsonl",
        "test_chain": base / "test_chain.jsonl",
        "test_all": base / "test_all.jsonl",
        "manifest": base / "split_manifest.json",
    }


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_stats(args: argparse.Namespace) -> None:
    from taskbench_sft.data.prepare import load_all_samples, load_catalogs
    from taskbench_sft.reports.dataset_stats import compute_dataset_stats, write_dataset_stats

    cfg = _load_cfg(args)
    samples = load_all_samples(cfg.data)
    catalogs = load_catalogs(cfg.data)
    stats = compute_dataset_stats(samples, catalogs)
    out = Path(args.out or "artifacts/dataset_stats.json")
    write_dataset_stats(stats, out)
    logger.info("Wrote dataset stats -> %s", out)
    print(json.dumps(stats["overall"], indent=2))


def cmd_split(args: argparse.Namespace) -> None:
    from taskbench_sft.data.prepare import load_all_samples
    from taskbench_sft.data.split import make_split, write_split

    cfg = _load_cfg(args)
    samples = load_all_samples(cfg.data)
    usable = [s for s in samples if s.is_usable and s.topology.value in cfg.data.include_topologies]
    if cfg.split.mode == "gnn4plan":
        from pathlib import Path as _Path
        from taskbench_sft.data.split import make_split_gnn4plan
        test_ids: list = []
        for domain in cfg.data.domains:
            sp = _Path(cfg.data.raw_dir) / domain / "split_ids.json"
            if not sp.exists():
                raise FileNotFoundError(
                    f"split.mode=gnn4plan but {sp} missing -- run scripts/download_gnn4plan.sh first"
                )
            with open(sp, "r", encoding="utf-8") as f:
                test_ids.extend(json.load(f)["test_ids"]["chain"])
        # DAG augmentation (only when 'dag' is in include_topologies); split runs per-domain.
        dag_test_n = dag_train_cap = 0
        if "dag" in cfg.data.include_topologies:
            dag_train_cap = cfg.split.dag_train_cap
            dom = cfg.data.domains[0] if cfg.data.domains else None
            dag_test_n = cfg.split.dag_test_per_domain.get(dom, 0)
        train, val, test, used_seed = make_split_gnn4plan(
            usable, cfg.split, test_ids, dag_test_n, dag_train_cap
        )
    else:
        train, val, test, used_seed = make_split(usable, cfg.split)
    manifest = write_split(train, val, test, cfg.split, used_seed)
    logger.info("Split written to %s (used_seed=%d)", cfg.split.out_dir, used_seed)
    print(json.dumps({k: v["total"] for k, v in manifest["splits"].items()}, indent=2))


def cmd_token_report(args: argparse.Namespace) -> None:
    from taskbench_sft.data.prepare import load_catalogs
    from taskbench_sft.data.split import load_split_file
    from taskbench_sft.reports.token_length import (
        compute_token_length_report,
        write_token_length_report,
    )
    from taskbench_sft.train.model import load_tokenizer

    cfg = _load_cfg(args)
    paths = _split_paths(cfg)
    catalogs = load_catalogs(cfg.data)
    samples = load_split_file(paths["train"]) + load_split_file(paths["validation"])
    tokenizer = load_tokenizer(cfg)
    report = compute_token_length_report(tokenizer, samples, catalogs, cfg)
    out = Path(args.out or cfg.tokenization.report_path)
    write_token_length_report(report, out)
    logger.info("Wrote token-length report -> %s", out)
    print(
        json.dumps(
            {
                "max_seq_length": report["max_seq_length"],
                "coverage_satisfied_full_json": report["coverage_satisfied_full_json"],
                "full_json_total_p99": report["full_json"]["total_tokens"]["p99"],
                "trajectory_total_p99": report["trajectory"]["total_tokens"]["p99"],
                "shared_excluded_count": report["shared_excluded_count"],
            },
            indent=2,
        )
    )


def _excluded_ids(cfg: ExperimentConfig) -> set:
    """Load the shared truncation-exclusion set from the token-length report if present."""
    path = Path(cfg.tokenization.report_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("shared_excluded_ids", []))
    return set()


def cmd_train(args: argparse.Namespace) -> None:
    from taskbench_sft.data.prepare import load_catalogs
    import os

    from taskbench_sft.data.split import load_split_file
    from taskbench_sft.manifest import build_run_manifest, write_run_manifest
    from taskbench_sft.train.trainer import train_mode
    from taskbench_sft.wandb_utils import init_run

    cfg = _load_cfg(args)
    mode = Mode(args.mode)
    paths = _split_paths(cfg)
    catalogs = load_catalogs(cfg.data)
    train_samples = load_split_file(paths["train"])
    val_samples = load_split_file(paths["validation"])
    run_dir = Path(cfg.output_dir) / args.run_name

    # W&B run for live monitoring (train/loss, grad_norm, lr; eval/node_f1, etc.).
    wrun = None
    if cfg.wandb.enabled and not args.no_wandb:
        exp_id = os.environ.get("EXPERIMENT_RUN_ID") or cfg.experiment_run_id or "sweep"
        wrun = init_run(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=args.run_name,
            run_id=f"{exp_id}-{args.run_name}",
            tags=list(cfg.wandb.base_tags) + ["sft", mode.value, "sweep"],
            config={
                "mode": mode.value, "run_name": args.run_name,
                "learning_rate": cfg.training.learning_rate, "lora_rank": cfg.lora.r,
                "lora_alpha": cfg.lora.alpha, "epochs": cfg.training.epochs,
                "method": cfg.training.method, "max_grad_norm": cfg.training.max_grad_norm,
                "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
                "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
                "warmup_ratio": cfg.training.warmup_ratio, "seed": cfg.training.seed,
                "model_name": cfg.model.name, "max_seq_length": cfg.tokenization.max_seq_length,
            },
            requested_mode=cfg.wandb.mode,
            log_model=cfg.wandb.log_model,
            dir=str(run_dir),
        )
    try:
        summary = train_mode(
            mode, train_samples, val_samples, catalogs, cfg, run_dir, _excluded_ids(cfg),
            wandb_run=wrun,
        )
    finally:
        if wrun is not None:
            wrun.finish()
    manifest = build_run_manifest(
        cfg, args.run_name, paths["manifest"], extra={"train_summary": summary, "mode": mode.value}
    )
    write_run_manifest(manifest, run_dir)
    print(json.dumps(summary["compute"], indent=2))


def cmd_infer(args: argparse.Namespace) -> None:
    from taskbench_sft.data.prepare import load_catalogs
    from taskbench_sft.data.split import load_split_file
    from taskbench_sft.infer.generate import run_inference
    from taskbench_sft.train.model import load_for_inference, load_tokenizer

    cfg = _load_cfg(args)
    mode = Mode(args.mode)
    paths = _split_paths(cfg)
    catalogs = load_catalogs(cfg.data)
    test_samples = load_split_file(paths[args.split])
    if getattr(args, "limit", None):
        test_samples = test_samples[: args.limit]
        logger.info("Limiting inference to first %d samples", args.limit)
    tokenizer = load_tokenizer(cfg)
    adapter = args.adapter if args.adapter else None
    model = load_for_inference(cfg, adapter_dir=adapter)
    out_path = Path(args.out or (Path(cfg.output_dir) / args.run_name / f"predictions_{args.split}.jsonl"))
    run_inference(
        model, tokenizer, test_samples, catalogs, mode, cfg, out_path,
        checkpoint_label=str(adapter or cfg.model.name),
    )
    print(str(out_path))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from taskbench_sft.data.prepare import load_catalogs
    from taskbench_sft.data.split import load_split_file
    from taskbench_sft.eval.evaluator import evaluate_predictions, load_predictions, write_report

    cfg = _load_cfg(args)
    mode = Mode(args.mode)
    paths = _split_paths(cfg)
    catalogs = load_catalogs(cfg.data)
    golds_by_id = {}
    for key in ["test_node", "test_chain", "test_all", "validation"]:
        for s in load_split_file(paths[key]):
            golds_by_id[s.id] = s
    predictions = load_predictions(args.predictions)
    report = evaluate_predictions(predictions, golds_by_id, catalogs, mode, cfg)
    out = Path(args.out or (Path(args.predictions).with_suffix(".metrics.json")))
    write_report(report, out)
    logger.info("Wrote metrics -> %s", out)
    overall = report["groups"]["overall"]["overall"]
    print(json.dumps({k: overall[k] for k in [
        "n_samples", "node_f1", "edge_f1", "ned", "trajectory_exact_match",
        "hallucinated_tool_rate", "parse_valid_rate",
    ]}, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    from taskbench_sft.reports.compare import build_comparison_table

    table = build_comparison_table(args.reports, args.out)
    print(table)


def cmd_run_matrix(args: argparse.Namespace) -> None:
    from taskbench_sft.experiment import run_matrix

    cfg = _load_cfg(args)
    run_matrix(cfg, smoke=args.smoke)


def _default_run_id() -> str:
    from taskbench_sft.manifest import _git_commit

    commit = _git_commit()
    return f"smoke-{commit[:12]}" if commit else "smoke-local"


def cmd_gpu_smoke(args: argparse.Namespace) -> None:
    """Unattended GPU smoke test (4 settings, W&B, resumable, diagnostics)."""
    import os
    import platform
    import sys
    import traceback

    from taskbench_sft.smoke_gpu import run_gpu_smoke

    cfg = _load_cfg(args)
    overrides: Dict[str, Any] = {}
    if os.environ.get("MODEL_NAME"):
        overrides.setdefault("model", {})["name"] = os.environ["MODEL_NAME"]
    if os.environ.get("OUTPUT_DIR"):
        overrides["output_dir"] = os.environ["OUTPUT_DIR"]
    if overrides:
        cfg = cfg.merged_with(overrides)

    # Qwen3 reasons by default (<think> traces) which would consume the answer
    # budget and break direct JSON/array output; disable it unless overridden.
    if "qwen3" in cfg.model.name.lower() and not cfg.model.chat_template_kwargs:
        cfg = cfg.merged_with({"model": {"chat_template_kwargs": {"enable_thinking": False}}})
        logger.info("Detected Qwen3: setting chat_template_kwargs.enable_thinking=False")

    run_id = os.environ.get("EXPERIMENT_RUN_ID") or cfg.experiment_run_id or _default_run_id()
    out_root = Path(cfg.output_dir)
    logger.info("GPU smoke: experiment_run_id=%s model=%s output=%s", run_id, cfg.model.name, out_root)
    try:
        run_gpu_smoke(
            cfg, run_id,
            train_n=args.train_n, val_n=args.val_n,
            test_node_n=args.test_node_n, test_chain_n=args.test_chain_n,
        )
    except Exception as exc:  # noqa: BLE001 - we want to capture everything
        out_root.mkdir(parents=True, exist_ok=True)
        diag = {
            "failed": True,
            "experiment_run_id": run_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "model_name": cfg.model.name,
        }
        try:
            import torch

            diag["cuda_available"] = torch.cuda.is_available()
            diag["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            pass
        with open(out_root / "diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diag, f, ensure_ascii=False, indent=2)
        logger.exception("GPU smoke failed; diagnostics -> %s", out_root / "diagnostics.json")
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="taskbench_sft", description=__doc__)
    p.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    p.add_argument("--seed", type=int, default=None, help="Override split/training seed.")
    p.add_argument("--model-name", type=str, default=None, help="Override base model name.")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override any config field, e.g. --set training.learning_rate=5e-4 "
             "--set lora.r=32 (repeatable; used for hyperparameter sweeps).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("stats", help="Dataset statistics")
    s.add_argument("--out", type=str, default=None)
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("split", help="Build train/val/test split")
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("token-report", help="Token-length report")
    s.add_argument("--out", type=str, default=None)
    s.set_defaults(func=cmd_token_report)

    s = sub.add_parser("train", help="Train one mode (W&B-monitored)")
    s.add_argument("--mode", required=True, choices=["full_json", "trajectory"])
    s.add_argument("--run-name", required=True)
    s.add_argument("--max-steps", type=int, default=None)
    s.add_argument("--no-wandb", action="store_true", help="Disable W&B for this run")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("infer", help="Generate predictions")
    s.add_argument("--mode", required=True, choices=["full_json", "trajectory"])
    s.add_argument("--run-name", required=True)
    s.add_argument("--split", default="test_all",
                   choices=["test_node", "test_chain", "test_all", "validation"])
    s.add_argument("--adapter", type=str, default=None, help="Adapter dir (omit for base model).")
    s.add_argument("--out", type=str, default=None)
    s.add_argument("--limit", type=int, default=None, help="Only generate for the first N test samples (smoke).")
    s.set_defaults(func=cmd_infer)

    s = sub.add_parser("evaluate", help="Evaluate predictions")
    s.add_argument("--mode", required=True, choices=["full_json", "trajectory"])
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", type=str, default=None)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("compare", help="Build comparison table from metric reports")
    s.add_argument("--reports", nargs="+", required=True, help="run_name=path pairs")
    s.add_argument("--out", type=str, default=None)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("run-matrix", help="Run the 4-run experiment matrix")
    s.add_argument("--max-steps", type=int, default=None)
    s.add_argument("--smoke", action="store_true", help="Tiny smoke configuration")
    s.set_defaults(func=cmd_run_matrix)

    s = sub.add_parser("gpu-smoke", help="Unattended GPU smoke test (W&B, resumable)")
    s.add_argument("--max-steps", type=int, default=None)
    s.add_argument("--train-n", type=int, default=24)
    s.add_argument("--val-n", type=int, default=6)
    s.add_argument("--test-node-n", type=int, default=4)
    s.add_argument("--test-chain-n", type=int, default=4)
    s.set_defaults(func=cmd_gpu_smoke)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
