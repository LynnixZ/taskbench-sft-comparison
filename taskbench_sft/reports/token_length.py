"""Token-length report.

Full-JSON targets are much longer than trajectory targets, so before training we
measure input / assistant-target / total sequence lengths for *both* modes and
report mean / median / p90 / p95 / p99 / max plus the truncated-sample count.

``max_seq_length`` must cover at least ``coverage_target`` (default 99%) of the
full-JSON samples. Any sample whose target would be truncated in *either* mode is
added to a shared exclusion set so both modes train on exactly the same IDs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from taskbench_sft.config import ExperimentConfig
from taskbench_sft.logging_utils import get_logger
from taskbench_sft.prompts.builder import build_messages
from taskbench_sft.schema import GoldSample, Mode, ToolCatalog
from taskbench_sft.targets import build_target
from taskbench_sft.tokenization import measure_lengths

logger = get_logger(__name__)


def _percentile_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {k: 0.0 for k in ["mean", "median", "p90", "p95", "p99", "max"]}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _mode_report(
    tokenizer: Any,
    samples: List[GoldSample],
    catalogs: Dict[str, ToolCatalog],
    mode: Mode,
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    input_lens: List[int] = []
    target_lens: List[int] = []
    total_lens: List[int] = []
    truncated_ids: List[str] = []
    for s in samples:
        messages = build_messages(s, mode, catalogs[s.domain], cfg.prompt)
        target = build_target(s, mode)
        m = measure_lengths(
            tokenizer,
            messages,
            target,
            cfg.tokenization.max_seq_length,
            use_chat_template=cfg.model.use_chat_template,
            chat_template_kwargs=cfg.model.chat_template_kwargs,
        )
        input_lens.append(m["input_tokens"])
        target_lens.append(m["target_tokens"])
        total_lens.append(m["total_tokens"])
        if m["truncated"]:
            truncated_ids.append(s.id)
    return {
        "n_samples": len(samples),
        "input_tokens": _percentile_stats(input_lens),
        "target_tokens": _percentile_stats(target_lens),
        "total_tokens": _percentile_stats(total_lens),
        "truncated_count": len(truncated_ids),
        "truncated_ids": sorted(truncated_ids),
    }


def compute_token_length_report(
    tokenizer: Any,
    samples: List[GoldSample],
    catalogs: Dict[str, ToolCatalog],
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    """Compute the token-length report over both modes for the given samples."""
    full = _mode_report(tokenizer, samples, catalogs, Mode.FULL_JSON, cfg)
    traj = _mode_report(tokenizer, samples, catalogs, Mode.TRAJECTORY, cfg)

    # Shared exclusion set: a sample truncated in EITHER mode is excluded from both.
    excluded = sorted(set(full["truncated_ids"]) | set(traj["truncated_ids"]))

    max_len = cfg.tokenization.max_seq_length
    full_total_p = full["total_tokens"]
    coverage_ok = full_total_p["p99"] <= max_len
    logger.info(
        "Token report: full_json total p99=%.0f, max_seq_length=%d -> coverage(99%%) %s",
        full_total_p["p99"], max_len, "OK" if coverage_ok else "INSUFFICIENT",
    )
    if not coverage_ok:
        logger.warning(
            "max_seq_length=%d does not cover %.0f%% of full_json samples "
            "(p99 total tokens=%.0f). Increase tokenization.max_seq_length.",
            max_len, cfg.tokenization.coverage_target * 100, full_total_p["p99"],
        )

    return {
        "max_seq_length": max_len,
        "coverage_target": cfg.tokenization.coverage_target,
        "coverage_satisfied_full_json": coverage_ok,
        "full_json": full,
        "trajectory": traj,
        "shared_excluded_ids": excluded,
        "shared_excluded_count": len(excluded),
    }


def write_token_length_report(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
