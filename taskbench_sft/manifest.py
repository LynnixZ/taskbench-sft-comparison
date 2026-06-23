"""Reproducibility manifest for each run."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from taskbench_sft.config import ExperimentConfig
from taskbench_sft.data.loader import data_file_hashes, file_sha256


def _safe_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"cuda_available": False, "gpu_name": None, "cuda_version": None}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def build_run_manifest(
    cfg: ExperimentConfig,
    run_name: str,
    split_manifest_path: Optional[str | Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the full reproducibility manifest dict."""
    split_hash = None
    split_ids: Dict[str, Any] = {}
    if split_manifest_path and Path(split_manifest_path).exists():
        split_hash = file_sha256(split_manifest_path)
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            sm = json.load(f)
        split_ids = {
            "train_sample_ids_count": len(sm.get("train_sample_ids", [])),
            "validation_sample_ids_count": len(sm.get("validation_sample_ids", [])),
            "test_sample_ids_count": len(sm.get("test_sample_ids", [])),
            "used_seed": sm.get("used_seed"),
        }

    manifest: Dict[str, Any] = {
        "run_name": run_name,
        "project_name": cfg.project_name,
        "base_model_name": cfg.model.name,
        "base_model_revision": cfg.model.revision,
        "tokenizer_name": cfg.model.tokenizer_name or cfg.model.name,
        "tokenizer_revision": cfg.model.revision,
        "dataset_file_sha256": data_file_hashes(cfg.data.domains, cfg.data.raw_dir),
        "split_manifest_sha256": split_hash,
        "split_summary": split_ids,
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "library_versions": {
            "torch": _safe_version("torch"),
            "transformers": _safe_version("transformers"),
            "peft": _safe_version("peft"),
            "datasets": _safe_version("datasets"),
            "numpy": _safe_version("numpy"),
            "scipy": _safe_version("scipy"),
            "sklearn": _safe_version("sklearn"),
        },
        "random_seed": cfg.training.seed,
        "training_config": cfg.training.model_dump(),
        "lora_config": cfg.lora.model_dump(),
        "inference_config": cfg.inference.model_dump(),
        "checkpoint_selection_config": cfg.checkpoint_selection.model_dump(),
        **_cuda_info(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_run_manifest(manifest: Dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path
