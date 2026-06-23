"""Weights & Biases integration with graceful degradation.

Design goals (per the unattended-GPU requirements):

* All secrets (``WANDB_API_KEY``) come from the environment — never hard-coded,
  never printed.
* If W&B is unavailable or online init fails, we **warn and fall back** to
  ``offline`` (and finally ``disabled``) rather than crashing the experiment.
  The offline directory is kept under the run output so it can be ``wandb sync``-ed
  later from the results tarball.
* Stable, deterministic run IDs (``{experiment_run_id}-{setting}``) so resubmits
  resume instead of creating duplicate runs.
* Never upload the base model or full checkpoints (``WANDB_LOG_MODEL=false``).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from taskbench_sft.logging_utils import get_logger

logger = get_logger(__name__)


def _import_wandb():
    try:
        import wandb  # noqa: PLC0415

        return wandb
    except Exception:  # pragma: no cover - exercised only without the dep
        return None


class WandbRun:
    """Thin wrapper so callers never branch on whether W&B is active."""

    def __init__(self, run: Any, mode: str) -> None:
        self._run = run
        self.mode = mode

    @property
    def enabled(self) -> bool:
        return self._run is not None

    @property
    def run(self) -> Any:
        return self._run

    def log(self, data: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        if self._run is not None:
            try:
                self._run.log(data, step=step, commit=commit)
            except Exception as exc:  # never let logging crash training
                logger.warning("wandb.log failed: %s", type(exc).__name__)

    def update_summary(self, data: Dict[str, Any]) -> None:
        if self._run is not None:
            try:
                for k, v in data.items():
                    self._run.summary[k] = v
            except Exception as exc:
                logger.warning("wandb summary update failed: %s", type(exc).__name__)

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
            self._run = None


def _resolve_mode(requested: str) -> str:
    """Resolve the effective mode from config + env, with API-key gating."""
    mode = os.environ.get("WANDB_MODE", requested or "online").lower()
    if mode == "disabled":
        return "disabled"
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY is not set; falling back to offline W&B logging")
        return "offline"
    return mode


def init_run(
    *,
    project: str,
    entity: Optional[str],
    group: str,
    name: str,
    run_id: str,
    tags: List[str],
    config: Dict[str, Any],
    requested_mode: str = "online",
    log_model: bool = False,
    dir: Optional[str] = None,
) -> WandbRun:
    """Initialize a W&B run, degrading gracefully on any failure.

    Returns a :class:`WandbRun`; ``.enabled`` is False if W&B is disabled or all
    init attempts failed. The active run is also picked up automatically by the
    HF ``Trainer`` W&B callback (so train/* metrics flow to the same run).
    """
    wandb = _import_wandb()
    if wandb is None:
        logger.warning("wandb not installed; proceeding without W&B logging")
        return WandbRun(None, "disabled")

    # Never upload large artifacts from a smoke test.
    os.environ.setdefault("WANDB_LOG_MODEL", "true" if log_model else "false")

    effective = _resolve_mode(requested_mode)
    if effective == "disabled":
        logger.info("W&B disabled by configuration")
        return WandbRun(None, "disabled")

    if dir:
        os.makedirs(dir, exist_ok=True)

    # Try the requested mode, then offline, then give up (disabled).
    for attempt in [effective, "offline"]:
        try:
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", project),
                entity=os.environ.get("WANDB_ENTITY", entity),
                group=os.environ.get("WANDB_RUN_GROUP", group),
                name=name,
                id=run_id,
                resume="allow",
                tags=tags,
                config=config,
                mode=attempt,
                dir=dir,
                reinit=True,
                settings=wandb.Settings(start_method="thread"),
            )
            if attempt != effective:
                logger.warning("W&B init fell back to mode=%s", attempt)
            else:
                logger.info("W&B run started: name=%s id=%s mode=%s", name, run_id, attempt)
            return WandbRun(run, attempt)
        except Exception as exc:
            logger.warning("W&B init failed (mode=%s): %s: %s", attempt, type(exc).__name__, exc)

    logger.warning("W&B disabled after failed init attempts")
    return WandbRun(None, "disabled")
