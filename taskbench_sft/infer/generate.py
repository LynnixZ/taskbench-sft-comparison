"""Deterministic, resumable inference.

Decoding is greedy/deterministic (``do_sample=False``, ``num_beams=1``) for the
main experiments — no grammar-constrained decoding, so the model's raw ability to
produce each format is what we measure. Each prediction is written as a JSONL
record (resumable: already-inferred sample IDs are skipped). Mode A and Mode B
use different ``max_new_tokens`` set from the validation target-length
distribution; the actual output token count is recorded per sample.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from taskbench_sft.config import ExperimentConfig
from taskbench_sft.logging_utils import get_logger
from taskbench_sft.prompts.builder import build_messages
from taskbench_sft.schema import GoldSample, Mode, ToolCatalog
from taskbench_sft.targets import build_target

logger = get_logger(__name__)


def _max_new_tokens(mode: Mode, cfg: ExperimentConfig) -> int:
    return (
        cfg.inference.full_json_max_new_tokens
        if mode == Mode.FULL_JSON
        else cfg.inference.trajectory_max_new_tokens
    )


def _already_done(path: Path) -> Set[str]:
    done: Set[str] = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(str(json.loads(line)["sample_id"]))
                    except Exception:
                        continue
    return done


def _build_prompt_text(tokenizer: Any, messages: List[Dict[str, str]], cfg: ExperimentConfig) -> str:
    if cfg.model.use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return f"{system}\n\n{user}\n\n"


def run_inference(
    model: Any,
    tokenizer: Any,
    samples: Sequence[GoldSample],
    catalogs: Dict[str, ToolCatalog],
    mode: Mode,
    cfg: ExperimentConfig,
    output_path: str | Path,
    checkpoint_label: str = "",
    include_gold: bool = True,
    progress_cb: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> Path:
    """Generate predictions for ``samples`` and append JSONL records to disk.

    ``progress_cb(completed, total, record)`` is invoked after each generation
    (used to stream inference progress to W&B).
    """
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _already_done(output_path)
    if done:
        logger.info("Resuming inference: %d samples already done in %s", len(done), output_path)

    model.eval()
    device = next(model.parameters()).device
    max_new = _max_new_tokens(mode, cfg)
    todo = [s for s in samples if s.id not in done]
    logger.info("Inference [%s]: %d to generate (max_new_tokens=%d)", mode.value, len(todo), max_new)

    with open(output_path, "a", encoding="utf-8") as wf:
        for i, s in enumerate(todo):
            messages = build_messages(s, mode, catalogs[s.domain], cfg.prompt)
            prompt_text = _build_prompt_text(tokenizer, messages, cfg)
            enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=cfg.inference.do_sample,
                    temperature=cfg.inference.temperature if cfg.inference.do_sample else None,
                    num_beams=cfg.inference.num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                )
            latency = time.time() - t0
            gen_ids = out[0][enc["input_ids"].shape[1]:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            record: Dict[str, Any] = {
                "sample_id": s.id,
                "domain": s.domain,
                "topology": s.topology.value,
                "mode": mode.value,
                "prompt": prompt_text,
                "raw_response": raw,
                "parsed_prediction": None,
                "gold": build_target(s, mode) if include_gold else None,
                "input_tokens": int(enc["input_ids"].shape[1]),
                "output_tokens": int(gen_ids.shape[0]),
                "latency_seconds": round(latency, 4),
                "checkpoint": checkpoint_label,
            }
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
            wf.flush()
            if progress_cb is not None:
                try:
                    progress_cb(i + 1, len(todo), record)
                except Exception:  # progress logging must never break inference
                    pass
            if (i + 1) % 50 == 0:
                logger.info("  ... %d/%d generated", i + 1, len(todo))

    logger.info("Inference [%s] complete -> %s", mode.value, output_path)
    return output_path
