"""Canonical SFT target serialization for the two modes.

* Mode A (full_json): the complete TaskBench plan object
  ``{"task_steps", "task_nodes", "task_links"}`` built directly from the gold
  sample — arguments and links are taken verbatim from the official data.
* Mode B (trajectory): a JSON array of the execution-ordered tool IDs.

Both modes use the same canonical JSON formatting so the one-shot example shown
in the prompt matches the assistant target exactly.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from taskbench_sft.schema import GoldSample, Mode


def canonical_json(obj: Any) -> str:
    """Deterministic, compact JSON (UTF-8, stable key order as constructed)."""
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))


def full_json_object(sample: GoldSample) -> Dict[str, Any]:
    """Build the Mode A target object from the gold sample (verbatim content)."""
    return {
        "task_steps": list(sample.task_steps),
        "task_nodes": [
            {"task": n.task, "arguments": n.arguments} for n in sample.task_nodes
        ],
        "task_links": [
            {"source": l.source, "target": l.target} for l in sample.task_links
        ],
    }


def full_json_target(sample: GoldSample) -> str:
    """Canonical Mode A assistant target string."""
    return canonical_json(full_json_object(sample))


def trajectory_list(sample: GoldSample) -> List[str]:
    """The Mode B target list: execution-ordered tool IDs.

    Requires the sample to have a recovered ``trajectory`` (single node or simple
    chain). Raises if called on an excluded sample.
    """
    if sample.trajectory is None:
        raise ValueError(
            f"Sample {sample.id} has no recovered trajectory "
            f"(excluded: {sample.exclusion_reason})"
        )
    return list(sample.trajectory)


def trajectory_target(sample: GoldSample) -> str:
    """Canonical Mode B assistant target string (a JSON array of tool IDs)."""
    return canonical_json(trajectory_list(sample))


def build_target(sample: GoldSample, mode: Mode) -> str:
    """Dispatch to the correct target serializer for the given mode."""
    if mode == Mode.FULL_JSON:
        return full_json_target(sample)
    if mode == Mode.TRAJECTORY:
        return trajectory_target(sample)
    raise ValueError(f"Unknown mode: {mode}")
