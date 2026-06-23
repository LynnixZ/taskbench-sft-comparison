"""Parse raw model responses into a unified prediction structure.

We never silently fix predictions: invalid JSON, schema violations, and
off-catalog ("hallucinated") tool names are recorded, not repaired. Both modes
are projected onto a common structure (node names, edges, trajectory) so the
shared planning metrics can be computed identically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from taskbench_sft.schema import Mode, Topology


@dataclass
class ParsedPrediction:
    sample_id: str
    domain: str
    topology: str
    mode: str
    parse_valid: bool
    schema_valid: bool
    pred_node_names: List[str] = field(default_factory=list)
    pred_edges: List[Tuple[str, str]] = field(default_factory=list)
    pred_trajectory: List[str] = field(default_factory=list)
    pred_task_steps: Optional[List[str]] = None
    pred_arguments: Optional[List[List[Any]]] = None
    failure_reason: Optional[str] = None
    # Diagnostics computed during parsing.
    step_node_aligned: bool = False
    links_valid: bool = False


def _extract_json_object(text: str) -> Optional[Any]:
    """Extract the outermost JSON object from raw text (mirrors official logic)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _extract_json_array(text: str) -> Optional[Any]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _adjacent_edges(traj: List[str]) -> List[Tuple[str, str]]:
    return [(traj[i], traj[i + 1]) for i in range(len(traj) - 1)]


def _recover_pred_trajectory(
    node_names: List[str], edges: List[Tuple[str, str]]
) -> List[str]:
    """Recover an ordered trajectory from predicted nodes + links.

    Uses an index-based topological sort when the predicted links form a simple
    path; otherwise falls back to the order nodes were listed (a documented,
    deterministic fallback — we never invent ordering information).
    """
    from taskbench_sft.data.topology import is_simple_path, topological_order

    n = len(node_names)
    if n <= 1:
        return list(node_names)
    idx_edges: List[Tuple[int, int]] = []
    ambiguous = False
    for s, t in edges:
        si = [k for k, nm in enumerate(node_names) if nm == s]
        ti = [k for k, nm in enumerate(node_names) if nm == t]
        if len(si) != 1 or len(ti) != 1:
            ambiguous = True
            break
        idx_edges.append((si[0], ti[0]))
    if ambiguous or not is_simple_path(n, idx_edges):
        return list(node_names)
    order = topological_order(n, idx_edges)
    if order is None:
        return list(node_names)
    return [node_names[i] for i in order]


def parse_full_json(sample_id: str, domain: str, topology: str, raw: str) -> ParsedPrediction:
    obj = _extract_json_object(raw)
    if obj is None or not isinstance(obj, dict):
        return ParsedPrediction(
            sample_id, domain, topology, Mode.FULL_JSON.value,
            parse_valid=False, schema_valid=False, failure_reason="invalid_json",
        )
    has_keys = all(k in obj for k in ("task_steps", "task_nodes", "task_links"))
    nodes = obj.get("task_nodes", [])
    links = obj.get("task_links", [])
    steps = obj.get("task_steps", [])
    schema_valid = (
        has_keys
        and isinstance(nodes, list)
        and isinstance(links, list)
        and isinstance(steps, list)
        and all(isinstance(n, dict) and "task" in n for n in nodes)
        and all(isinstance(l, dict) and "source" in l and "target" in l for l in links)
    )
    node_names = [str(n["task"]) for n in nodes if isinstance(n, dict) and "task" in n]
    arguments = [list(n.get("arguments", [])) for n in nodes if isinstance(n, dict) and "task" in n]
    edges = [
        (str(l["source"]), str(l["target"]))
        for l in links
        if isinstance(l, dict) and "source" in l and "target" in l
    ]
    links_valid = isinstance(links, list) and all(
        isinstance(l, dict) and "source" in l and "target" in l for l in links
    )
    step_node_aligned = isinstance(steps, list) and len(steps) == len(node_names)
    trajectory = _recover_pred_trajectory(node_names, edges)
    return ParsedPrediction(
        sample_id, domain, topology, Mode.FULL_JSON.value,
        parse_valid=True,
        schema_valid=bool(schema_valid),
        pred_node_names=node_names,
        pred_edges=edges,
        pred_trajectory=trajectory,
        pred_task_steps=[str(s) for s in steps] if isinstance(steps, list) else None,
        pred_arguments=arguments,
        step_node_aligned=step_node_aligned,
        links_valid=links_valid,
        failure_reason=None if schema_valid else "schema_invalid",
    )


def parse_trajectory(sample_id: str, domain: str, topology: str, raw: str) -> ParsedPrediction:
    arr = _extract_json_array(raw)
    if arr is None or not isinstance(arr, list):
        return ParsedPrediction(
            sample_id, domain, topology, Mode.TRAJECTORY.value,
            parse_valid=False, schema_valid=False, failure_reason="invalid_json",
        )
    if not all(isinstance(x, str) for x in arr):
        # Tolerate non-string entries by stringifying, but mark schema invalid.
        node_names = [str(x) for x in arr]
        schema_valid = False
    else:
        node_names = list(arr)
        schema_valid = True
    edges = _adjacent_edges(node_names)
    return ParsedPrediction(
        sample_id, domain, topology, Mode.TRAJECTORY.value,
        parse_valid=True,
        schema_valid=schema_valid,
        pred_node_names=node_names,
        pred_edges=edges,
        pred_trajectory=node_names,
        links_valid=True,
        step_node_aligned=True,
        failure_reason=None if schema_valid else "non_string_entries",
    )


def parse_prediction(mode: Mode, sample_id: str, domain: str, topology: str, raw: str) -> ParsedPrediction:
    if mode == Mode.FULL_JSON:
        return parse_full_json(sample_id, domain, topology, raw)
    return parse_trajectory(sample_id, domain, topology, raw)
