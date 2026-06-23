"""Full-JSON-specific metrics (Mode A only).

These supplement — but never replace — the common planning metrics. They cover
task-step ROUGE, parameter name/value F1 (reusing the official argument-flatten +
binary-PRFS approach and ``get_content_type``), exact-JSON match, and the various
JSON/schema/alignment validity rates.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import precision_recall_fscore_support as prfs

from taskbench_sft.eval.parse import ParsedPrediction
from taskbench_sft.official.evaluate_lib import flatten, get_content_type
from taskbench_sft.schema import DependencyType, GoldSample
from taskbench_sft.targets import full_json_object

try:
    from rouge_score import rouge_scorer

    _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
except Exception:  # pragma: no cover
    _ROUGE = None


def _arg_identifiers(
    task: str, arguments: List[Any], dependency_type: DependencyType
) -> Tuple[List[str], List[str]]:
    """Return (name-ids, name-value-ids) for one node's arguments.

    temporal: arguments are ``{name, value}`` dicts (used directly).
    resource: arguments are strings; the name is derived with the official
    ``get_content_type`` (resource references keep the tag as value, name
    ``"resource"``) — a deterministic normalization, never a guess at gold.
    """
    names: List[str] = []
    name_values: List[str] = []
    for arg in arguments:
        if dependency_type == DependencyType.TEMPORAL and isinstance(arg, dict):
            an = str(arg.get("name", ""))
            av = str(arg.get("value", ""))
        elif isinstance(arg, dict):
            an = str(arg.get("name", next(iter(arg.keys()), "")))
            av = str(arg.get("value", next(iter(arg.values()), "")))
        else:
            s = str(arg)
            if s.startswith("<node-") or s.startswith("<output_of"):
                an, av = "resource", s
            else:
                an, av = get_content_type(s), s
        names.append(f"{task}-{an}")
        name_values.append(f"{task}-{an}-{av}")
    return names, name_values


def _binary_f1(gold: List[List[str]], pred: List[List[str]]) -> float:
    gt_flat, pred_flat = flatten(gold, pred)
    if not gt_flat:
        return 0.0
    micro = prfs(gt_flat, pred_flat, average="binary", zero_division=0)[:-1]
    return float(micro[-1])


def compute_fulljson_metrics(
    golds: List[GoldSample],
    preds: List[ParsedPrediction],
    compute_rouge: bool = True,
) -> Dict[str, float]:
    """Compute Mode-A-only metrics over aligned gold/prediction lists."""
    n = len(golds)
    if n == 0:
        return {"n_samples": 0}

    # ---- exact JSON match + validity rates ----
    exact = 0
    for g, p in zip(golds, preds):
        if not p.parse_valid or not p.schema_valid:
            continue
        gold_obj = full_json_object(g)
        pred_obj = {
            "task_steps": p.pred_task_steps or [],
            "task_nodes": [
                {"task": t, "arguments": a}
                for t, a in zip(p.pred_node_names, p.pred_arguments or [])
            ],
            "task_links": [{"source": s, "target": t} for s, t in p.pred_edges],
        }
        if gold_obj == pred_obj:
            exact += 1

    parse_valid_rate = sum(1 for p in preds if p.parse_valid) / n
    schema_valid_rate = sum(1 for p in preds if p.schema_valid) / n
    step_align_rate = sum(1 for p in preds if p.step_node_aligned) / n
    link_valid_rate = sum(1 for p in preds if p.links_valid) / n

    # ---- parameter name / value F1 ----
    gold_name_ids: List[List[str]] = []
    pred_name_ids: List[List[str]] = []
    gold_nameval_ids: List[List[str]] = []
    pred_nameval_ids: List[List[str]] = []
    for g, p in zip(golds, preds):
        dt = g.dependency_type
        g_names, g_nv = [], []
        for node in g.task_nodes:
            a, b = _arg_identifiers(node.task, node.arguments, dt)
            g_names += a
            g_nv += b
        p_names, p_nv = [], []
        for task, args in zip(p.pred_node_names, p.pred_arguments or []):
            a, b = _arg_identifiers(task, args, dt)
            p_names += a
            p_nv += b
        gold_name_ids.append(g_names)
        pred_name_ids.append(p_names)
        gold_nameval_ids.append(g_nv)
        pred_nameval_ids.append(p_nv)

    param_name_f1 = _binary_f1(gold_name_ids, pred_name_ids)
    param_value_f1 = _binary_f1(gold_nameval_ids, pred_nameval_ids)

    out: Dict[str, float] = {
        "n_samples": n,
        "exact_json_match": exact / n,
        "json_parse_validity": parse_valid_rate,
        "schema_validity": schema_valid_rate,
        "step_node_alignment_validity": step_align_rate,
        "link_validity": link_valid_rate,
        "parameter_name_f1": param_name_f1,
        "parameter_value_f1": param_value_f1,
    }

    # ---- task-step ROUGE ----
    if compute_rouge and _ROUGE is not None:
        r1, rl, cnt = 0.0, 0.0, 0
        for g, p in zip(golds, preds):
            if not p.parse_valid or p.pred_task_steps is None:
                continue
            ref = "\n".join(g.task_steps)
            hyp = "\n".join(p.pred_task_steps)
            scores = _ROUGE.score(ref, hyp)
            r1 += scores["rouge1"].fmeasure
            rl += scores["rougeL"].fmeasure
            cnt += 1
        out["task_step_rouge1"] = (r1 / cnt) if cnt else 0.0
        out["task_step_rougeL"] = (rl / cnt) if cnt else 0.0
        out["rouge_support"] = cnt

    return out
