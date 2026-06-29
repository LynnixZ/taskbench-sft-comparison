"""Common planning metrics computable for BOTH modes.

These are the metrics on which the core Full-JSON vs Trajectory comparison is
based: Node F1, Edge F1, NED, Trajectory Exact Match, and Hallucination Rate,
plus tool-count and prefix diagnostics. The official set-based Node F1 and Link
F1 reuse :mod:`taskbench_sft.official.evaluate_lib`; a multiset-aware F1 is added
on top.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

from taskbench_sft.eval.parse import ParsedPrediction
from taskbench_sft.official.evaluate_lib import link_binary_f1, node_prf_no_matching
from taskbench_sft.schema import GoldSample, ToolCatalog


# --------------------------------------------------------------------------- #
# Sequence helpers
# --------------------------------------------------------------------------- #
def list_edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein edit distance between two sequences of tokens (DP)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def normalized_edit_distance(pred: Sequence[str], gold: Sequence[str]) -> float:
    """NED = edit_distance / max(len(pred), len(gold), 1). Lower is better."""
    denom = max(len(pred), len(gold), 1)
    return list_edit_distance(pred, gold) / denom


def longest_common_prefix_len(a: Sequence[str], b: Sequence[str]) -> int:
    k = 0
    for x, y in zip(a, b):
        if x == y:
            k += 1
        else:
            break
    return k


# --------------------------------------------------------------------------- #
# Multiset-aware node F1
# --------------------------------------------------------------------------- #
def multiset_node_f1(
    gold_names: Sequence[Sequence[str]], pred_names: Sequence[Sequence[str]]
) -> Dict[str, float]:
    """Micro multiset-aware Node P/R/F1 (repeated tools counted with Counter)."""
    tp = 0
    pred_total = 0
    gold_total = 0
    for g, p in zip(gold_names, pred_names):
        gc, pc = Counter(g), Counter(p)
        for tool, c in pc.items():
            tp += min(c, gc.get(tool, 0))
        pred_total += sum(pc.values())
        gold_total += sum(gc.values())
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "multiset_node_precision": precision,
        "multiset_node_recall": recall,
        "multiset_node_f1": f1,
    }


# --------------------------------------------------------------------------- #
# Aggregate common metrics over aligned (gold, pred) lists
# --------------------------------------------------------------------------- #
def compute_common_metrics(
    golds: List[GoldSample],
    preds: List[ParsedPrediction],
    catalogs: Dict[str, ToolCatalog],
) -> Dict[str, float]:
    """Compute all common metrics over aligned gold/prediction lists."""
    assert len(golds) == len(preds)
    n = len(golds)
    if n == 0:
        return {"n_samples": 0}

    gold_names = [g.node_names for g in golds]
    pred_names = [p.pred_node_names for p in preds]
    gold_traj = [g.trajectory or g.node_names for g in golds]
    pred_traj = [p.pred_trajectory for p in preds]
    gold_edges = [[(l.source, l.target) for l in g.task_links] for g in golds]
    pred_edges = [list(p.pred_edges) for p in preds]

    # ---- Node F1 (official set-based) ----
    tool_names: List[str] = []
    seen = set()
    for g in golds:
        for tid in catalogs[g.domain].tool_ids:
            if tid not in seen:
                seen.add(tid)
                tool_names.append(tid)
    node_official = node_prf_no_matching(gold_names, pred_names, tool_names)

    # ---- Node F1 (multiset-aware) ----
    node_multiset = multiset_node_f1(gold_names, pred_names)

    # ---- Edge F1 (link-based, official) ----
    edge_f1 = link_binary_f1(gold_edges, pred_edges)

    # ---- Adjacent-edge F1 from recovered trajectories (unified) ----
    gold_adj = [[(t[i], t[i + 1]) for i in range(len(t) - 1)] for t in gold_traj]
    pred_adj = [[(t[i], t[i + 1]) for i in range(len(t) - 1)] for t in pred_traj]
    adjacent_edge_f1 = link_binary_f1(gold_adj, pred_adj)

    # ---- NED + trajectory exact match + prefix diagnostics ----
    ned_vals = [normalized_edit_distance(p, g) for p, g in zip(pred_traj, gold_traj)]
    exact = [1.0 if list(p) == list(g) else 0.0 for p, g in zip(pred_traj, gold_traj)]
    lcp = [longest_common_prefix_len(p, g) for p, g in zip(pred_traj, gold_traj)]
    prefix_acc = [
        (l / len(g)) if len(g) else 1.0 for l, g in zip(lcp, gold_traj)
    ]

    # ---- Tool count metrics ----
    count_match = [1.0 if len(p) == len(g) else 0.0 for p, g in zip(pred_names, gold_names)]
    count_mae = [abs(len(p) - len(g)) for p, g in zip(pred_names, gold_names)]
    over = [1.0 if len(p) > len(g) else 0.0 for p, g in zip(pred_names, gold_names)]
    under = [1.0 if len(p) < len(g) else 0.0 for p, g in zip(pred_names, gold_names)]
    length_match = [1.0 if len(p) == len(g) else 0.0 for p, g in zip(pred_traj, gold_traj)]

    # ---- Hallucination (off-catalog predicted tools) ----
    hall_tokens = 0
    total_pred_tokens = 0
    samples_with_hall = 0
    for g, p in zip(golds, preds):
        cat = catalogs[g.domain]
        sample_hall = 0
        for name in p.pred_node_names:
            total_pred_tokens += 1
            if name not in cat:
                hall_tokens += 1
                sample_hall += 1
        if sample_hall > 0:
            samples_with_hall += 1

    # ---- Validity rates ----
    parse_valid_rate = sum(1.0 for p in preds if p.parse_valid) / n
    schema_valid_rate = sum(1.0 for p in preds if p.schema_valid) / n

    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_samples": n,
        # node
        "node_f1": node_official["node_micro_f1_no_matching"],
        "node_micro_precision": node_official["node_micro_precision_no_matching"],
        "node_micro_recall": node_official["node_micro_recall_no_matching"],
        "node_macro_f1": node_official["node_macro_f1_no_matching"],
        **node_multiset,
        # edges
        "edge_f1": edge_f1,
        "adjacent_edge_f1": adjacent_edge_f1,
        # sequence
        "ned": mean(ned_vals),
        "trajectory_exact_match": mean(exact),
        "sequence_exact_match": mean(exact),
        "exact_match": mean(exact),  # alias: ordered tool-sequence exact match (same def as GTool's EM)
        "length_accuracy": mean(length_match),
        "average_correct_prefix_length": mean([float(x) for x in lcp]),
        "prefix_accuracy": mean(prefix_acc),
        # counts
        "tool_count_accuracy": mean(count_match),
        "tool_count_mae": mean([float(x) for x in count_mae]),
        "over_selection_rate": mean(over),
        "under_selection_rate": mean(under),
        # hallucination + validity
        "hallucinated_tool_rate": (hall_tokens / total_pred_tokens) if total_pred_tokens else 0.0,
        "samples_with_hallucination_rate": samples_with_hall / n,
        "parse_valid_rate": parse_valid_rate,
        "schema_valid_rate": schema_valid_rate,
    }
