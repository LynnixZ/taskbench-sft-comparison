"""Metric hand-cases: Node/Edge F1, NED, multiset F1."""
from __future__ import annotations

import math

from taskbench_sft.eval.metrics_common import (
    multiset_node_f1,
    normalized_edit_distance,
    list_edit_distance,
)
from taskbench_sft.official.evaluate_lib import link_binary_f1, node_prf_no_matching


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_ned_identical_is_zero():
    """(10) NED of identical trajectories is 0."""
    assert normalized_edit_distance(["A", "B", "C"], ["A", "B", "C"]) == 0.0


def test_ned_one_substitution():
    """(10) One substitution over length 3 -> NED = 1/3."""
    assert approx(normalized_edit_distance(["A", "X", "C"], ["A", "B", "C"]), 1 / 3)


def test_ned_insertion():
    # pred shorter by one -> distance 1, max len 3.
    assert approx(normalized_edit_distance(["A", "B"], ["A", "B", "C"]), 1 / 3)


def test_list_edit_distance_basic():
    assert list_edit_distance(["A", "B", "C"], ["A", "C"]) == 1
    assert list_edit_distance([], ["A"]) == 1
    assert list_edit_distance(["A"], []) == 1


def test_official_node_f1_hand_case():
    """(10) gold={A,B,C}, pred={A,B}: precision=1, recall=2/3, micro F1=0.8."""
    tool_names = ["A", "B", "C"]
    res = node_prf_no_matching([["A", "B", "C"]], [["A", "B"]], tool_names)
    assert approx(res["node_micro_precision_no_matching"], 1.0)
    assert approx(res["node_micro_recall_no_matching"], 2 / 3)
    assert approx(res["node_micro_f1_no_matching"], 0.8)


def test_edge_f1_hand_case():
    """(10) gold edges {(A,B),(B,C)}, pred {(A,B)}: precision=1, recall=1/2, F1=2/3."""
    f1 = link_binary_f1([[("A", "B"), ("B", "C")]], [[("A", "B")]])
    assert approx(f1, 2 / 3)


def test_edge_f1_all_empty_no_crash():
    """All-node group (no edges anywhere) -> edge F1 = 0.0, not a crash on empty prfs."""
    assert link_binary_f1([[], [], []], [[], [], []]) == 0.0


def test_multiset_node_f1_counts_duplicates():
    """(5)+(10) Multiset F1 distinguishes [A,A,B] from [A,B] (no dedup)."""
    # gold has A twice; pred has A once -> tp=1(A)+1(B)=2, gold_total=3, pred_total=2
    res = multiset_node_f1([["A", "A", "B"]], [["A", "B"]])
    assert approx(res["multiset_node_precision"], 1.0)      # 2/2
    assert approx(res["multiset_node_recall"], 2 / 3)        # 2/3
    assert approx(res["multiset_node_f1"], 0.8)


def test_multiset_vs_set_difference():
    """Set-based F1 would treat [A,A] == [A]; multiset does not."""
    # pred predicts A twice, gold once: multiset precision penalizes the extra A.
    res = multiset_node_f1([["A"]], [["A", "A"]])
    assert approx(res["multiset_node_precision"], 0.5)  # 1 tp / 2 predicted
    assert approx(res["multiset_node_recall"], 1.0)
