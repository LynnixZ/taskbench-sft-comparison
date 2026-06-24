"""Pure metric functions reused from the official TaskBench ``evaluate.py``.

The functions ``sim``, ``create_cost_matrix``, ``compute_assignment_matrix``,
``matching``, ``ratio_levenshtein``, ``flatten`` and ``get_content_type`` are
**copied verbatim** from microsoft/JARVIS ``taskbench/evaluate.py`` (see the
vendored, unmodified copy in ``jarvis_evaluate.py``). Copying — rather than
importing — is only because the original module performs a now-removed
``from datasets import load_metric`` at import time; the algorithms here are
byte-for-byte identical, so results match the official evaluator exactly.

On top of those primitives we provide thin wrappers
(``node_prf_no_matching``, ``link_binary_f1``, ``edit_distance_score``) that
reproduce, line for line, the corresponding blocks of the official ``evaluate``
function so that Node-F1 / Link-F1 / Edit-Distance are computed identically.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_fscore_support as prfs

# The official evaluator does ``warnings.filterwarnings("ignore")``; we scope the
# (harmless) ill-defined precision/recall warnings from sparse label sets instead.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:  # The official code depends on python-Levenshtein.
    import Levenshtein
except ImportError:  # pragma: no cover - exercised only without the dep.
    Levenshtein = None


# --------------------------------------------------------------------------- #
# Verbatim primitives from official evaluate.py
# --------------------------------------------------------------------------- #
def sim(name_1, name_2):  # noqa: ANN001 - verbatim signature
    if name_1 == "<PAD>" or name_2 == "<PAD>":
        return 0
    return 1 if name_1 == name_2 else 0


def create_cost_matrix(graph_1, graph_2):  # noqa: ANN001 - verbatim
    nodes_1 = graph_1["nodes"]
    nodes_2 = graph_2["nodes"]

    num_nodes_1 = len(nodes_1)
    num_nodes_2 = len(nodes_2)

    nodes_similarity_matrix = np.zeros((num_nodes_1, num_nodes_2))

    for i, node_1 in enumerate(graph_1["nodes"]):
        for j, node_2 in enumerate(graph_2["nodes"]):
            nodes_similarity_matrix[i, j] = sim(node_1, node_2)

    links_similarity_matrix = np.zeros((num_nodes_1, num_nodes_2))
    for link_1 in graph_1["links"]:
        for link_2 in graph_2["links"]:
            if link_1["source"] == link_2["source"] and link_1["target"] == link_2["target"]:
                try:
                    i_index_1 = nodes_1.index(link_1["source"])
                    i_index_2 = nodes_2.index(link_2["source"])
                    j_index_1 = nodes_1.index(link_1["target"])
                    j_index_2 = nodes_2.index(link_2["target"])
                except ValueError:
                    continue
                links_similarity_matrix[i_index_1, i_index_2] += 1
                links_similarity_matrix[j_index_1, j_index_2] += 1

    cost_matrix = 2 - nodes_similarity_matrix - 0.5 * links_similarity_matrix
    return cost_matrix


def compute_assignment_matrix(graph_1, graph_2):  # noqa: ANN001 - verbatim
    cost_matrix = create_cost_matrix(graph_1, graph_2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind, cost_matrix[row_ind, col_ind].sum()


def matching(graph_1, graph_2):  # noqa: ANN001 - verbatim
    indices_1, indices_2, total_cost = compute_assignment_matrix(graph_1, graph_2)
    return indices_1, indices_2


def ratio_levenshtein(x, y):  # noqa: ANN001 - verbatim
    assert len(x) == len(y)
    n = len(x)
    total = 0
    for i in range(n):
        total += Levenshtein.ratio(x[i], y[i])
    return total / n


def flatten(gt, pred, types=None):  # noqa: ANN001 - verbatim
    assert len(gt) == len(pred)

    gt_flat = []
    pred_flat = []

    for (sample_gt, sample_pred) in zip(gt, pred):
        union = set()

        union.update(sample_gt)
        union.update(sample_pred)

        for s in union:
            if types:
                if s in types:
                    if s in sample_gt:
                        gt_flat.append(types.index(s) + 1)
                    else:
                        gt_flat.append(0)

                    if s in sample_pred:
                        pred_flat.append(types.index(s) + 1)
                    else:
                        pred_flat.append(0)
                else:
                    gt_flat.append(0)
                    pred_flat.append(0)
            else:
                if s in sample_gt:
                    gt_flat.append(1)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    pred_flat.append(1)
                else:
                    pred_flat.append(0)
    return gt_flat, pred_flat


def get_content_type(content):  # noqa: ANN001 - verbatim
    content = content.strip("'")
    assert isinstance(content, str), content
    for ext in ["jpg", "png", "jpeg", "gif", "bmp", "tiff", "svg", "ico"]:
        if "." + ext in content:
            return "image"
    for ext in ["mp3", "wav", "wma", "ogg", "aac", "flac", "aiff", "au"]:
        if "." + ext in content:
            return "audio"
    for ext in ["mp4", "avi", "mov", "flv", "wmv", "mkv", "webm", "m4v", "mpg", "mpeg"]:
        if "." + ext in content:
            return "video"
    return "text"


# --------------------------------------------------------------------------- #
# Thin wrappers reproducing the official ``evaluate()`` metric blocks
# --------------------------------------------------------------------------- #
def node_prf_no_matching(
    label_names: Sequence[Sequence[str]],
    prediction_names: Sequence[Sequence[str]],
    tool_names: List[str],
) -> dict:
    """Official set-based Node P/R/F1 (the "[ No Matching ]" block).

    ``tool_names`` is the ordered list of canonical tool names (i.e. the
    ``types_name`` of the official evaluator). Names must already be in the same
    spelling used by the official code (resource domains: spaces, not
    underscores).
    """
    types = list(range(1, len(tool_names) + 1))
    types_name = list(tool_names)
    gt_flat, pred_flat = flatten(label_names, prediction_names, types=types_name)

    micro = prfs(gt_flat, pred_flat, labels=types, average="micro", zero_division=0)[:-1]
    macro = prfs(gt_flat, pred_flat, labels=types, average="macro", zero_division=0)[:-1]
    return {
        "node_micro_precision_no_matching": float(micro[0]),
        "node_micro_recall_no_matching": float(micro[1]),
        "node_micro_f1_no_matching": float(micro[2]),
        "node_macro_precision_no_matching": float(macro[0]),
        "node_macro_recall_no_matching": float(macro[1]),
        "node_macro_f1_no_matching": float(macro[2]),
    }


def link_binary_f1(
    label_links: Sequence[Sequence[Tuple[str, str]]],
    prediction_links: Sequence[Sequence[Tuple[str, str]]],
) -> float:
    """Official Link (edge) binary F1 (the "link" metric block)."""
    gt_flat, pred_flat = flatten(label_links, prediction_links)
    if not gt_flat:
        # No edges present or predicted in ANY sample (e.g. an all-node group):
        # edge F1 is undefined -> 0.0 (and avoids prfs erroring on empty input).
        return 0.0
    micro = prfs(gt_flat, pred_flat, average="binary", zero_division=0)[:-1]
    return float(micro[-1])


def edit_distance_score(
    label_int_seqs: Sequence[Sequence[int]],
    prediction_int_seqs: Sequence[Sequence[int]],
) -> float:
    """Official edit-distance score: ``1 - ratio_levenshtein`` (higher better)."""
    ed = ratio_levenshtein(prediction_int_seqs, label_int_seqs)
    return float(1 - ed)
