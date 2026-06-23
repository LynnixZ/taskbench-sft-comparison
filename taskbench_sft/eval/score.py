"""Composite ``validation_common_score`` used for checkpoint selection.

The score combines the common planning metrics that are shared by Node and Chain
samples. The weights are fully configurable (see
``CheckpointSelectionConfig``).
"""
from __future__ import annotations

from typing import Dict

from taskbench_sft.config import CheckpointSelectionConfig


def common_score(metrics: Dict[str, float], weights: CheckpointSelectionConfig) -> float:
    """Weighted combination of node_f1, edge_f1, sequence_exact_match, parse rate."""
    return (
        weights.node_f1 * metrics.get("node_f1", 0.0)
        + weights.edge_f1 * metrics.get("edge_f1", 0.0)
        + weights.sequence_exact_match * metrics.get("sequence_exact_match", 0.0)
        + weights.parse_valid_rate * metrics.get("parse_valid_rate", 0.0)
    )
