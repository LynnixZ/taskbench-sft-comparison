"""Target generation tests (Mode A node target + Mode B trajectory)."""
from __future__ import annotations

import json

from tests.conftest import make_chain_sample

from taskbench_sft.schema import Topology
from taskbench_sft.targets import (
    build_target,
    full_json_object,
    trajectory_list,
)
from taskbench_sft.schema import Mode
from taskbench_sft.data.topology import annotate_sample


def test_node_target_generated_correctly():
    """(1) A single-node sample produces the correct full-JSON + trajectory target."""
    s = make_chain_sample("n1", names=["Translation"], links=[], topology=Topology.SINGLE)
    annotate_sample(s)
    obj = full_json_object(s)
    assert obj["task_nodes"] == [{"task": "Translation", "arguments": []}]
    assert obj["task_links"] == []  # single-tool task -> empty links
    assert len(obj["task_steps"]) == 1
    # Trajectory for a node is a length-1 array.
    assert trajectory_list(s) == ["Translation"]
    assert json.loads(build_target(s, Mode.TRAJECTORY)) == ["Translation"]


def test_chain_full_json_target_matches_gold():
    s = make_chain_sample("c1", names=["A", "B", "C"])
    annotate_sample(s)
    obj = full_json_object(s)
    assert [n["task"] for n in obj["task_nodes"]] == ["A", "B", "C"]
    assert obj["task_links"] == [
        {"source": "A", "target": "B"},
        {"source": "B", "target": "C"},
    ]


def test_chain_trajectory_target_is_ordered_array():
    s = make_chain_sample("c2", names=["A", "B", "C"], links=[("B", "C"), ("A", "B")])
    annotate_sample(s)
    assert json.loads(build_target(s, Mode.TRAJECTORY)) == ["A", "B", "C"]
