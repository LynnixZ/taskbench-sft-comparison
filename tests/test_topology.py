"""Topology tests: chain recovery, DAG/disconnected exclusion, repeated names."""
from __future__ import annotations

from tests.conftest import make_chain_sample

from taskbench_sft.data.topology import (
    annotate_sample,
    is_simple_path,
    recover_index_edges,
    topological_order,
)
from taskbench_sft.schema import Topology


def test_chain_recovered_from_unordered_links():
    """(2) A chain is recovered in execution order from UNORDERED links."""
    names = ["A", "B", "C", "D"]
    # Links provided out of order; the recovered trajectory must be A,B,C,D.
    links = [("C", "D"), ("A", "B"), ("B", "C")]
    s = make_chain_sample("chain1", names=names, links=links)
    annotate_sample(s)
    assert s.is_usable
    assert s.trajectory == ["A", "B", "C", "D"]


def test_dag_excluded():
    """(3) DAG-typed samples are excluded."""
    s = make_chain_sample("dag1", names=["A", "B", "C"], topology=Topology.DAG)
    annotate_sample(s)
    assert not s.is_usable
    assert s.exclusion_reason == "dag_excluded"
    assert s.trajectory is None


def test_disconnected_graph_excluded():
    """(4) A 'chain' whose links leave a node disconnected is excluded."""
    # Three nodes but only one edge -> node C is isolated (not a simple path).
    s = make_chain_sample("disc1", names=["A", "B", "C"], links=[("A", "B")])
    annotate_sample(s)
    assert not s.is_usable
    assert s.exclusion_reason == "not_simple_connected_path"


def test_branching_dag_shaped_chain_excluded():
    """A 'chain' that actually branches (A->B, A->C) is excluded, not linearized."""
    s = make_chain_sample("branch1", names=["A", "B", "C"], links=[("A", "B"), ("A", "C")])
    annotate_sample(s)
    assert not s.is_usable
    assert s.exclusion_reason == "not_simple_connected_path"


def test_repeated_tool_names_not_deduped_at_index_level():
    """(5) Repeated tool names are preserved (no dedup) when edges are by index."""
    names = ["A", "B", "A"]  # legitimately repeats tool A
    idx_edges = [(0, 1), (1, 2)]
    assert is_simple_path(3, idx_edges)
    order = topological_order(3, idx_edges)
    assert order == [0, 1, 2]
    trajectory = [names[i] for i in order]
    assert trajectory == ["A", "B", "A"]  # duplicate retained


def test_repeated_names_in_links_are_excluded_not_guessed():
    """Name-based recovery refuses to guess on ambiguous repeated names."""
    s = make_chain_sample("rep1", names=["A", "B", "A"], links=[("A", "B"), ("B", "A")])
    edges = recover_index_edges(s)
    assert edges is None  # ambiguous, so excluded rather than silently guessed
    annotate_sample(s)
    assert not s.is_usable
    assert s.exclusion_reason == "ambiguous_repeated_names"


def test_single_node_trajectory():
    s = make_chain_sample("s1", names=["A"], links=[], topology=Topology.SINGLE)
    annotate_sample(s)
    assert s.is_usable
    assert s.trajectory == ["A"]
