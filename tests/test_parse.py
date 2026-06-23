"""Prediction parsing tests (round-trip parse for both modes)."""
from __future__ import annotations

from tests.conftest import make_chain_sample

from taskbench_sft.data.topology import annotate_sample
from taskbench_sft.eval.parse import parse_full_json, parse_trajectory
from taskbench_sft.schema import Mode
from taskbench_sft.targets import build_target


def test_full_json_round_trip_parse():
    """(6) A canonical full-JSON target parses back to the same plan."""
    s = make_chain_sample("c1", names=["A", "B", "C"])
    annotate_sample(s)
    raw = build_target(s, Mode.FULL_JSON)
    pp = parse_full_json("c1", "data_huggingface", "chain", raw)
    assert pp.parse_valid and pp.schema_valid
    assert pp.pred_node_names == ["A", "B", "C"]
    assert pp.pred_edges == [("A", "B"), ("B", "C")]
    assert pp.pred_trajectory == ["A", "B", "C"]


def test_trajectory_round_trip_parse():
    """(7) A canonical trajectory target parses back to the same list."""
    s = make_chain_sample("c1", names=["A", "B", "C"])
    annotate_sample(s)
    raw = build_target(s, Mode.TRAJECTORY)
    pp = parse_trajectory("c1", "data_huggingface", "chain", raw)
    assert pp.parse_valid and pp.schema_valid
    assert pp.pred_node_names == ["A", "B", "C"]
    assert pp.pred_trajectory == ["A", "B", "C"]
    assert pp.pred_edges == [("A", "B"), ("B", "C")]


def test_full_json_invalid_is_recorded_not_fixed():
    pp = parse_full_json("x", "data_huggingface", "chain", "not json at all")
    assert not pp.parse_valid
    assert pp.failure_reason == "invalid_json"


def test_trajectory_handles_markdown_wrapping():
    raw = 'Here you go:\n```json\n["A", "B"]\n```'
    pp = parse_trajectory("x", "data_huggingface", "chain", raw)
    assert pp.parse_valid
    assert pp.pred_node_names == ["A", "B"]


def test_full_json_with_prose_prefix():
    raw = 'Sure! {"task_steps": ["s"], "task_nodes": [{"task": "A", "arguments": []}], "task_links": []}'
    pp = parse_full_json("x", "data_huggingface", "single", raw)
    assert pp.parse_valid and pp.schema_valid
    assert pp.pred_node_names == ["A"]
