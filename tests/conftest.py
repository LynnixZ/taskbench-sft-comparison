"""Shared pytest fixtures.

A lightweight fake tokenizer keeps the tokenization tests fast and offline (no
model download). It tokenizes on whitespace into deterministic integer IDs.
"""
from __future__ import annotations

from typing import Dict, List

import pytest

from taskbench_sft.schema import (
    DependencyType,
    GoldSample,
    TaskLink,
    TaskNode,
    Tool,
    ToolCatalog,
    Topology,
)


class FakeTokenizer:
    chat_template = None
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._next = 10

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = self._next
            self._next += 1
        return self._vocab[tok]

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = [self._id(t) for t in text.split()]
        return {"input_ids": ids}


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def resource_catalog() -> ToolCatalog:
    tools = [
        Tool(id="Image-to-Text", desc="extract text", input_type=["image"], output_type=["text"]),
        Tool(id="Translation", desc="translate", input_type=["text"], output_type=["text"]),
        Tool(id="Text-to-Speech", desc="tts", input_type=["text"], output_type=["audio"]),
        Tool(id="Image Classification", desc="classify", input_type=["image"], output_type=["text"]),
    ]
    return ToolCatalog(domain="data_huggingface", dependency_type=DependencyType.RESOURCE, tools=tools)


def make_chain_sample(
    sample_id: str = "c1",
    names: List[str] = None,
    links: List[tuple] = None,
    topology: Topology = Topology.CHAIN,
    domain: str = "data_huggingface",
    dependency_type: DependencyType = DependencyType.RESOURCE,
) -> GoldSample:
    names = names or ["Image-to-Text", "Translation", "Text-to-Speech"]
    if links is None:
        links = [(names[i], names[i + 1]) for i in range(len(names) - 1)]
    return GoldSample(
        id=sample_id,
        domain=domain,
        dependency_type=dependency_type,
        topology=topology,
        n_tools=len(names),
        user_request="do the thing",
        task_steps=[f"Step {i+1}" for i in range(len(names))],
        task_nodes=[TaskNode(task=n, arguments=[]) for n in names],
        task_links=[TaskLink(source=s, target=t) for s, t in links],
        chain_length_bucket="chain_length_3",
    )


@pytest.fixture
def chain_sample() -> GoldSample:
    return make_chain_sample()
