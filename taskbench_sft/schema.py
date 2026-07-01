"""Core data structures mirroring the official TaskBench gold schema.

These dataclasses are a thin, typed normalization layer over the *official*
``data.json`` / ``tool_desc.json`` files. We never invent or modify fields of the
gold labels; we only parse the official string-encoded fields into typed objects.

Official field mapping (shipped ``data.json`` -> canonical TaskBench schema used
by ``evaluate.py`` / ``inference.py``)::

    instruction              -> user_request
    json.loads(tool_steps)   -> task_steps   : List[str]
    json.loads(tool_nodes)   -> task_nodes   : List[{task, arguments}]
    json.loads(tool_links)   -> task_links   : List[{source, target}]
    type                     -> topology      ("single" | "chain" | "dag")

Dependency types (also from the official code):

* ``resource``  – data_huggingface, data_multimedia. Node arguments are a list of
  strings; ``<node-j>`` tags encode resource dependencies. Tool names contain
  spaces.
* ``temporal``  – data_dailylifeapis. Node arguments are ``{name, value}`` dicts;
  dependencies are given explicitly via ``task_links``. Tool names use
  underscores.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


class DependencyType(str, enum.Enum):
    """How tool dependencies are encoded in a domain (from official TaskBench)."""

    RESOURCE = "resource"
    TEMPORAL = "temporal"


class Topology(str, enum.Enum):
    """Official sample topology label (the ``type`` field of ``data.json``)."""

    SINGLE = "single"
    CHAIN = "chain"
    DAG = "dag"


class Mode(str, enum.Enum):
    """SFT target format under comparison."""

    FULL_JSON = "full_json"
    TRAJECTORY = "trajectory"


# Mapping from domain directory name -> dependency type, per official code.
DOMAIN_DEPENDENCY: Dict[str, DependencyType] = {
    "data_huggingface": DependencyType.RESOURCE,
    "data_multimedia": DependencyType.RESOURCE,
    "data_dailylifeapis": DependencyType.TEMPORAL,
    # GNN4Plan extra benchmarks: explicit {source,target} task_links (like dailylife) -> TEMPORAL.
    "data_ultratool": DependencyType.TEMPORAL,
    "data_tmdb": DependencyType.TEMPORAL,
}


# An argument is either a raw string (resource domains) or a {name, value} dict
# (temporal domain). We keep the gold structure untouched.
Argument = Union[str, Dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    """A single tool from ``tool_desc.json``."""

    id: str
    desc: str
    # resource domains:
    input_type: Optional[List[str]] = None
    output_type: Optional[List[str]] = None
    # temporal domain:
    parameters: Optional[List[Dict[str, Any]]] = None

    @property
    def parameter_names(self) -> List[str]:
        if self.parameters is None:
            return []
        return [p["name"] for p in self.parameters]


@dataclass
class ToolCatalog:
    """The catalog of tools available for one domain (from ``tool_desc.json``)."""

    domain: str
    dependency_type: DependencyType
    tools: List[Tool]

    def __post_init__(self) -> None:
        self._by_id: Dict[str, Tool] = {t.id: t for t in self.tools}
        # Resource-dependency tool names are compared with underscores replaced
        # by spaces in the official evaluator; we index both spellings so the
        # rest of the pipeline can look a tool up regardless of spelling.
        self._normalized: Dict[str, Tool] = {}
        for t in self.tools:
            self._normalized[t.id] = t
            self._normalized[t.id.replace("_", " ")] = t

    @property
    def tool_ids(self) -> List[str]:
        return [t.id for t in self.tools]

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, tool_id: str) -> bool:
        return tool_id in self._normalized

    def get(self, tool_id: str) -> Optional[Tool]:
        return self._normalized.get(tool_id)


@dataclass
class TaskNode:
    """A node of the gold plan: a tool invocation with its arguments."""

    task: str
    arguments: List[Argument] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "arguments": self.arguments}


@dataclass
class TaskLink:
    """A directed dependency edge between two tools."""

    source: str
    target: str

    def to_dict(self) -> Dict[str, str]:
        return {"source": self.source, "target": self.target}


@dataclass
class GoldSample:
    """A normalized TaskBench gold sample.

    ``trajectory`` is the canonical execution-ordered list of tool names. It is
    derived from the gold links by :mod:`taskbench_sft.data.topology` (and is
    ``None`` for samples whose topology is excluded, e.g. DAG or disconnected).
    """

    id: str
    domain: str
    dependency_type: DependencyType
    topology: Topology
    n_tools: int
    user_request: str
    task_steps: List[str]
    task_nodes: List[TaskNode]
    task_links: List[TaskLink]
    seed: Optional[int] = None
    trajectory: Optional[List[str]] = None
    chain_length_bucket: Optional[str] = None
    # Whether this sample passed topology validation (path / single node).
    is_usable: bool = True
    exclusion_reason: Optional[str] = None

    @property
    def node_names(self) -> List[str]:
        return [n.task for n in self.task_nodes]

    @property
    def stratify_key(self) -> str:
        """Composite stratification key: domain x topology x length bucket."""
        return f"{self.domain}|{self.topology.value}|{self.chain_length_bucket}"

    def gold_dict(self) -> Dict[str, Any]:
        """The gold object in the canonical TaskBench schema (for the evaluator)."""
        return {
            "id": self.id,
            "type": self.topology.value,
            "user_request": self.user_request,
            "task_steps": list(self.task_steps),
            "task_nodes": [n.to_dict() for n in self.task_nodes],
            "task_links": [l.to_dict() for l in self.task_links],
        }

    def to_record(self) -> Dict[str, Any]:
        """Full JSON-serializable record (for split JSONL files)."""
        return {
            "id": self.id,
            "domain": self.domain,
            "dependency_type": self.dependency_type.value,
            "topology": self.topology.value,
            "n_tools": self.n_tools,
            "user_request": self.user_request,
            "task_steps": list(self.task_steps),
            "task_nodes": [n.to_dict() for n in self.task_nodes],
            "task_links": [l.to_dict() for l in self.task_links],
            "seed": self.seed,
            "trajectory": list(self.trajectory) if self.trajectory is not None else None,
            "chain_length_bucket": self.chain_length_bucket,
            "is_usable": self.is_usable,
            "exclusion_reason": self.exclusion_reason,
        }

    @classmethod
    def from_record(cls, rec: Dict[str, Any]) -> "GoldSample":
        return cls(
            id=str(rec["id"]),
            domain=rec["domain"],
            dependency_type=DependencyType(rec["dependency_type"]),
            topology=Topology(rec["topology"]),
            n_tools=rec["n_tools"],
            user_request=rec["user_request"],
            task_steps=list(rec["task_steps"]),
            task_nodes=[TaskNode(task=n["task"], arguments=n.get("arguments", [])) for n in rec["task_nodes"]],
            task_links=[TaskLink(source=l["source"], target=l["target"]) for l in rec["task_links"]],
            seed=rec.get("seed"),
            trajectory=rec.get("trajectory"),
            chain_length_bucket=rec.get("chain_length_bucket"),
            is_usable=rec.get("is_usable", True),
            exclusion_reason=rec.get("exclusion_reason"),
        )


# Chain-length bucket names (used for stratification and grouped reporting).
BUCKET_NODE = "node"
BUCKET_LEN2 = "chain_length_2"
BUCKET_LEN3 = "chain_length_3"
BUCKET_LEN4_PLUS = "chain_length_4_plus"
BUCKET_DAG = "dag"

CHAIN_LENGTH_BUCKETS = [BUCKET_NODE, BUCKET_LEN2, BUCKET_LEN3, BUCKET_LEN4_PLUS, BUCKET_DAG]


def chain_length_bucket(topology: Topology, n_tools: int) -> str:
    """Bucket a sample by chain length for stratification / reporting."""
    if topology == Topology.DAG:
        return BUCKET_DAG  # branching graph -> its own stratum (not a chain length)
    if topology == Topology.SINGLE or n_tools <= 1:
        return BUCKET_NODE
    if n_tools == 2:
        return BUCKET_LEN2
    if n_tools == 3:
        return BUCKET_LEN3
    return BUCKET_LEN4_PLUS
