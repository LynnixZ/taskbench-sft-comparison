"""Serialize a tool catalog for inclusion in the prompt.

The same serialization is used for both modes so the only thing that differs
between Mode A and Mode B is the *output* task description, never the catalog.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from taskbench_sft.schema import DependencyType, ToolCatalog


def _tool_entry(tool, dependency_type: DependencyType, include_io_types: bool) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"id": tool.id, "description": tool.desc}
    if not include_io_types:
        return entry
    if dependency_type == DependencyType.RESOURCE:
        if tool.input_type is not None:
            entry["input-type"] = tool.input_type
        if tool.output_type is not None:
            entry["output-type"] = tool.output_type
    else:
        # temporal: expose the parameter names so the model can fill arguments.
        entry["parameters"] = tool.parameter_names
    return entry


def serialize_catalog(catalog: ToolCatalog, include_io_types: bool = True) -> str:
    """Serialize the catalog as a JSON array of tool entries (one per line)."""
    entries = [
        _tool_entry(t, catalog.dependency_type, include_io_types) for t in catalog.tools
    ]
    # One tool per line keeps it readable and tokenizes consistently.
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    return "[\n" + ",\n".join("  " + ln for ln in lines) + "\n]"
