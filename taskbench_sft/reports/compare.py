"""Build the final cross-run comparison table.

The core comparison is restricted to the metrics that are *commonly computable*
for both modes: Node F1, Edge F1, NED, Trajectory Exact Match, and Hallucination
Rate (plus parse validity). Full-JSON-specific metrics are reported separately
and never used as the head-to-head comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CORE_METRICS = [
    ("node_f1", "Node F1"),
    ("edge_f1", "Edge F1"),
    ("ned", "NED"),
    ("trajectory_exact_match", "Traj EM"),
    ("hallucinated_tool_rate", "Halluc."),
    ("parse_valid_rate", "Parse OK"),
]


def _parse_report_args(reports: List[str]) -> List[Tuple[str, str]]:
    """Parse ``name=path`` CLI pairs into (name, path)."""
    out = []
    for item in reports:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        out.append((name, path))
    return out


def _overall(report: Dict[str, Any]) -> Dict[str, Any]:
    return report.get("groups", {}).get("overall", {}).get("overall", {})


def build_comparison_table(reports: List[str], out: Optional[str] = None) -> str:
    """Build a markdown comparison table from ``name=path`` metric reports."""
    pairs = _parse_report_args(reports)
    loaded: List[Tuple[str, Dict[str, Any]]] = []
    for name, path in pairs:
        with open(path, "r", encoding="utf-8") as f:
            loaded.append((name, json.load(f)))

    header = "| Run | " + " | ".join(label for _, label in CORE_METRICS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in CORE_METRICS) + " |"
    rows = [header, sep]
    json_summary: Dict[str, Any] = {}
    for name, report in loaded:
        ov = _overall(report)
        json_summary[name] = {k: ov.get(k) for k, _ in CORE_METRICS}
        cells = []
        for key, _ in CORE_METRICS:
            val = ov.get(key)
            cells.append(f"{val:.3f}" if isinstance(val, (int, float)) else "—")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")

    table = "\n".join(rows)
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table + "\n\n")
            f.write("```json\n")
            f.write(json.dumps(json_summary, ensure_ascii=False, indent=2))
            f.write("\n```\n")
    return table
