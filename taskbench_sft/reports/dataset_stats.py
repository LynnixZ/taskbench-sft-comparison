"""Compute dataset statistics from the actual downloaded data.

We never hard-code paper sample counts; everything here is derived from the data
on disk so the numbers reflect exactly what was downloaded.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from taskbench_sft.schema import GoldSample, ToolCatalog


def compute_dataset_stats(
    all_samples: List[GoldSample],
    catalogs: Dict[str, ToolCatalog],
) -> Dict[str, Any]:
    """Return a nested statistics dict over all loaded samples."""
    by_domain: Dict[str, Any] = {}
    for domain, catalog in catalogs.items():
        domain_samples = [s for s in all_samples if s.domain == domain]
        usable = [s for s in domain_samples if s.is_usable and s.topology.value in ("single", "chain")]
        topo = Counter(s.topology.value for s in domain_samples)
        usable_topo = Counter(s.topology.value for s in usable)
        buckets = Counter(s.chain_length_bucket for s in usable)
        excl = Counter(s.exclusion_reason for s in domain_samples if not s.is_usable)
        tool_freq: Counter = Counter()
        for s in usable:
            tool_freq.update(s.node_names)
        covered = set(tool_freq.keys())
        catalog_ids = set(catalog.tool_ids) | {tid.replace("_", " ") for tid in catalog.tool_ids}
        uncovered = sorted(
            t.id for t in catalog.tools
            if t.id not in covered and t.id.replace("_", " ") not in covered
        )
        by_domain[domain] = {
            "dependency_type": catalog.dependency_type.value,
            "num_tools": len(catalog),
            "raw_total": len(domain_samples),
            "raw_by_topology": dict(topo),
            "usable_total": len(usable),
            "usable_by_topology": dict(usable_topo),
            "usable_by_bucket": dict(buckets),
            "exclusions": dict(excl),
            "tools_used_in_usable": len(covered & ({t.id for t in catalog.tools} | catalog_ids)),
            "tools_never_used_in_usable": uncovered,
            "tool_frequency_top10": tool_freq.most_common(10),
        }

    grand_usable = [s for s in all_samples if s.is_usable and s.topology.value in ("single", "chain")]
    overall = {
        "raw_total": len(all_samples),
        "usable_total": len(grand_usable),
        "usable_by_topology": dict(Counter(s.topology.value for s in grand_usable)),
        "usable_by_bucket": dict(Counter(s.chain_length_bucket for s in grand_usable)),
        "usable_by_domain": dict(Counter(s.domain for s in grand_usable)),
    }
    return {"overall": overall, "by_domain": by_domain}


def write_dataset_stats(stats: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, sort_keys=True)
