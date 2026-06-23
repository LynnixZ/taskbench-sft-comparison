"""Orchestrate evaluation: parse predictions, compute grouped metrics.

Results are reported overall and grouped by domain, topology, and chain length
(2 / 3 / 4+). The common metrics apply to both modes; the full-JSON-specific
metrics are added only for Mode A.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from taskbench_sft.config import ExperimentConfig
from taskbench_sft.eval.metrics_common import compute_common_metrics
from taskbench_sft.eval.metrics_fulljson import compute_fulljson_metrics
from taskbench_sft.eval.parse import ParsedPrediction, parse_prediction
from taskbench_sft.logging_utils import get_logger
from taskbench_sft.schema import GoldSample, Mode, ToolCatalog

logger = get_logger(__name__)

_BUCKET_TO_LEN = {
    "chain_length_2": "2",
    "chain_length_3": "3",
    "chain_length_4_plus": "4+",
}


def load_predictions(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _align(
    predictions: List[Dict[str, Any]],
    golds_by_id: Dict[str, GoldSample],
    mode: Mode,
) -> Tuple[List[GoldSample], List[ParsedPrediction]]:
    golds: List[GoldSample] = []
    parsed: List[ParsedPrediction] = []
    missing = 0
    for rec in predictions:
        sid = str(rec["sample_id"])
        gold = golds_by_id.get(sid)
        if gold is None:
            missing += 1
            continue
        pp = parse_prediction(
            mode, sid, gold.domain, gold.topology.value, rec.get("raw_response", "")
        )
        golds.append(gold)
        parsed.append(pp)
    if missing:
        logger.warning("%d predictions had no matching gold sample (ignored)", missing)
    return golds, parsed


def _grouped(
    golds: List[GoldSample], preds: List[ParsedPrediction]
) -> Dict[str, Dict[str, Tuple[List[GoldSample], List[ParsedPrediction]]]]:
    groups: Dict[str, Dict[str, Tuple[List[GoldSample], List[ParsedPrediction]]]] = {
        "overall": {"overall": (golds, preds)},
        "domain": defaultdict(lambda: ([], [])),
        "topology": defaultdict(lambda: ([], [])),
        "chain_length": defaultdict(lambda: ([], [])),
    }
    dom: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))
    topo: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))
    clen: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))
    for g, p in zip(golds, preds):
        dom[g.domain][0].append(g)
        dom[g.domain][1].append(p)
        topo[g.topology.value][0].append(g)
        topo[g.topology.value][1].append(p)
        if g.topology.value == "chain":
            key = _BUCKET_TO_LEN.get(g.chain_length_bucket, g.chain_length_bucket)
            clen[key][0].append(g)
            clen[key][1].append(p)
    groups["domain"] = dict(dom)
    groups["topology"] = dict(topo)
    groups["chain_length"] = dict(clen)
    return groups


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    golds_by_id: Dict[str, GoldSample],
    catalogs: Dict[str, ToolCatalog],
    mode: Mode,
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    """Compute the full grouped metric report for one prediction set."""
    golds, parsed = _align(predictions, golds_by_id, mode)
    groups = _grouped(golds, parsed)

    def metrics_for(gs: List[GoldSample], ps: List[ParsedPrediction]) -> Dict[str, Any]:
        m = compute_common_metrics(gs, ps, catalogs)
        if mode == Mode.FULL_JSON:
            m["full_json_specific"] = compute_fulljson_metrics(
                gs, ps, compute_rouge=cfg.eval.compute_rouge
            )
        return m

    report: Dict[str, Any] = {"mode": mode.value, "n_total": len(golds), "groups": {}}
    for dim in cfg.eval.group_by + ["overall"]:
        if dim not in groups:
            continue
        report["groups"][dim] = {}
        for key in sorted(groups[dim].keys()):
            gs, ps = groups[dim][key]
            if gs:
                report["groups"][dim][key] = metrics_for(gs, ps)
    return report


def write_report(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
