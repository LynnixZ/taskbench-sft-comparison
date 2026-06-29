"""Unit tests for rule-aware label smoothing (train/rule_smoothing.py).

The builder is tokenizer-agnostic, so we test it with a trivial CHARACTER tokenizer
(each char = 1 token, offsets are identity) — no transformers/torch needed. This
exercises span-finding, candidate derivation, and multi-token-span alignment
deterministically. The loss test is skipped when torch is unavailable.
"""
import importlib.util
import json
import os

import pytest

_RS = os.path.join(os.path.dirname(__file__), "..", "taskbench_sft", "train", "rule_smoothing.py")
_spec = importlib.util.spec_from_file_location("rule_smoothing", _RS)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


class CharTok:
    """Minimal char-level tokenizer: token id = ord(char), offsets = identity."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def _link(a, b):
    return type("L", (), {"source": a, "target": b})()


def test_bookflight_first_position_smoothed():
    tok = CharTok()
    traj = ["CheckWeather", "SearchFlight", "BookFlight"]
    links = [("SearchFlight", "BookFlight")]            # SearchFlight is prereq of BookFlight
    target = json.dumps(traj)                            # ["CheckWeather", "SearchFlight", "BookFlight"]
    soft, ids = rs.build_soft_targets(traj, links, target, tok, alpha_max=0.1)

    # First char of CheckWeather ('C') should get 0.9, candidate SearchFlight ('S') 0.1.
    c_idx = target.index("CheckWeather")
    d = soft[c_idx]
    assert d[ord("C")] == pytest.approx(0.9)
    assert d[ord("S")] == pytest.approx(0.1)
    assert sum(d.values()) == pytest.approx(1.0)
    # Multi-token span: 2nd char ('h' of Check vs 'e' of Search) also smoothed.
    d2 = soft[c_idx + 1]
    assert d2[ord("h")] == pytest.approx(0.9)
    assert d2[ord("e")] == pytest.approx(0.1)


def test_first_token_collision_merges_then_distinguishes():
    tok = CharTok()
    # gold[0]='Image Editing'; prereq candidate 'Image Classification' (consumer later).
    traj = ["Image Editing", "Image Classification", "Image Segmentation"]
    links = [("Image Segmentation", "Image Classification")]
    target = json.dumps(traj)
    soft, ids = rs.build_soft_targets(traj, links, target, tok, alpha_max=0.2)

    base = target.index("Image Editing")
    # Shared prefix "Image " -> those positions merge back to ~1.0 on the gold char.
    assert soft[base][ord("I")] == pytest.approx(1.0)          # 'I' gold 0.8 + cand 0.2
    # First DIVERGING char: gold 'E'(diting) 0.8 vs candidate 'C'(lassification) 0.2.
    div = base + len("Image ")
    assert target[div] == "E"
    assert soft[div][ord("E")] == pytest.approx(0.8)
    assert soft[div][ord("C")] == pytest.approx(0.2)


def test_no_candidate_no_smoothing():
    tok = CharTok()
    traj = ["A", "B"]
    soft, _ = rs.build_soft_targets(traj, [], json.dumps(traj), tok, alpha_max=0.1)
    assert soft == {}


def test_loss_matches_ce_without_soft_then_shifts_with_soft():
    torch = pytest.importorskip("torch")
    V, L = 6, 4
    logits = torch.randn(1, L, V)
    labels = torch.tensor([[-100, 2, 3, 4]])   # predict tokens at positions 1..3
    # No soft targets -> equals plain shifted CE.
    import torch.nn.functional as F
    plain = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), labels[:, 1:].reshape(-1),
                            ignore_index=-100)
    got = rs.rule_smoothing_loss(logits, labels, [{}])
    assert torch.allclose(got, plain, atol=1e-5)

    # A soft target at label index 2 ({tok2:0.7, tok5:0.3}) changes the loss.
    soft = [{2: {2: 0.7, 5: 0.3}}]
    got2 = rs.rule_smoothing_loss(logits, labels, soft)
    assert not torch.allclose(got2, plain, atol=1e-4)
