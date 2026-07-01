"""Rule-aware label smoothing for the TRAJECTORY mode (method 2).

Idea: the gold one-hot next-token target over-penalizes emitting a tool that is
ALSO a valid prerequisite of a later step. Instead, build a soft target that puts a
small mass alpha on the *prerequisite* tools that could legitimately appear at this
position, derived from the sample's own dependency edges (task_links).

Granularity = **multi-token span** (the choice that survives first-token collisions:
HuggingFace/Multimedia tools share first tokens like "Image"/"Text"). For a gold tool
occupying token span [s..e), and a candidate prerequisite B with name-tokens b_0..b_{m-1},
we overlay B's tokens position-by-position from the span start:
    pos s+k :  (1-alpha)*gold_tok(s+k)  +  alpha * B_tok(k)
so the WHOLE name disambiguates the candidate, not just its (colliding) first token.

This module is intentionally torch-free in the BUILDER so it can be unit-tested on a
laptop with just a tokenizer; the LOSS imports torch lazily (runs on the GPU node).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

# Matches each quoted tool name in a trajectory JSON list:  ["Image Classification", ...]
_QUOTED = re.compile(r'"([^"]*)"')


def _name_tokens(name: str, tokenizer: Any) -> List[int]:
    """Token ids of a tool name AS IT APPEARS after an opening quote in the list.

    We tokenize `"` + name and drop the leading quote token(s), so the name's first
    token matches its in-list tokenization (byte-level BPE: the name starts fresh
    after the quote token).
    """
    q = tokenizer('"', add_special_tokens=False)["input_ids"]
    full = tokenizer('"' + name, add_special_tokens=False)["input_ids"]
    return list(full[len(q):]) if full[: len(q)] == list(q) else list(full)


def _tool_spans(target_text: str, tokenizer: Any) -> Tuple[List[Tuple[str, int, int]], List[int]]:
    """Return [(name, tok_start, tok_end_exclusive), ...] and the target token ids.

    Uses offset mapping (fast tokenizers) to map each quoted name's CHAR span to its
    TOKEN span in tokenizer(target_text). Order matches the JSON list order.
    """
    enc = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    spans: List[Tuple[str, int, int]] = []
    for m in _QUOTED.finditer(target_text):
        cs, ce = m.start(1), m.end(1)
        if ce == cs:  # empty string token, skip
            continue
        toks = [i for i, (a, b) in enumerate(offs) if b > a and a < ce and b > cs]
        if toks:
            spans.append((m.group(1), toks[0], toks[-1] + 1))
    return spans, list(enc["input_ids"])


def _candidates(traj: Sequence[str], links: Sequence[Tuple[str, str]], max_lag: int | None,
                ) -> Dict[int, Dict[str, float]]:
    """For each gold position r, {prerequisite_name: rule_weight}.

    Direction-robust: for an edge {a,b}, the one appearing EARLIER in the gold
    trajectory is the prerequisite, the later one the consumer (so we don't depend on
    the source/target convention). The prerequisite is a candidate at positions r
    BEFORE the consumer, where it hasn't appeared yet and isn't the gold tool there.
    Rule weight decays with lag (rho ~ 1/lag); cap with max_lag if set.
    """
    pos: Dict[str, int] = {}
    for i, n in enumerate(traj):
        pos.setdefault(n, i)  # first occurrence
    cand: Dict[int, Dict[str, float]] = {}
    for a, b in links:
        if a not in pos or b not in pos or pos[a] == pos[b]:
            continue
        prereq, consumer = (a, b) if pos[a] < pos[b] else (b, a)
        t = pos[consumer]
        for r in range(t):
            if traj[r] == prereq:          # prereq is the gold tool here -> no-op
                continue
            if prereq in traj[:r]:         # already emitted in prefix y_<r -> stop
                continue
            lag = t - r
            if max_lag and lag > max_lag:
                continue
            cand.setdefault(r, {})[prereq] = cand.get(r, {}).get(prereq, 0.0) + 1.0 / lag
    return cand


def build_soft_targets(
    traj: Sequence[str],
    links: Sequence[Tuple[str, str]],
    target_text: str,
    tokenizer: Any,
    alpha_max: float = 0.1,
    max_lag: int | None = None,
    span_decay: float = 0.5,
) -> Tuple[Dict[int, Dict[int, float]], List[int]]:
    """Build {target_token_index: {token_id: prob}} soft targets + the target ids.

    The smoothing mass DECAYS along a tool name: at the k-th name token the mass is
    ``alpha_max * span_decay**k`` (k=0 is the first/decision token). ``span_decay=1``
    smooths the whole span flat, ``0`` smooths only the first token, ``0<d<1`` decays.
    Only positions that differ from one-hot are returned; each distribution sums to 1.
    """
    spans, ids = _tool_spans(target_text, tokenizer)
    cand = _candidates(traj, links, max_lag)
    soft: Dict[int, Dict[int, float]] = {}
    n = min(len(spans), len(traj))
    for r in range(n):
        cs = cand.get(r)
        if not cs:
            continue
        name, ts, te = spans[r]
        ctoks = {c: _name_tokens(c, tokenizer) for c in cs}
        tot_w = sum(cs.values())
        for k in range(te - ts):
            alpha_k = alpha_max * (span_decay ** k)   # 0**0 == 1 -> first token keeps alpha_max
            if alpha_k < 1e-4:                          # decayed to ~0 -> stop smoothing this name
                break
            gtid = ids[ts + k]
            share: Dict[int, float] = {}
            for c, w in cs.items():
                ct = ctoks[c]
                if k < len(ct):
                    tid = ct[k]
                    share[tid] = share.get(tid, 0.0) + alpha_k * (w / tot_w)
            if not share:
                continue
            alpha = sum(share.values())
            dist: Dict[int, float] = {gtid: 1.0 - alpha}
            for tid, p in share.items():
                dist[tid] = dist.get(tid, 0.0) + p
            soft[ts + k] = dist
    return soft, ids


def rule_smoothing_loss(logits: Any, labels: Any, soft_targets: List[Dict[int, Dict[int, float]]],
                        ignore_index: int = -100) -> Any:
    """Causal-LM loss = per-token CE, with soft cross-entropy at marked positions.

    ``logits``: (B, L, V); ``labels``: (B, L) with prompt masked to ignore_index.
    ``soft_targets[b]``: {label_index p: {token_id: prob}} (p indexes ``labels``;
    the prediction comes from logits[p-1] after the causal shift).
    """
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    logp = F.log_softmax(shift_logits, dim=-1)            # (B, L-1, V)
    mask = shift_labels != ignore_index
    safe = shift_labels.clamp_min(0)
    nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # (B, L-1) hard CE

    for b, st in enumerate(soft_targets or []):
        for p, dist in st.items():
            j = p - 1                                      # causal shift
            if j < 0 or j >= nll.shape[1] or not bool(mask[b, j]):
                continue
            term = logp.new_zeros(())
            for tid, prob in dist.items():
                term = term - prob * logp[b, j, tid]
            nll[b, j] = term

    denom = mask.sum().clamp_min(1)
    return (nll * mask).sum() / denom
