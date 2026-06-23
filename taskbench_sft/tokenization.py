"""Build supervised training examples with correct loss masking.

The prompt portion (system + user, plus any chat-template generation prefix) is
masked with ``-100`` so it is *not* supervised; only the assistant target tokens
(plus EOS) carry a loss. We never silently truncate an assistant target: if the
full sequence exceeds ``max_seq_length`` the example is flagged as truncated and
the caller excludes it (the same excluded IDs are shared across both modes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

IGNORE_INDEX = -100


@dataclass
class EncodedExample:
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]
    n_prompt_tokens: int
    n_target_tokens: int
    n_total_tokens: int
    truncated: bool


def _encode_prompt_ids(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    use_chat_template: bool,
) -> List[int]:
    """Tokenize the prompt (without the target), adding a generation prefix."""
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback plain-text prompt for tokenizers without a chat template.
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        text = f"{system}\n\n{user}\n\n"
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_example(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    target: str,
    max_seq_length: int,
    use_chat_template: bool = True,
    eos_token_id: Optional[int] = None,
) -> EncodedExample:
    """Encode one (prompt, target) pair into masked-label training tensors."""
    prompt_ids = _encode_prompt_ids(tokenizer, messages, use_chat_template)
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        target_ids = list(target_ids) + [eos_token_id]

    input_ids = list(prompt_ids) + list(target_ids)
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)
    attention_mask = [1] * len(input_ids)

    truncated = len(input_ids) > max_seq_length
    return EncodedExample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        n_prompt_tokens=len(prompt_ids),
        n_target_tokens=len(target_ids),
        n_total_tokens=len(input_ids),
        truncated=truncated,
    )


def measure_lengths(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    target: str,
    max_seq_length: int,
    use_chat_template: bool = True,
) -> Dict[str, Any]:
    """Token-length measurement only (no tensor allocation), for the report."""
    ex = build_example(
        tokenizer, messages, target, max_seq_length, use_chat_template
    )
    return {
        "input_tokens": ex.n_prompt_tokens,
        "target_tokens": ex.n_target_tokens,
        "total_tokens": ex.n_total_tokens,
        "truncated": ex.truncated,
    }
