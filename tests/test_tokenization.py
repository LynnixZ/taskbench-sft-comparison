"""Tokenization tests: prompt masking and target supervision."""
from __future__ import annotations

from taskbench_sft.tokenization import IGNORE_INDEX, build_example


def test_prompt_tokens_masked_and_target_supervised(fake_tokenizer):
    """(8)+(9) Prompt tokens are masked (-100); target tokens are supervised."""
    messages = [
        {"role": "system", "content": "sys prompt here"},
        {"role": "user", "content": "user request text"},
    ]
    target = "alpha beta gamma"
    ex = build_example(
        fake_tokenizer, messages, target, max_seq_length=128, use_chat_template=False
    )
    n_prompt = ex.n_prompt_tokens
    # Prompt portion fully masked.
    assert all(l == IGNORE_INDEX for l in ex.labels[:n_prompt])
    # Target portion supervised and equal to the input ids there.
    assert ex.labels[n_prompt:] == ex.input_ids[n_prompt:]
    assert all(l != IGNORE_INDEX for l in ex.labels[n_prompt:])
    # EOS appended and supervised.
    assert ex.input_ids[-1] == fake_tokenizer.eos_token_id
    assert ex.labels[-1] == fake_tokenizer.eos_token_id
    # Target tokens = 3 words + EOS.
    assert ex.n_target_tokens == 4


def test_truncation_flag_set(fake_tokenizer):
    messages = [{"role": "system", "content": "a b c"}, {"role": "user", "content": "d e f"}]
    target = "g h i j k"
    ex = build_example(fake_tokenizer, messages, target, max_seq_length=4, use_chat_template=False)
    assert ex.truncated
    # We never silently truncate: full ids are still present, only the flag is set.
    assert ex.n_total_tokens > 4


def test_label_length_matches_input(fake_tokenizer):
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "y z"}]
    ex = build_example(fake_tokenizer, messages, "t1 t2", max_seq_length=128, use_chat_template=False)
    assert len(ex.labels) == len(ex.input_ids) == len(ex.attention_mask)
