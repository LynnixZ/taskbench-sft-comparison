"""Create a tiny, randomly-initialized Qwen2 causal LM for smoke testing.

This builds a few-layer Qwen2 model that shares the real Qwen2.5 tokenizer (and
therefore the chat template + LoRA target module names q_proj/k_proj/.../down_proj).
It downloads only the small tokenizer, not any large weights, so the smoke test
runs in seconds on CPU.

Usage:
    python scripts/make_tiny_model.py [output_dir] [tokenizer_name]
"""
from __future__ import annotations

import sys

from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/tiny_model"
    tok_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Tiny model saved to {out_dir} (vocab={len(tokenizer)}, params={model.num_parameters()})")


if __name__ == "__main__":
    main()
