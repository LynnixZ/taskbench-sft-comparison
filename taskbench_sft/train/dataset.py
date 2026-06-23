"""Torch dataset + collator for supervised fine-tuning.

The dataset applies the shared truncation-exclusion set so that both modes train
on exactly the same sample IDs. Loss masking is handled by
:mod:`taskbench_sft.tokenization`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from taskbench_sft.config import ExperimentConfig
from taskbench_sft.logging_utils import get_logger
from taskbench_sft.prompts.builder import build_messages
from taskbench_sft.schema import GoldSample, Mode, ToolCatalog
from taskbench_sft.targets import build_target
from taskbench_sft.tokenization import IGNORE_INDEX, build_example

logger = get_logger(__name__)


class SupervisedDataset(Dataset):
    """Encodes (prompt, target) pairs for a single mode."""

    def __init__(
        self,
        samples: Sequence[GoldSample],
        catalogs: Dict[str, ToolCatalog],
        tokenizer: Any,
        mode: Mode,
        cfg: ExperimentConfig,
        excluded_ids: Optional[set] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.examples: List[Dict[str, List[int]]] = []
        excluded_ids = excluded_ids or set()
        n_excluded = 0
        n_truncated = 0
        for s in samples:
            if s.id in excluded_ids:
                n_excluded += 1
                continue
            messages = build_messages(s, mode, catalogs[s.domain], cfg.prompt)
            target = build_target(s, mode)
            enc = build_example(
                tokenizer,
                messages,
                target,
                cfg.tokenization.max_seq_length,
                use_chat_template=cfg.model.use_chat_template,
                chat_template_kwargs=cfg.model.chat_template_kwargs,
            )
            if enc.truncated and cfg.tokenization.drop_truncated_targets:
                n_truncated += 1
                continue
            self.examples.append(
                {
                    "input_ids": enc.input_ids,
                    "labels": enc.labels,
                    "attention_mask": enc.attention_mask,
                }
            )
        logger.info(
            "SupervisedDataset[%s]: %d examples (%d shared-excluded, %d truncated-dropped)",
            mode.value, len(self.examples), n_excluded, n_truncated,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


class DataCollatorForCausalSFT:
    """Pad input_ids/labels/attention_mask to the longest example in the batch."""

    def __init__(self, tokenizer: Any, label_pad_token_id: int = IGNORE_INDEX) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [self.label_pad_token_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
