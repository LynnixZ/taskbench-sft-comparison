"""Assemble system/user prompts and the supervised (prompt, target) pair."""
from __future__ import annotations

from typing import Dict, List, Tuple

from taskbench_sft.config import PromptConfig
from taskbench_sft.prompts import templates
from taskbench_sft.prompts.catalog import serialize_catalog
from taskbench_sft.schema import GoldSample, Mode, ToolCatalog
from taskbench_sft.targets import build_target


def _system_prompt(mode: Mode) -> str:
    return templates.SYSTEM_FULL_JSON if mode == Mode.FULL_JSON else templates.SYSTEM_TRAJECTORY


def _user_prompt(sample: GoldSample, mode: Mode, catalog: ToolCatalog, cfg: PromptConfig) -> str:
    catalog_str = serialize_catalog(catalog, include_io_types=cfg.catalog_include_io_types)
    if cfg.include_one_shot:
        example = (
            templates.ONE_SHOT_FULL_JSON
            if mode == Mode.FULL_JSON
            else templates.ONE_SHOT_TRAJECTORY
        )
        example_block = f"\nEXAMPLE:\n{example}\n\n"
    else:
        example_block = "\n"
    template = templates.USER_FULL_JSON if mode == Mode.FULL_JSON else templates.USER_TRAJECTORY
    return template.format(
        catalog=catalog_str,
        example_block=example_block,
        user_request=sample.user_request,
    )


def build_messages(
    sample: GoldSample, mode: Mode, catalog: ToolCatalog, cfg: PromptConfig
) -> List[Dict[str, str]]:
    """Return chat-style messages (system + user) for the sample."""
    return [
        {"role": "system", "content": _system_prompt(mode)},
        {"role": "user", "content": _user_prompt(sample, mode, catalog, cfg)},
    ]


def build_prompt_text(
    sample: GoldSample, mode: Mode, catalog: ToolCatalog, cfg: PromptConfig
) -> str:
    """A plain-text prompt (system + user) for tokenizers without chat templates."""
    msgs = build_messages(sample, mode, catalog, cfg)
    return f"{msgs[0]['content']}\n\n{msgs[1]['content']}"


def build_supervised_pair(
    sample: GoldSample, mode: Mode, catalog: ToolCatalog, cfg: PromptConfig
) -> Tuple[List[Dict[str, str]], str]:
    """Return (messages, assistant_target_string) for one supervised example."""
    messages = build_messages(sample, mode, catalog, cfg)
    target = build_target(sample, mode)
    return messages, target
