"""Typed, YAML-backed configuration.

Every knob in the experiment is expressed here as a Pydantic model so that
nothing is hard-coded: model name, training hyper-parameters, LoRA settings,
sequence lengths, decoding parameters, and the checkpoint-selection metric
weights are all overridable from YAML / the CLI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    domains: List[str] = Field(
        default_factory=lambda: [
            "data_huggingface",
            "data_multimedia",
            "data_dailylifeapis",
        ]
    )
    # Only these topologies are kept (DAG is excluded for this study).
    include_topologies: List[str] = Field(default_factory=lambda: ["single", "chain"])
    # Drop chain samples whose gold graph is not a simple connected path.
    require_simple_chain: bool = True
    # Exclude samples whose gold ``tool_nodes`` contain tool names that are not in
    # the domain catalog. The official ``tool_nodes`` label drifts off-catalog in
    # a minority of samples (the sampler ground truth ``sampled_nodes`` is always
    # catalog-faithful). Excluding them keeps the gold catalog-faithful so that
    # hallucination rate is well-defined and train tool coverage is tractable.
    require_catalog_faithful_gold: bool = True


class SplitConfig(BaseModel):
    train_frac: float = 0.8
    validation_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42
    stratify_by: List[str] = Field(
        default_factory=lambda: ["domain", "topology", "chain_length_bucket"]
    )
    # Re-draw the split up to this many times to satisfy train tool coverage.
    max_resamples: int = 50
    out_dir: str = "artifacts/splits"


class PromptConfig(BaseModel):
    include_one_shot: bool = True
    # Serialize the tool catalog as compact JSON lines (id + desc [+ params]).
    catalog_include_io_types: bool = True


class TokenizationConfig(BaseModel):
    max_seq_length: int = 2048
    # Coverage target the max_seq_length must satisfy for full_json samples.
    coverage_target: float = 0.99
    # Samples whose assistant target would be truncated are excluded (never
    # silently truncated). The same excluded IDs are shared across both modes.
    drop_truncated_targets: bool = True
    report_path: str = "artifacts/token_length_report.json"


class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainingConfig(BaseModel):
    method: str = "qlora"  # one of: full | lora | qlora
    epochs: float = 3.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    # Optional cap on training steps (mainly for smoke tests). None = full run.
    max_steps: Optional[int] = None
    # Budget mode: "same_samples" (default) or "equal_target_tokens".
    budget_mode: str = "same_samples"


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: Optional[str] = None
    tokenizer_name: Optional[str] = None  # defaults to ``name`` if None
    trust_remote_code: bool = True
    # Chat template usage: build supervised examples via the tokenizer's chat
    # template when available (recommended for instruct models).
    use_chat_template: bool = True


class CheckpointSelectionConfig(BaseModel):
    """Weights for ``validation_common_score`` (configurable, sums need not be 1)."""

    node_f1: float = 0.4
    edge_f1: float = 0.3
    sequence_exact_match: float = 0.2
    parse_valid_rate: float = 0.1


class InferenceConfig(BaseModel):
    do_sample: bool = False
    temperature: float = 0.0
    num_beams: int = 1
    full_json_max_new_tokens: int = 1024
    trajectory_max_new_tokens: int = 256
    batch_size: int = 1


class EvalConfig(BaseModel):
    # Group results along these dimensions (in addition to "overall").
    group_by: List[str] = Field(
        default_factory=lambda: ["domain", "topology", "chain_length"]
    )
    compute_rouge: bool = True
    compute_bertscore: bool = False
    # During training, generate on at most this many validation samples to compute
    # the generation-based ``validation_common_score`` (0 = use all). Capped for
    # cost; checkpoint selection uses this subset deterministically.
    max_val_eval_samples: int = 256


class ExperimentConfig(BaseModel):
    """Top-level config aggregating all sub-configs."""

    project_name: str = "taskbench-sft-format-comparison"
    output_dir: str = "outputs"
    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    tokenization: TokenizationConfig = Field(default_factory=TokenizationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    checkpoint_selection: CheckpointSelectionConfig = Field(
        default_factory=CheckpointSelectionConfig
    )
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    def merged_with(self, overrides: Dict[str, Any]) -> "ExperimentConfig":
        """Return a copy with a (possibly nested) dict of overrides applied."""
        base = self.model_dump()
        _deep_update(base, overrides)
        return ExperimentConfig.model_validate(base)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Load config from YAML (or defaults) and apply optional overrides."""
    cfg = ExperimentConfig.from_yaml(path) if path else ExperimentConfig()
    if overrides:
        cfg = cfg.merged_with(overrides)
    return cfg
