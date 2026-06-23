"""Model + tokenizer construction for full / LoRA / QLoRA fine-tuning."""
from __future__ import annotations

from typing import Any, Tuple

from taskbench_sft.config import ExperimentConfig
from taskbench_sft.logging_utils import get_logger

logger = get_logger(__name__)


def load_tokenizer(cfg: ExperimentConfig) -> Any:
    from transformers import AutoTokenizer

    name = cfg.model.tokenizer_name or cfg.model.name
    tok = AutoTokenizer.from_pretrained(
        name,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        logger.info("Tokenizer had no pad token; set pad_token = eos_token")
    # Decoder-only models generate on the right; pad left for batched generation.
    tok.padding_side = "right"
    return tok


def _torch_dtype(cfg: ExperimentConfig):
    import torch

    if cfg.training.bf16:
        return torch.bfloat16
    if cfg.training.fp16:
        return torch.float16
    return torch.float32


def load_model(cfg: ExperimentConfig, for_training: bool = True) -> Any:
    """Load the causal LM, applying QLoRA/LoRA/full configuration."""
    import torch
    from transformers import AutoModelForCausalLM

    method = cfg.training.method.lower()
    quant_config = None
    if method == "qlora":
        try:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_torch_dtype(cfg),
                bnb_4bit_use_double_quant=True,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("QLoRA requested but bitsandbytes unavailable (%s); falling back to LoRA", exc)
            method = "lora"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=_torch_dtype(cfg),
        quantization_config=quant_config,
        device_map="auto" if (quant_config is not None or torch.cuda.is_available()) else None,
    )

    if for_training and cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model

        if method == "qlora":
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.training.gradient_checkpointing
            )
        lora = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    elif method != "full":
        raise ValueError(f"Unknown training.method: {cfg.training.method}")

    return model


def load_model_and_tokenizer(cfg: ExperimentConfig, for_training: bool = True) -> Tuple[Any, Any]:
    return load_model(cfg, for_training=for_training), load_tokenizer(cfg)


def load_for_inference(cfg: ExperimentConfig, adapter_dir: Any = None) -> Any:
    """Load the base model for inference, optionally with a trained adapter.

    ``adapter_dir=None`` loads the untouched base model (the no-SFT baseline runs).
    """
    import torch
    from transformers import AutoModelForCausalLM

    # For QLoRA, load the base in 4-bit for inference too: fits small GPUs (e.g.
    # an 8B model on 12GB) AND keeps the base precision identical between the
    # Base and SFT runs (a fairer comparison, since SFT trained on a 4-bit base).
    quant_config = None
    if cfg.training.method.lower() == "qlora":
        try:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_torch_dtype(cfg),
                bnb_4bit_use_double_quant=True,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "4-bit inference requested but bitsandbytes unavailable (%s); loading full precision",
                exc,
            )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=_torch_dtype(cfg),
        quantization_config=quant_config,
        device_map="auto" if (quant_config is not None or torch.cuda.is_available()) else None,
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        logger.info("Loaded adapter from %s", adapter_dir)
    model.eval()
    return model
