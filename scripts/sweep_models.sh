#!/usr/bin/env bash
# MODEL comparison: one fixed robust LoRA(r=16) recipe across several models,
# to validate that SFT on TaskBench improves over the base model.
#
# Per model: Base inference (full_json + trajectory) + SFT (train -> infer) ->
# evaluate on test -> per-model comparison.md (Base vs SFT). The split (sample
# IDs) is SHARED across all models for a fair comparison; the token-length report
# (which depends on the tokenizer) is computed per model.
#
#   source scripts/setup_US.sh ; export WANDB_API_KEY=... EXPERIMENT_RUN_ID=models-$(date +%Y%m%d)
#   GPUS="0 1 2 3" bash scripts/sweep_models.sh
#
# Gated models (Llama-2/3) need HF_TOKEN + accepted license. Single-GPU / quick:
#   MAX_STEPS=50 MODELS="Qwen/Qwen2.5-1.5B" bash scripts/sweep_models.sh
set -Eeuo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/experiment_models.yaml}"
MODES="${MODES:-full_json trajectory}"
SPLIT="${TEST_SPLIT:-test_all}"
OUT_ROOT="${OUTPUT_DIR:-outputs/models}"
GPUS="${GPUS:-}"
EXTRA=""
[ -n "${MAX_STEPS:-}" ] && EXTRA="--set training.max_steps=$MAX_STEPS"

# Models to compare (override with MODELS="a b c"). All are Llama-style archs,
# so the LoRA target_modules in the config apply to every one.
DEFAULT_MODELS=(
  "Qwen/Qwen3-8B"
  "lmsys/vicuna-7b-v1.5"
  "meta-llama/Llama-2-7b-hf"
  "Qwen/Qwen2.5-1.5B"
  "meta-llama/Llama-3.2-3B"
  "mistralai/Mistral-7B-v0.1"
)
if [ -n "${MODELS:-}" ]; then read -ra MODEL_LIST <<< "$MODELS"; else MODEL_LIST=("${DEFAULT_MODELS[@]}"); fi

CLI="python -m taskbench_sft.cli --config $CONFIG"

# ---- 1. Shared split once (same sample IDs for every model) ----
$CLI stats
$CLI split

slugify() { echo "$1" | tr '/:' '__'; }

run_model() {  # train + test one model end to end (Base x modes, SFT x modes)
  local model="$1" gpu="$2"
  [ -n "$gpu" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  local slug; slug=$(slugify "$model")
  local mout="$OUT_ROOT/$slug"
  local mset="--set model.name=$model --set output_dir=$mout --set tokenization.report_path=artifacts/token_report_$slug.json"
  echo "=================== MODEL $model (gpu=${gpu:-default}) ==================="

  # per-model token-length report (tokenizer-dependent truncation/exclusion)
  python -m taskbench_sft.cli --config "$CONFIG" $mset token-report || return 0

  local reports=()
  for mode in $MODES; do
    # Base (no SFT)
    python -m taskbench_sft.cli --config "$CONFIG" $mset infer --mode "$mode" --run-name "Base-$mode" --split "$SPLIT"
    python -m taskbench_sft.cli --config "$CONFIG" $mset evaluate --mode "$mode" \
        --predictions "$mout/Base-$mode/predictions_$SPLIT.jsonl" --out "$mout/Base-$mode/metrics.json"
    reports+=("Base-$mode=$mout/Base-$mode/metrics.json")

    # SFT
    if python -m taskbench_sft.cli --config "$CONFIG" $mset $EXTRA train --mode "$mode" --run-name "SFT-$mode"; then
      local adapter="$mout/SFT-$mode/best_by_common_score"
      [ -d "$adapter" ] || adapter="$mout/SFT-$mode/best_by_loss"
      python -m taskbench_sft.cli --config "$CONFIG" $mset infer --mode "$mode" --run-name "SFT-$mode" --adapter "$adapter" --split "$SPLIT"
      python -m taskbench_sft.cli --config "$CONFIG" $mset evaluate --mode "$mode" \
          --predictions "$mout/SFT-$mode/predictions_$SPLIT.jsonl" --out "$mout/SFT-$mode/metrics.json"
      reports+=("SFT-$mode=$mout/SFT-$mode/metrics.json")
    else
      echo "[models] WARN: SFT-$mode diverged for $model; skipping"
    fi
  done
  python -m taskbench_sft.cli --config "$CONFIG" compare --reports "${reports[@]}" --out "$mout/comparison.md"
  echo "[models] $model done -> $mout/comparison.md"
}

# ---- 2. Run models: parallel across GPUS (one model per GPU) or serial ----
if [ -n "$GPUS" ]; then
  read -ra GPU_ARR <<< "$GPUS"; NG=${#GPU_ARR[@]}
  echo "[models] parallel across $NG GPU(s): $GPUS"
  i=0
  for model in "${MODEL_LIST[@]}"; do
    ( run_model "$model" "${GPU_ARR[$((i % NG))]}" ) &
    i=$((i + 1)); (( i % NG == 0 )) && wait
  done
  wait
else
  for model in "${MODEL_LIST[@]}"; do run_model "$model" ""; done
fi

echo "[models] ALL DONE. Per-model tables: $OUT_ROOT/<model>/comparison.md"
