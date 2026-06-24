#!/usr/bin/env bash
# Experiment GRID: each MODEL, WITHIN each DOMAIN separately, runs the 4 settings
# (Base/SFT x Full-JSON/Trajectory). Per-domain isolation: each domain gets its
# own split, the model is SFT-trained on that domain's train and tested on that
# domain's test (this is the standard per-domain TaskBench setup).
#
# Matrix: |MODELS| x |DOMAINS| cells; each cell = Base x2 + SFT x2 + a Base-vs-SFT
# comparison.md. A grand_comparison.md over all cells is written at the end.
#
#   source scripts/setup_US.sh ; export WANDB_API_KEY=... EXPERIMENT_RUN_ID=grid-$(date +%Y%m%d)
#   export HF_TOKEN=...                       # for gated models (Llama/Mistral)
#   GPUS="0 1 2 3" bash scripts/run_grid.sh   # one (model,domain) cell per GPU
#
# Quick check (single GPU): MAX_STEPS=50 MODELS="Qwen/Qwen2.5-1.5B-Instruct" \
#   DOMAINS="data_huggingface" bash scripts/run_grid.sh
set -Eeuo pipefail
cd "$(dirname "$0")/.."

# The dynamic GPU scheduler uses associative arrays + `wait -n` (bash >= 4.3).
if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]:-0}" -lt 3 ]; }; then
  echo "FATAL: this script needs bash >= 4.3 (found $BASH_VERSION). On macOS: 'brew install bash'." >&2
  exit 1
fi

CONFIG="${CONFIG:-configs/experiment_models.yaml}"
MODES="${MODES:-full_json trajectory}"
SPLIT="${TEST_SPLIT:-test_all}"           # within a domain, test_all = node + chain
OUT_ROOT="${OUTPUT_DIR:-outputs/grid}"
GPUS="${GPUS:-}"

# Smoke knob: MAX_STEPS caps optimizer steps AND switches to step-based eval (so an
# eval/checkpoint actually happens within the few steps) and disables early stop.
EXTRA=()
if [ -n "${MAX_STEPS:-}" ]; then
  HALF=$(( MAX_STEPS / 2 )); [ "$HALF" -lt 1 ] && HALF=1
  EXTRA=(--set "training.max_steps=$MAX_STEPS"
         --set "training.eval_strategy=steps"
         --set "training.eval_steps=$HALF"
         --set "training.early_stopping_patience=null")
fi
# Smoke knob: INFER_LIMIT caps how many test samples each inference generates.
LIMIT_ARGS=()
[ -n "${INFER_LIMIT:-}" ] && LIMIT_ARGS=(--limit "$INFER_LIMIT")

DOMAINS_DEFAULT=(data_huggingface data_multimedia data_dailylifeapis)
if [ -n "${DOMAINS:-}" ]; then read -ra DOMAIN_LIST <<< "$DOMAINS"; else DOMAIN_LIST=("${DOMAINS_DEFAULT[@]}"); fi

MODELS_DEFAULT=(
  "Qwen/Qwen3-8B"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "lmsys/vicuna-7b-v1.5"
  "meta-llama/Llama-2-7b-chat-hf"
  "meta-llama/Llama-3.2-3B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
)
if [ -n "${MODELS:-}" ]; then read -ra MODEL_LIST <<< "$MODELS"; else MODEL_LIST=("${MODELS_DEFAULT[@]}"); fi

slugify() { echo "$1" | tr '/:' '__'; }

# ---- Phase 1: per-domain splits (depend only on domain + seed; shared by models) ----
for domain in "${DOMAIN_LIST[@]}"; do
  echo "[grid] split for domain $domain"
  python -m taskbench_sft.cli --config "$CONFIG" \
    --set "data.domains=[\"$domain\"]" \
    --set "split.out_dir=artifacts/splits/$domain" \
    split
done

# ---- one (model, domain) cell: token-report + Base x2 + SFT x2 + compare ----
run_cell() {
  local model="$1" domain="$2" gpu="$3"
  [ -n "$gpu" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  local mslug; mslug=$(slugify "$model")
  local cout="$OUT_ROOT/$mslug/$domain"
  # Shared overrides for every CLI call in this cell (bash array = safe quoting).
  local -a base=(
    --config "$CONFIG"
    --set "data.domains=[\"$domain\"]"
    --set "split.out_dir=artifacts/splits/$domain"
    --set "model.name=$model"
    --set "output_dir=$cout"
    --set "tokenization.report_path=artifacts/token_report_${mslug}_${domain}.json"
  )
  echo "=================== MODEL $model | DOMAIN $domain (gpu=${gpu:-default}) ==================="
  python -m taskbench_sft.cli "${base[@]}" token-report || { echo "[grid] token-report failed for $model/$domain"; return 0; }

  local reports=()
  for mode in $MODES; do
    # Base (no SFT)
    python -m taskbench_sft.cli "${base[@]}" infer --mode "$mode" --run-name "Base-$mode" --split "$SPLIT" "${LIMIT_ARGS[@]}"
    python -m taskbench_sft.cli "${base[@]}" evaluate --mode "$mode" \
        --predictions "$cout/Base-$mode/predictions_$SPLIT.jsonl" --out "$cout/Base-$mode/metrics.json"
    reports+=("Base-$mode=$cout/Base-$mode/metrics.json")

    # SFT (train within this domain -> test within this domain)
    if python -m taskbench_sft.cli "${base[@]}" "${EXTRA[@]}" train --mode "$mode" --run-name "SFT-$mode-$domain"; then
      local adapter="$cout/SFT-$mode-$domain/best_by_common_score"
      [ -d "$adapter" ] || adapter="$cout/SFT-$mode-$domain/best_by_loss"
      python -m taskbench_sft.cli "${base[@]}" infer --mode "$mode" --run-name "SFT-$mode-$domain" --adapter "$adapter" --split "$SPLIT" "${LIMIT_ARGS[@]}"
      python -m taskbench_sft.cli "${base[@]}" evaluate --mode "$mode" \
          --predictions "$cout/SFT-$mode-$domain/predictions_$SPLIT.jsonl" --out "$cout/SFT-$mode-$domain/metrics.json"
      reports+=("SFT-$mode=$cout/SFT-$mode-$domain/metrics.json")
    else
      echo "[grid] WARN: SFT-$mode diverged for $model/$domain; skipping"
    fi
  done
  python -m taskbench_sft.cli --config "$CONFIG" compare --reports "${reports[@]}" --out "$cout/comparison.md"
  echo "[grid] cell done: $model/$domain -> $cout/comparison.md"
}

# Pre-cache each model once (serial) so parallel cells don't race the download.
if [ -n "$GPUS" ]; then
  for model in "${MODEL_LIST[@]}"; do
    [ -d "$model" ] && continue
    echo "[grid] pre-caching $model"
    MODEL_ID="$model" python - <<'PY' || echo "[grid] pre-cache skipped (gated/offline?)"
import os
from pathlib import Path
m = os.environ["MODEL_ID"]
if not Path(m).exists():
    from huggingface_hub import snapshot_download
    snapshot_download(m, token=os.environ.get("HF_TOKEN") or None,
                      ignore_patterns=["original/*", "*.pth", "*.gguf", "consolidated*"])
PY
  done
fi

# ---- Phase 2: run all (model, domain) cells, parallel across GPUS or serial ----
CELLS=()
for model in "${MODEL_LIST[@]}"; do for domain in "${DOMAIN_LIST[@]}"; do CELLS+=("$model|$domain"); done; done

if [ -n "$GPUS" ]; then
  read -ra GPU_ARR <<< "$GPUS"
  echo "[grid] ${#CELLS[@]} cells, dynamic dispatch across ${#GPU_ARR[@]} GPU(s): $GPUS"
  # Work queue: each GPU picks up the next pending cell the moment it frees up
  # (better balanced than batch-syncing, which idles fast GPUs on the slow one).
  declare -A PID_GPU=()
  free=("${GPU_ARR[@]}")
  idx=0
  while [ "$idx" -lt ${#CELLS[@]} ] || [ ${#PID_GPU[@]} -gt 0 ]; do
    while [ ${#free[@]} -gt 0 ] && [ "$idx" -lt ${#CELLS[@]} ]; do
      gpu="${free[0]}"; free=("${free[@]:1}")
      cell="${CELLS[$idx]}"; idx=$((idx + 1))
      ( run_cell "${cell%|*}" "${cell#*|}" "$gpu" ) &
      PID_GPU[$!]="$gpu"
    done
    wait -n 2>/dev/null || wait     # wait for any one cell (fallback: wait all)
    for pid in "${!PID_GPU[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        free+=("${PID_GPU[$pid]}"); unset 'PID_GPU[$pid]'
      fi
    done
  done
else
  for cell in "${CELLS[@]}"; do run_cell "${cell%|*}" "${cell#*|}" ""; done
fi

# ---- Phase 3: grand comparison over every cell (model x domain x setting) ----
GRAND=()
for model in "${MODEL_LIST[@]}"; do
  mslug=$(slugify "$model")
  for domain in "${DOMAIN_LIST[@]}"; do
    cout="$OUT_ROOT/$mslug/$domain"
    for mode in $MODES; do
      for s in "Base-$mode" "SFT-$mode-$domain"; do
        [ -f "$cout/$s/metrics.json" ] && GRAND+=("$mslug:$domain:$s=$cout/$s/metrics.json")
      done
    done
  done
done
if [ ${#GRAND[@]} -gt 0 ]; then
  python -m taskbench_sft.cli --config "$CONFIG" compare --reports "${GRAND[@]}" --out "$OUT_ROOT/grand_comparison.md"
  echo "[grid] GRAND comparison -> $OUT_ROOT/grand_comparison.md"
fi
echo "[grid] ALL DONE."
