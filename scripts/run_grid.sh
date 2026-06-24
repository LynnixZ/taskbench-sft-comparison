#!/usr/bin/env bash
# Experiment GRID: each MODEL, WITHIN each DOMAIN separately, runs the 4 settings
# (Base/SFT x Full-JSON/Trajectory). Per-domain isolation: each domain gets its
# own split; the model is SFT-trained on that domain's train and tested on its test.
#
# MODEL-MAJOR + disk-frugal: models are processed ONE AT A TIME. For each model
# its 12 experiments (|DOMAINS| x |MODES| x {Base, SFT} = 3x2x2) are dispatched
# across the GPUs (dynamic work-queue, balanced), and once they all finish the
# model's weights are DELETED from the HF cache before the next model is fetched.
# So only ONE base model is on disk at a time. Set DELETE_MODELS=0 to keep them.
#
#   source scripts/setup_US.sh ; export WANDB_API_KEY=... HF_TOKEN=... EXPERIMENT_RUN_ID=grid-$(date +%Y%m%d)
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/run_grid.sh
#
# 2-GPU smoke (~10 min; keep the model so a re-run doesn't re-download):
#   GPUS="0 1" MAX_STEPS=20 INFER_LIMIT=16 DELETE_MODELS=0 \
#     MODELS="Qwen/Qwen2.5-1.5B-Instruct" DOMAINS="data_huggingface data_multimedia" \
#     bash scripts/run_grid.sh
set -Eeuo pipefail
cd "$(dirname "$0")/.."

# The dynamic GPU scheduler uses associative arrays + `wait -n` (bash >= 4.3).
if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]:-0}" -lt 3 ]; }; then
  echo "FATAL: this script needs bash >= 4.3 (found $BASH_VERSION). On macOS: 'brew install bash'." >&2
  exit 1
fi

CONFIG="${CONFIG:-configs/experiment_models.yaml}"
MODES="${MODES:-full_json trajectory}"
SPLIT="${TEST_SPLIT:-test_all}"
OUT_ROOT="${OUTPUT_DIR:-outputs/grid}"
GPUS="${GPUS:-}"
DELETE_MODELS="${DELETE_MODELS:-1}"      # 1 = delete each model's weights after its 12 runs
HF_HUB_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"

# Smoke knob: MAX_STEPS caps steps AND switches to step-based eval (so an eval
# happens) + disables early stopping. INFER_LIMIT caps test samples generated.
EXTRA=()
if [ -n "${MAX_STEPS:-}" ]; then
  HALF=$(( MAX_STEPS / 2 )); [ "$HALF" -lt 1 ] && HALF=1
  EXTRA=(--set "training.max_steps=$MAX_STEPS" --set "training.eval_strategy=steps"
         --set "training.eval_steps=$HALF" --set "training.early_stopping_patience=null")
fi
LIMIT_ARGS=()
[ -n "${INFER_LIMIT:-}" ] && LIMIT_ARGS=(--limit "$INFER_LIMIT")

DOMAINS_DEFAULT=(data_huggingface data_multimedia data_dailylifeapis)
if [ -n "${DOMAINS:-}" ]; then read -ra DOMAIN_LIST <<< "$DOMAINS"; else DOMAIN_LIST=("${DOMAINS_DEFAULT[@]}"); fi
MODELS_DEFAULT=(
  "Qwen/Qwen3-8B" "Qwen/Qwen2.5-1.5B-Instruct" "lmsys/vicuna-7b-v1.5"
  "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3"
)
if [ -n "${MODELS:-}" ]; then read -ra MODEL_LIST <<< "$MODELS"; else MODEL_LIST=("${MODELS_DEFAULT[@]}"); fi

slugify() { echo "$1" | tr '/:' '__'; }
hub_dir_for() { echo "$HF_HUB_CACHE_DIR/models--$(echo "$1" | sed 's#/#--#g')"; }

# ---- Phase 1: per-domain splits (depend only on domain + seed; shared by all models) ----
for domain in "${DOMAIN_LIST[@]}"; do
  echo "[grid] split for domain $domain"
  python -m taskbench_sft.cli --config "$CONFIG" \
    --set "data.domains=[\"$domain\"]" --set "split.out_dir=artifacts/splits/$domain" split
done

# ---- one experiment unit: (model, domain, mode, kind in {base,sft}) ----
run_unit() {
  local model="$1" domain="$2" mode="$3" kind="$4" gpu="$5"
  [ -n "$gpu" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  local mslug; mslug=$(slugify "$model")
  local cout="$OUT_ROOT/$mslug/$domain"
  local -a base=(
    --config "$CONFIG"
    --set "data.domains=[\"$domain\"]"
    --set "split.out_dir=artifacts/splits/$domain"
    --set "model.name=$model"
    --set "output_dir=$cout"
    --set "tokenization.report_path=artifacts/token_report_${mslug}_${domain}.json"
  )
  if [ "$kind" = base ]; then
    python -m taskbench_sft.cli "${base[@]}" infer --mode "$mode" --run-name "Base-$mode" --split "$SPLIT" "${LIMIT_ARGS[@]}"
    python -m taskbench_sft.cli "${base[@]}" evaluate --mode "$mode" \
        --predictions "$cout/Base-$mode/predictions_$SPLIT.jsonl" --out "$cout/Base-$mode/metrics.json"
  else
    if python -m taskbench_sft.cli "${base[@]}" "${EXTRA[@]}" train --mode "$mode" --run-name "SFT-$mode-$domain"; then
      local adapter="$cout/SFT-$mode-$domain/best_by_common_score"
      [ -d "$adapter" ] || adapter="$cout/SFT-$mode-$domain/best_by_loss"
      python -m taskbench_sft.cli "${base[@]}" infer --mode "$mode" --run-name "SFT-$mode-$domain" --adapter "$adapter" --split "$SPLIT" "${LIMIT_ARGS[@]}"
      python -m taskbench_sft.cli "${base[@]}" evaluate --mode "$mode" \
          --predictions "$cout/SFT-$mode-$domain/predictions_$SPLIT.jsonl" --out "$cout/SFT-$mode-$domain/metrics.json"
    else
      echo "[grid] WARN: SFT-$mode/$domain diverged for $model; skipping"
    fi
  fi
}

# ---- dispatch a list of "model|domain|mode|kind" units across GPUs (or serial) ----
dispatch_units() {
  local -a units=("$@")
  if [ -n "$GPUS" ]; then
    local -a GPU_ARR; read -ra GPU_ARR <<< "$GPUS"
    local -A PID_GPU=(); local -a free=("${GPU_ARR[@]}"); local idx=0
    while [ "$idx" -lt ${#units[@]} ] || [ ${#PID_GPU[@]} -gt 0 ]; do
      while [ ${#free[@]} -gt 0 ] && [ "$idx" -lt ${#units[@]} ]; do
        local gpu="${free[0]}"; free=("${free[@]:1}")
        local m d mo k; IFS='|' read -r m d mo k <<< "${units[$idx]}"; idx=$((idx + 1))
        ( run_unit "$m" "$d" "$mo" "$k" "$gpu" ) &
        PID_GPU[$!]="$gpu"
      done
      wait -n 2>/dev/null || wait
      local pid
      for pid in "${!PID_GPU[@]}"; do
        kill -0 "$pid" 2>/dev/null || { free+=("${PID_GPU[$pid]}"); unset 'PID_GPU['"$pid"']'; }
      done
    done
  else
    local u m d mo k
    for u in "${units[@]}"; do IFS='|' read -r m d mo k <<< "$u"; run_unit "$m" "$d" "$mo" "$k" ""; done
  fi
}

# ---- Phase 2: MODEL-MAJOR loop ----
for model in "${MODEL_LIST[@]}"; do
  mslug=$(slugify "$model")
  echo "############################## MODEL $model ##############################"

  # (a) fetch this model once (serial) -- so only one base model is on disk
  if [ ! -d "$model" ]; then
    echo "[grid] fetching $model"
    MODEL_ID="$model" python - <<'PY' || { echo "[grid] cannot fetch $model (gated/offline?); skipping model"; continue; }
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["MODEL_ID"], token=os.environ.get("HF_TOKEN") or None,
                  ignore_patterns=["original/*", "*.pth", "*.gguf", "consolidated*"])
PY
  fi

  # (b) per-domain token-length reports (tokenizer-dependent; CPU; run in parallel)
  tr_pids=()
  for domain in "${DOMAIN_LIST[@]}"; do
    ( python -m taskbench_sft.cli --config "$CONFIG" \
        --set "data.domains=[\"$domain\"]" --set "split.out_dir=artifacts/splits/$domain" \
        --set "model.name=$model" --set "output_dir=$OUT_ROOT/$mslug/$domain" \
        --set "tokenization.report_path=artifacts/token_report_${mslug}_${domain}.json" \
        token-report || echo "[grid] token-report failed for $model/$domain" ) &
    tr_pids+=($!)
  done
  wait "${tr_pids[@]}" 2>/dev/null || true

  # (c) the model's 12 units: domain x mode x {base, sft}
  units=()
  for domain in "${DOMAIN_LIST[@]}"; do
    for mode in $MODES; do
      for kind in base sft; do units+=("$model|$domain|$mode|$kind"); done
    done
  done
  echo "[grid] $model: dispatching ${#units[@]} units across GPUs [$GPUS]"
  dispatch_units "${units[@]}"

  # (d) per-domain Base-vs-SFT comparison for this model
  for domain in "${DOMAIN_LIST[@]}"; do
    cout="$OUT_ROOT/$mslug/$domain"; reports=()
    for mode in $MODES; do
      [ -f "$cout/Base-$mode/metrics.json" ] && reports+=("Base-$mode=$cout/Base-$mode/metrics.json")
      [ -f "$cout/SFT-$mode-$domain/metrics.json" ] && reports+=("SFT-$mode=$cout/SFT-$mode-$domain/metrics.json")
    done
    [ ${#reports[@]} -gt 0 ] && python -m taskbench_sft.cli --config "$CONFIG" compare --reports "${reports[@]}" --out "$cout/comparison.md"
  done

  # (e) delete this model's weights from disk before the next model
  if [ "$DELETE_MODELS" = 1 ] && [ ! -d "$model" ]; then
    hub="$(hub_dir_for "$model")"
    if [ -d "$hub" ]; then echo "[grid] deleting model cache: $hub"; rm -rf "$hub"; fi
  fi
  echo "[grid] MODEL $model done."
done

# ---- Phase 3: grand comparison over every cell (results survive model deletion) ----
GRAND=()
for model in "${MODEL_LIST[@]}"; do
  mslug=$(slugify "$model")
  for domain in "${DOMAIN_LIST[@]}"; do
    cout="$OUT_ROOT/$mslug/$domain"
    for mode in $MODES; do
      [ -f "$cout/Base-$mode/metrics.json" ] && GRAND+=("$mslug:$domain:Base-$mode=$cout/Base-$mode/metrics.json")
      [ -f "$cout/SFT-$mode-$domain/metrics.json" ] && GRAND+=("$mslug:$domain:SFT-$mode=$cout/SFT-$mode-$domain/metrics.json")
    done
  done
done
if [ ${#GRAND[@]} -gt 0 ]; then
  python -m taskbench_sft.cli --config "$CONFIG" compare --reports "${GRAND[@]}" --out "$OUT_ROOT/grand_comparison.md"
  echo "[grid] GRAND comparison -> $OUT_ROOT/grand_comparison.md"
fi
echo "[grid] ALL DONE."
