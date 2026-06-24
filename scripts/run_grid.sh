#!/usr/bin/env bash
# Experiment GRID: each MODEL, WITHIN each DOMAIN separately, runs the 4 settings
# (Base/SFT x Full-JSON/Trajectory). Per-domain isolation: each domain gets its
# own split; the model is SFT-trained on that domain's train and tested on its test.
#
# Scheduling: a SINGLE global work-queue over ALL experiment units
# (model x domain x mode x {Base, SFT}) is dispatched across the GPUs -- GPUs are
# NEVER idle between models. Units are queued model-major (a model's units cluster
# together), and a model's weights are DELETED from the HF cache the moment its
# LAST unit finishes (completion-triggered, no barrier). So disk holds only the
# few models currently in flight. Set DELETE_MODELS=0 to keep all weights.
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

# Dynamic GPU scheduler uses associative arrays + `wait -n` (bash >= 4.3).
if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]:-0}" -lt 3 ]; }; then
  echo "FATAL: this script needs bash >= 4.3 (found $BASH_VERSION). On macOS: 'brew install bash'." >&2
  exit 1
fi

CONFIG="${CONFIG:-configs/experiment_models.yaml}"
MODES="${MODES:-full_json trajectory}"
SPLIT="${TEST_SPLIT:-test_all}"
OUT_ROOT="${OUTPUT_DIR:-outputs/grid}"
GPUS="${GPUS:-}"
DELETE_MODELS="${DELETE_MODELS:-1}"
HF_HUB_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"

# Smoke knobs: MAX_STEPS caps steps + step-based eval + no early stop; INFER_LIMIT
# caps how many test samples each inference generates.
EXTRA=()
if [ -n "${MAX_STEPS:-}" ]; then
  # Smoke: cap steps, eval ONCE at the end, disable early stop, and shrink BOTH
  # the eval_loss val set and the generation-based val eval (else fixed eval cost
  # dwarfs the few training steps -> minutes of low-util eval per run).
  EXTRA=(--set "training.max_steps=$MAX_STEPS" --set "training.eval_strategy=steps"
         --set "training.eval_steps=$MAX_STEPS" --set "training.early_stopping_patience=null"
         --set "eval.max_val_samples=16" --set "eval.max_val_eval_samples=8")
fi
LIMIT_ARGS=()
[ -n "${INFER_LIMIT:-}" ] && LIMIT_ARGS=(--limit "$INFER_LIMIT")

# --------------------------------------------------------------------------- #
# Dependency self-healing: run the CLI via pyrun(); on "No module named X" it
# pip-installs X (serialized across parallel cells with a lock) and retries.
# Also ensures core deps up front, so a bare conda base env still works.
# --------------------------------------------------------------------------- #
PYBIN="${PYBIN:-python}"
PIP_LOCK="${TMPDIR:-/tmp}/taskbench_pip.lock"
pip_name() { case "$1" in
  sklearn) echo scikit-learn ;; cv2) echo opencv-python ;; PIL) echo Pillow ;;
  yaml) echo pyyaml ;; *) echo "$1" ;; esac; }
pyrun() {
  local rc miss pkg attempt log
  for attempt in 1 2 3; do
    log="$(mktemp)"
    set +e; "$PYBIN" -m taskbench_sft.cli "$@" 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}; set -e
    [ "$rc" -eq 0 ] && { rm -f "$log"; return 0; }
    miss="$(grep -oE "No module named '[^']+'" "$log" | head -1 | sed -E "s/.*'([^']+)'.*/\1/")"
    rm -f "$log"
    [ -z "$miss" ] && return "$rc"
    pkg="$(pip_name "$miss")"
    echo "[grid][autoinstall] missing module '$miss' -> pip install $pkg (attempt $attempt)"
    ( flock 9; "$PYBIN" -m pip install -q "$pkg" || true ) 9>"$PIP_LOCK"
  done
  return "$rc"
}
if ! "$PYBIN" -c "import sklearn, Levenshtein, yaml, pydantic, networkx, transformers, peft" 2>/dev/null; then
  echo "[grid] installing project dependencies (pip install -r requirements.txt)..."
  ( flock 9; "$PYBIN" -m pip install -q -r requirements.txt || true ) 9>"$PIP_LOCK"
fi

DOMAINS_DEFAULT=(data_huggingface data_multimedia data_dailylifeapis)
if [ -n "${DOMAINS:-}" ]; then read -ra DOMAIN_LIST <<< "$DOMAINS"; else DOMAIN_LIST=("${DOMAINS_DEFAULT[@]}"); fi
MODELS_DEFAULT=(
  "Qwen/Qwen3-8B" "Qwen/Qwen2.5-1.5B-Instruct" "lmsys/vicuna-7b-v1.5"
  "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3"
)
if [ -n "${MODELS:-}" ]; then read -ra MODEL_LIST <<< "$MODELS"; else MODEL_LIST=("${MODELS_DEFAULT[@]}"); fi

slugify() { echo "$1" | tr '/:' '__'; }
hub_dir_for() { echo "$HF_HUB_CACHE_DIR/models--$(echo "$1" | sed 's#/#--#g')"; }
MAX_CACHED="${MAX_CACHED:-2}"   # max base models kept on disk at once (prefetch throttle)

fetch_model() {  # idempotent download (skips local paths); returns non-zero on failure
  local model="$1"
  [ -d "$model" ] && return 0
  MODEL_ID="$model" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["MODEL_ID"], token=os.environ.get("HF_TOKEN") or None,
                  ignore_patterns=["original/*", "*.pth", "*.gguf", "consolidated*"])
PY
}
count_cached() {  # how many of our models are currently on disk
  local n=0 m
  for m in "${MODEL_LIST[@]}"; do [ -d "$(hub_dir_for "$m")" ] && n=$((n + 1)); done
  echo "$n"
}
maybe_delete_model() {
  local model="$1"
  [ "$DELETE_MODELS" = 1 ] || return 0
  [ -d "$model" ] && return 0                 # local path, never delete
  local hub; hub="$(hub_dir_for "$model")"
  [ -d "$hub" ] && { echo "[grid] model $model fully done -> deleting $hub"; rm -rf "$hub"; }
}

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
    pyrun "${base[@]}" infer --mode "$mode" --run-name "Base-$mode" --split "$SPLIT" "${LIMIT_ARGS[@]}"
    pyrun "${base[@]}" evaluate --mode "$mode" \
        --predictions "$cout/Base-$mode/predictions_$SPLIT.jsonl" --out "$cout/Base-$mode/metrics.json"
  else
    if pyrun "${base[@]}" "${EXTRA[@]}" train --mode "$mode" --run-name "SFT-$mode-$domain"; then
      local adapter="$cout/SFT-$mode-$domain/best_by_common_score"
      [ -d "$adapter" ] || adapter="$cout/SFT-$mode-$domain/best_by_loss"
      pyrun "${base[@]}" infer --mode "$mode" --run-name "SFT-$mode-$domain" --adapter "$adapter" --split "$SPLIT" "${LIMIT_ARGS[@]}"
      pyrun "${base[@]}" evaluate --mode "$mode" \
          --predictions "$cout/SFT-$mode-$domain/predictions_$SPLIT.jsonl" --out "$cout/SFT-$mode-$domain/metrics.json"
    else
      echo "[grid] WARN: SFT-$mode/$domain diverged for $model; skipping"
    fi
  fi
}

# ---- Phase 0: ensure the official TaskBench data is present (auto-download) ----
if [ ! -f data/raw/data_huggingface/data.json ]; then
  echo "[grid] TaskBench data missing -> downloading..."
  bash scripts/download_data.sh data/raw
fi

# ---- Phase 1: per-domain splits (depend only on domain + seed; shared by all models) ----
for domain in "${DOMAIN_LIST[@]}"; do
  echo "[grid] split for domain $domain"
  pyrun --config "$CONFIG" \
    --set "data.domains=[\"$domain\"]" --set "split.out_dir=artifacts/splits/$domain" split
done

# ---- Build ALL units (model-major order) + per-model remaining counts ----
UNITS=()
declare -A REMAIN=()
for model in "${MODEL_LIST[@]}"; do
  for domain in "${DOMAIN_LIST[@]}"; do
    for mode in $MODES; do
      for kind in base sft; do
        UNITS+=("$model|$domain|$mode|$kind")
        REMAIN["$model"]=$(( ${REMAIN["$model"]:-0} + 1 ))
      done
    done
  done
done
echo "[grid] ${#UNITS[@]} total units across ${#MODEL_LIST[@]} models; GPUs=[${GPUS:-serial}]"

# ---- Prefetch: fetch the first model synchronously (so its units don't idle on
# download), then prefetch the rest in the background, throttled to keep at most
# MAX_CACHED models on disk (downloads overlap with compute; disk stays bounded). ----
if [ -n "$GPUS" ]; then
  echo "[grid] fetching first model ${MODEL_LIST[0]}"
  fetch_model "${MODEL_LIST[0]}" || echo "[grid] WARN: could not fetch ${MODEL_LIST[0]} (gated/offline?)"
  (
    for model in "${MODEL_LIST[@]:1}"; do
      [ -d "$model" ] && continue
      while [ "$(count_cached)" -ge "$MAX_CACHED" ]; do sleep 15; done
      echo "[grid][prefetch] $model"
      fetch_model "$model" || echo "[grid][prefetch] WARN: $model not fetched (gated/offline?)"
    done
  ) &
fi

# ---- Phase 2: ONE global work-queue; delete a model when its last unit finishes ----
finish_pid() {  # bookkeeping for a finished pid: free GPU, decrement model, maybe delete
  local pid="$1"
  free+=("${PID_GPU[$pid]}")
  local m="${PID_MODEL[$pid]}"
  REMAIN["$m"]=$(( REMAIN["$m"] - 1 ))
  [ "${REMAIN["$m"]}" -le 0 ] && maybe_delete_model "$m"
  unset 'PID_GPU['"$pid"']' 'PID_MODEL['"$pid"']'
}

if [ -n "$GPUS" ]; then
  read -ra GPU_ARR <<< "$GPUS"
  declare -A PID_GPU=() PID_MODEL=()
  free=("${GPU_ARR[@]}"); idx=0
  while [ "$idx" -lt ${#UNITS[@]} ] || [ ${#PID_GPU[@]} -gt 0 ]; do
    while [ ${#free[@]} -gt 0 ] && [ "$idx" -lt ${#UNITS[@]} ]; do
      gpu="${free[0]}"; free=("${free[@]:1}")
      IFS='|' read -r m d mo k <<< "${UNITS[$idx]}"; idx=$((idx + 1))
      ( run_unit "$m" "$d" "$mo" "$k" "$gpu" ) &
      PID_GPU[$!]="$gpu"; PID_MODEL[$!]="$m"
    done
    wait -n 2>/dev/null || wait
    for pid in "${!PID_GPU[@]}"; do
      kill -0 "$pid" 2>/dev/null || finish_pid "$pid"
    done
  done
else
  # serial: still delete a model right after its last unit
  for u in "${UNITS[@]}"; do
    IFS='|' read -r m d mo k <<< "$u"
    run_unit "$m" "$d" "$mo" "$k" ""
    REMAIN["$m"]=$(( REMAIN["$m"] - 1 ))
    [ "${REMAIN["$m"]}" -le 0 ] && maybe_delete_model "$m"
  done
fi

# ---- Phase 3: per-(model,domain) comparison + grand comparison (results survive deletion) ----
GRAND=()
for model in "${MODEL_LIST[@]}"; do
  mslug=$(slugify "$model")
  for domain in "${DOMAIN_LIST[@]}"; do
    cout="$OUT_ROOT/$mslug/$domain"; reports=()
    for mode in $MODES; do
      if [ -f "$cout/Base-$mode/metrics.json" ]; then
        reports+=("Base-$mode=$cout/Base-$mode/metrics.json")
        GRAND+=("$mslug:$domain:Base-$mode=$cout/Base-$mode/metrics.json")
      fi
      if [ -f "$cout/SFT-$mode-$domain/metrics.json" ]; then
        reports+=("SFT-$mode=$cout/SFT-$mode-$domain/metrics.json")
        GRAND+=("$mslug:$domain:SFT-$mode=$cout/SFT-$mode-$domain/metrics.json")
      fi
    done
    [ ${#reports[@]} -gt 0 ] && pyrun --config "$CONFIG" compare --reports "${reports[@]}" --out "$cout/comparison.md"
  done
done
if [ ${#GRAND[@]} -gt 0 ]; then
  pyrun --config "$CONFIG" compare --reports "${GRAND[@]}" --out "$OUT_ROOT/grand_comparison.md"
  echo "[grid] GRAND comparison -> $OUT_ROOT/grand_comparison.md"
fi
echo "[grid] ALL DONE."
