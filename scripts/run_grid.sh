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
#   source scripts/prep_env.sh ; export WANDB_API_KEY=... HF_TOKEN=... EXPERIMENT_RUN_ID=grid-$(date +%Y%m%d)
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
# Drop the LoRA adapters + hf_trainer optimizer state once a unit's metrics are
# computed (infer still needs the adapter transiently, so we delete AFTER eval).
# Keeps predictions/metrics/*.json. Set 0 to keep checkpoints for debugging.
DELETE_CHECKPOINTS="${DELETE_CHECKPOINTS:-1}"
HF_HUB_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"

# Capture ALL run output to a log UNDER OUT_ROOT so it travels with the results tarball
# (works on Slurm AND China; per-unit stdout/stderr are interleaved here). On Slurm the
# job's own grid-<id>.out/.err are ALSO copied in at packaging time below.
mkdir -p "$OUT_ROOT/_run_logs"
exec > >(tee "$OUT_ROOT/_run_logs/run_grid.log") 2>&1

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
  local model="$1" domain="$2" mode="$3" kind="$4" gpu="$5" alpha="${6:-}"
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
    # Tolerate a failed Base unit (e.g. a model that won't load) -> WARN + skip, same as SFT,
    # so one bad model/unit never aborts the whole grid.
    if pyrun "${base[@]}" infer --mode "$mode" --run-name "Base-$mode" --split "$SPLIT" "${LIMIT_ARGS[@]}"; then
      pyrun "${base[@]}" evaluate --mode "$mode" \
          --predictions "$cout/Base-$mode/predictions_$SPLIT.jsonl" --out "$cout/Base-$mode/metrics.json" \
          || echo "[grid] WARN: Base-$mode/$domain eval failed for $model; skipping"
    else
      echo "[grid] WARN: Base-$mode/$domain inference failed for $model; skipping"
    fi
  else
    # Run name: bare SFT-mode-domain, or a rule-smoothing sweep variant SFT-...-a<alpha>.
    local rn="SFT-$mode-$domain"
    local -a rule=()
    if [ -n "$alpha" ]; then
      rn="SFT-$mode-$domain-a$alpha"
      case "$alpha" in
        0|0.0) rule=(--set "training.rule_smoothing.enabled=false") ;;         # baseline of the sweep
        *)     rule=(--set "training.rule_smoothing.enabled=true"
                     --set "training.rule_smoothing.alpha_max=$alpha")
               [ -n "${RULE_MAX_LAG:-}" ]    && rule+=(--set "training.rule_smoothing.max_lag=$RULE_MAX_LAG")
               [ -n "${RULE_SPAN_DECAY:-}" ] && rule+=(--set "training.rule_smoothing.span_decay=$RULE_SPAN_DECAY") ;;
      esac
    fi
    local rdir="$cout/$rn"
    if pyrun "${base[@]}" "${rule[@]}" "${EXTRA[@]}" train --mode "$mode" --run-name "$rn"; then
      local adapter="$rdir/best_by_common_score"
      [ -d "$adapter" ] || adapter="$rdir/best_by_loss"
      pyrun "${base[@]}" infer --mode "$mode" --run-name "$rn" --adapter "$adapter" --split "$SPLIT" "${LIMIT_ARGS[@]}"
      pyrun "${base[@]}" evaluate --mode "$mode" \
          --predictions "$rdir/predictions_$SPLIT.jsonl" --out "$rdir/metrics.json"
      # Free disk but KEEP the best adapter: drop only the optimizer state + intermediate
      # checkpoints (hf_trainer) + redundant copies. best_by_common_score (the chosen best)
      # survives so a run can be re-used / re-inferred later.
      if [ "$DELETE_CHECKPOINTS" = 1 ] && [ -f "$rdir/metrics.json" ]; then
        rm -rf "$rdir/last_checkpoint" "$rdir/hf_trainer"
        [ -d "$rdir/best_by_common_score" ] && rm -rf "$rdir/best_by_loss"
      fi
    else
      echo "[grid] WARN: $rn/$domain diverged for $model; skipping"
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

# ---- Pre-flight: keep only models we can actually access. Gated-without-token /
# 404 / offline models are SKIPPED entirely (all their units), so we don't waste
# time failing each unit. We fetch config.json (tiny) AND require a real weights
# file in the SAME snapshot dir -- otherwise a partial/interrupted download (only
# config.json cached) passes preflight then crashes at model load. The weights
# check inspects already-cached files (it does NOT trigger a download), so this is
# cheap and behaves the same online (weights come from a prior prestage) & offline. ----
AVAILABLE=()
for model in "${MODEL_LIST[@]}"; do
  if [ -d "$model" ]; then AVAILABLE+=("$model"); continue; fi
  if MODEL_ID="$model" python - <<'PY' 2>/dev/null
import os, glob
from huggingface_hub import hf_hub_download
cfg = hf_hub_download(os.environ["MODEL_ID"], filename="config.json", token=os.environ.get("HF_TOKEN") or None)
snap = os.path.dirname(cfg)
weights = glob.glob(os.path.join(snap, "**", "*.safetensors"), recursive=True) \
        + glob.glob(os.path.join(snap, "**", "*.bin"), recursive=True)
raise SystemExit(0 if weights else 1)
PY
  then
    AVAILABLE+=("$model")
  else
    echo "[grid] SKIP model $model -- not accessible or no weights cached (gated w/o token / 404 / offline / partial download)"
  fi
done
MODEL_LIST=("${AVAILABLE[@]}")
if [ ${#MODEL_LIST[@]} -eq 0 ]; then
  echo "[grid] FATAL: none of the requested models are accessible (set HF_TOKEN + accept licenses?)"; exit 1
fi
echo "[grid] models to run: ${MODEL_LIST[*]}"

# ---- Build ALL units (model-major order) + per-model remaining counts ----
UNITS=()
declare -A REMAIN=()
# RULE_ALPHAS (e.g. "0 0.05 0.1 0.2"): sweep rule-aware label smoothing on TRAJECTORY
# SFT -- one run per alpha (0 = baseline / smoothing off). Empty -> no sweep.
for model in "${MODEL_LIST[@]}"; do
  for domain in "${DOMAIN_LIST[@]}"; do
    for mode in $MODES; do
      for kind in base sft; do
        if [ "$kind" = sft ] && [ "$mode" = trajectory ] && [ -n "${RULE_ALPHAS:-}" ]; then
          for a in $RULE_ALPHAS; do
            UNITS+=("$model|$domain|$mode|$kind|$a")
            REMAIN["$model"]=$(( ${REMAIN["$model"]:-0} + 1 ))
          done
        else
          UNITS+=("$model|$domain|$mode|$kind|")
          REMAIN["$model"]=$(( ${REMAIN["$model"]:-0} + 1 ))
        fi
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
      IFS='|' read -r m d mo k a <<< "${UNITS[$idx]}"; idx=$((idx + 1))
      ( run_unit "$m" "$d" "$mo" "$k" "$gpu" "$a" ) &
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
    IFS='|' read -r m d mo k a <<< "$u"
    run_unit "$m" "$d" "$mo" "$k" "" "$a" || echo "[grid] WARN: unit $m/$d/$mo failed; continuing"
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
      # Glob picks up the bare SFT-mode-domain AND any rule-smoothing sweep variants
      # (SFT-mode-domain-a<alpha>), so a RULE_ALPHAS sweep all lands in the comparison.
      for sdir in "$cout"/SFT-"$mode"-"$domain"*/; do
        [ -f "$sdir/metrics.json" ] || continue
        tag="$(basename "$sdir")"
        reports+=("$tag=$sdir/metrics.json")
        GRAND+=("$mslug:$domain:$tag=$sdir/metrics.json")
      done
    done
    [ ${#reports[@]} -gt 0 ] && pyrun --config "$CONFIG" compare --reports "${reports[@]}" --out "$cout/comparison.md"
  done
done
if [ ${#GRAND[@]} -gt 0 ]; then
  pyrun --config "$CONFIG" compare --reports "${GRAND[@]}" --out "$OUT_ROOT/grand_comparison.md"
  echo "[grid] GRAND comparison -> $OUT_ROOT/grand_comparison.md"
fi

# ---- Bundle run logs: run_grid.log (the tee'd output) is already under _run_logs;
# on Slurm also copy the job's stdout/stderr so failures (e.g. a model that won't load)
# travel with the tarball for offline debugging. ----
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${WORK_DIR:-}" ]; then
  for ext in out err; do
    s="$WORK_DIR/logs/grid-$SLURM_JOB_ID.$ext"
    [ -f "$s" ] && cp -f "$s" "$OUT_ROOT/_run_logs/" 2>/dev/null || true
  done
fi

# ---- Package the lightweight results (reports + metrics + predictions + run logs; the
# big adapters/checkpoints/wandb dirs are excluded -- they stay under $OUT_ROOT). ----
TARBALL="$(dirname "$OUT_ROOT")/grid_results_${EXPERIMENT_RUN_ID:-$(slugify "${MODEL_LIST[0]}")}.tar.gz"
echo "[grid] packaging results -> $TARBALL"
tar czf "$TARBALL" -C "$OUT_ROOT" \
  --exclude='*best_by_*' --exclude='*last_checkpoint*' --exclude='*hf_trainer*' --exclude='*/wandb' \
  . 2>/dev/null && echo "[grid] results tarball: $(cd "$(dirname "$TARBALL")" && pwd)/$(basename "$TARBALL")" || echo "[grid] WARN: packaging failed"
echo "[grid] ALL DONE."
