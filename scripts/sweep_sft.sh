#!/usr/bin/env bash
# Hyperparameter sweep for the SFT format-comparison experiment.
#
# Runs the Base baselines, then trains + tests several SFT hyperparameter groups
# per mode (full_json / trajectory). Each SFT run is a SEPARATE process and W&B
# run (train/loss, grad_norm, lr; eval/node_f1, ...), so you can watch validation
# improve -- or gradients explode -- live, then compare TEST results against Base
# in outputs/sweep_comparison.md.
#
# MULTI-GPU: set GPUS to a list and the independent runs are dispatched across
# them in parallel (one run per GPU). This is the right multi-GPU mode for a
# sweep -- each 8B QLoRA run fits on one GPU; we parallelize across runs, not
# within a run.
#
#   export EXPERIMENT_RUN_ID=exp-$(date +%Y%m%d)   # stable W&B resume ids
#   source scripts/setup_US.sh ; export WANDB_API_KEY=...
#   GPUS="0 1 2 3" bash scripts/sweep_sft.sh        # 4-way parallel
#
# Scale knobs (single GPU / China validation):
#   MAX_STEPS=50          cap optimizer steps per SFT run (fast pipeline check)
#   ONLY_GROUPS="a b"     run a subset of the groups below
#   MODES=trajectory      one mode only
#   CONFIG=configs/experiment_4090.yaml   24GB-safe config
set -Eeuo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/experiment.yaml}"
MODES="${MODES:-full_json trajectory}"
SPLIT="${TEST_SPLIT:-test_all}"
OUT="${OUTPUT_DIR:-outputs}"
GPUS="${GPUS:-}"                 # e.g. "0 1 2 3"; empty -> serial on the default device
CLI="python -m taskbench_sft.cli --config $CONFIG"
EXTRA=""
[ -n "${MAX_STEPS:-}" ] && EXTRA="--set training.max_steps=$MAX_STEPS"

# ---- Hyperparameter groups: name -> '--set ...' overrides. EDIT THESE. ----
declare -A GROUPS=(
  [lr1e4_r16]="--set training.learning_rate=1e-4 --set lora.r=16 --set lora.alpha=32"
  [lr2e4_r16]="--set training.learning_rate=2e-4 --set lora.r=16 --set lora.alpha=32"
  [lr5e4_r16]="--set training.learning_rate=5e-4 --set lora.r=16 --set lora.alpha=32"
  [lr2e4_r32]="--set training.learning_rate=2e-4 --set lora.r=32 --set lora.alpha=64"
  [lr1e3_r16]="--set training.learning_rate=1e-3 --set lora.r=16 --set lora.alpha=32"  # likely to blow up
)
GROUP_NAMES="${ONLY_GROUPS:-${!GROUPS[@]}}"

# ---- 1. Data + split + token preflight (serial; shared by every run) ----
$CLI stats
$CLI split
$CLI token-report

# Pre-cache the model once so parallel workers don't race on the first download.
if [ -n "$GPUS" ]; then
  SWEEP_CONFIG="$CONFIG" python - <<'PY' || echo "[sweep] WARN: model pre-cache skipped"
import os
from pathlib import Path
import yaml
model = os.environ.get("MODEL_NAME") or yaml.safe_load(open(os.environ["SWEEP_CONFIG"]))["model"]["name"]
if not Path(model).exists():
    from huggingface_hub import snapshot_download
    snapshot_download(model, ignore_patterns=["original/*", "*.pth"])
    print("pre-cached", model)
else:
    print("local model dir, no pre-cache needed:", model)
PY
fi

# ---- worker functions (one self-contained run) ----
run_base() {
  local mode="$1" gpu="$2" run="Base-$mode"
  [ -n "$gpu" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  $CLI infer --mode "$mode" --run-name "$run" --split "$SPLIT"
  $CLI evaluate --mode "$mode" \
      --predictions "$OUT/$run/predictions_$SPLIT.jsonl" --out "$OUT/$run/metrics.json"
}
run_sft() {
  local group="$1" mode="$2" gpu="$3" run="SFT-$mode-$group" over="${GROUPS[$group]} $EXTRA"
  [ -n "$gpu" ] && export CUDA_VISIBLE_DEVICES="$gpu"
  echo "================= $run (gpu=${gpu:-default}) :: $over ================="
  if ! python -m taskbench_sft.cli --config "$CONFIG" $over train --mode "$mode" --run-name "$run"; then
    echo "[sweep] WARN: $run training failed (diverged?); skipping its test eval"
    return 0
  fi
  local adapter="$OUT/$run/best_by_common_score"
  [ -d "$adapter" ] || adapter="$OUT/$run/best_by_loss"
  python -m taskbench_sft.cli --config "$CONFIG" $over infer \
      --mode "$mode" --run-name "$run" --adapter "$adapter" --split "$SPLIT"
  python -m taskbench_sft.cli --config "$CONFIG" $over evaluate \
      --mode "$mode" \
      --predictions "$OUT/$run/predictions_$SPLIT.jsonl" --out "$OUT/$run/metrics.json"
}
dispatch() {  # dispatch one "kind|a[|b]" job, optionally pinned to a GPU
  local job="$1" gpu="$2" kind a b
  IFS='|' read -r kind a b <<< "$job"
  if [ "$kind" = base ]; then run_base "$a" "$gpu"; else run_sft "$a" "$b" "$gpu"; fi
}

# ---- 2. Build the job list (Base baselines + SFT groups) ----
JOBS=()
for mode in $MODES; do JOBS+=("base|$mode"); done
for group in $GROUP_NAMES; do for mode in $MODES; do JOBS+=("sft|$group|$mode"); done; done

# ---- 3. Run jobs: parallel across GPUS (one per GPU) or serial ----
if [ -n "$GPUS" ]; then
  read -ra GPU_ARR <<< "$GPUS"; NG=${#GPU_ARR[@]}
  echo "[sweep] parallel across $NG GPU(s): $GPUS"
  i=0
  for job in "${JOBS[@]}"; do
    ( dispatch "$job" "${GPU_ARR[$((i % NG))]}" ) &
    i=$((i + 1))
    (( i % NG == 0 )) && wait    # one batch (NG jobs, distinct GPUs) at a time
  done
  wait
else
  for job in "${JOBS[@]}"; do dispatch "$job" ""; done
fi

# ---- 4. Comparison table over whatever completed (TEST set) ----
REPORTS=()
for mode in $MODES; do
  [ -f "$OUT/Base-$mode/metrics.json" ] && REPORTS+=("Base-$mode=$OUT/Base-$mode/metrics.json")
done
for group in $GROUP_NAMES; do for mode in $MODES; do
  r="SFT-$mode-$group"
  [ -f "$OUT/$r/metrics.json" ] && REPORTS+=("$r=$OUT/$r/metrics.json")
done; done
$CLI compare --reports "${REPORTS[@]}" --out "$OUT/sweep_comparison.md"
echo "[sweep] DONE. Test comparison: $OUT/sweep_comparison.md"
