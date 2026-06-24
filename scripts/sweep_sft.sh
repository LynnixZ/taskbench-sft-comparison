#!/usr/bin/env bash
# Hyperparameter sweep for the SFT format-comparison experiment.
#
# Runs the Base baselines ONCE, then trains + tests several SFT hyperparameter
# groups per mode (full_json / trajectory). Each SFT group is W&B-monitored
# (train/loss, train/grad_norm, lr, eval/node_f1, eval/edge_f1, ...), so you can
# watch validation improve -- or gradients explode -- live, then compare TEST
# results against the Base baseline in outputs/sweep_comparison.md.
#
# Designed for a big GPU / US cluster. Edit GROUPS below to define your own.
#   export EXPERIMENT_RUN_ID=exp-$(date +%Y%m%d)   # stable W&B resume ids
#   export WANDB_API_KEY=...                        # + source scripts/setup_US.sh
#   bash scripts/sweep_sft.sh
set -Eeuo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/experiment.yaml}"
MODES="${MODES:-full_json trajectory}"
SPLIT="${TEST_SPLIT:-test_all}"
OUT="${OUTPUT_DIR:-outputs}"
CLI="python -m taskbench_sft.cli --config $CONFIG"

# Scale knobs (handy when validating the sweep on a single GPU / in China):
#   MAX_STEPS=50          -> cap optimizer steps per SFT run (fast pipeline check)
#   ONLY_GROUPS="a b"     -> run a subset of the groups below
#   MODES=trajectory      -> one mode only
EXTRA=""
[ -n "${MAX_STEPS:-}" ] && EXTRA="--set training.max_steps=$MAX_STEPS"

# ---- Hyperparameter groups: name -> '--set ...' overrides. EDIT THESE. ----
# Vary the levers that matter most for QLoRA SFT: learning rate, LoRA rank,
# epochs, grad clipping. A high LR group is included to probe gradient explosion.
declare -A GROUPS=(
  [lr1e4_r16]="--set training.learning_rate=1e-4 --set lora.r=16 --set lora.alpha=32"
  [lr2e4_r16]="--set training.learning_rate=2e-4 --set lora.r=16 --set lora.alpha=32"
  [lr5e4_r16]="--set training.learning_rate=5e-4 --set lora.r=16 --set lora.alpha=32"
  [lr2e4_r32]="--set training.learning_rate=2e-4 --set lora.r=32 --set lora.alpha=64"
  [lr1e3_r16]="--set training.learning_rate=1e-3 --set lora.r=16 --set lora.alpha=32"  # likely to blow up
)

# ---- 1. Data + split + token preflight (once) ----
$CLI stats
$CLI split
$CLI token-report

REPORTS=()

# ---- 2. Base baselines, once per mode (no SFT) ----
for mode in $MODES; do
  run="Base-$mode"
  $CLI infer --mode "$mode" --run-name "$run" --split "$SPLIT"
  $CLI evaluate --mode "$mode" \
      --predictions "$OUT/$run/predictions_$SPLIT.jsonl" \
      --out "$OUT/$run/metrics.json"
  REPORTS+=("$run=$OUT/$run/metrics.json")
done

# ---- 3. SFT groups: train (W&B) -> test inference -> evaluate ----
GROUP_NAMES="${ONLY_GROUPS:-${!GROUPS[@]}}"
for group in $GROUP_NAMES; do
  over="${GROUPS[$group]} $EXTRA"
  for mode in $MODES; do
    run="SFT-$mode-$group"
    echo "================= $run :: $over ================="
    # train fails (e.g. gradient explosion / NaN loss) are tolerated: log + skip.
    if ! python -m taskbench_sft.cli --config "$CONFIG" $over train --mode "$mode" --run-name "$run"; then
      echo "[sweep] WARN: training failed for $run (diverged?), skipping its test eval"
      continue
    fi
    adapter="$OUT/$run/best_by_common_score"
    [ -d "$adapter" ] || adapter="$OUT/$run/best_by_loss"
    python -m taskbench_sft.cli --config "$CONFIG" $over infer \
        --mode "$mode" --run-name "$run" --adapter "$adapter" --split "$SPLIT"
    python -m taskbench_sft.cli --config "$CONFIG" $over evaluate \
        --mode "$mode" \
        --predictions "$OUT/$run/predictions_$SPLIT.jsonl" \
        --out "$OUT/$run/metrics.json"
    REPORTS+=("$run=$OUT/$run/metrics.json")
  done
done

# ---- 4. Comparison table across Base + all SFT groups (TEST set) ----
$CLI compare --reports "${REPORTS[@]}" --out "$OUT/sweep_comparison.md"
echo "[sweep] DONE. Test comparison: $OUT/sweep_comparison.md"
