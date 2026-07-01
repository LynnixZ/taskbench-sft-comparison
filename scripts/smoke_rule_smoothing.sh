#!/usr/bin/env bash
# Small A/B for rule-aware label smoothing (method 2), TRAJECTORY mode.
# Trains the SAME (model, domain) twice -- baseline vs rule-smoothing -- then prints
# the EM of each so you can see whether the method helps. Run on a GPU node, in the
# repo, with the venv active + deps installed + TaskBench data present.
#
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct DOMAIN=data_dailylifeapis MAX_STEPS=80 \
#     bash scripts/smoke_rule_smoothing.sh
#
# Knobs: MODEL, DOMAIN, MAX_STEPS (smoke cap), ALPHA (alpha_max), SPAN_DECAY (0=first-token
#        only, 1=flat span, 0.5=decay [default]), CONFIG, OUT.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/experiment_gnn4plan.yaml}"   # GNN4Plan-aligned (chain-only; comparable to GRAFT/GTool)
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DOMAIN="${DOMAIN:-data_dailylifeapis}"
MAX_STEPS="${MAX_STEPS:-80}"
ALPHA="${ALPHA:-0.1}"
OUT="${OUT:-outputs/rs_smoke}"
SPLIT="${TEST_SPLIT:-test_all}"
PY="${PYBIN:-python} -m taskbench_sft.cli --config $CONFIG"
BASE=(--set "data.domains=[\"$DOMAIN\"]" --set "split.out_dir=artifacts/splits/$DOMAIN" --set "model.name=$MODEL")
SMOKE=(--set "training.max_steps=$MAX_STEPS" --set "training.eval_strategy=steps"
       --set "training.eval_steps=$MAX_STEPS" --set "training.early_stopping_patience=null"
       --set "eval.max_val_samples=16" --set "eval.max_val_eval_samples=8")

# GNN4Plan-aligned config reads vendored data (data/gnn4plan) -> fetch it once if missing.
case "$CONFIG" in *gnn4plan*) [ -d data/gnn4plan ] || bash scripts/download_gnn4plan.sh data/gnn4plan ;; esac

echo "[rs-smoke] split $DOMAIN"
$PY "${BASE[@]}" split

run_variant() {  # $1=tag  $2=extra --set (rule on/off)
  local tag="$1"; shift
  local cout="$OUT/$tag"
  echo "[rs-smoke] === $tag : train ==="
  $PY "${BASE[@]}" "${SMOKE[@]}" "$@" --set "output_dir=$cout" train --mode trajectory --run-name "$tag"
  local adapter="$cout/$tag/best_by_common_score"; [ -d "$adapter" ] || adapter="$cout/$tag/best_by_loss"
  echo "[rs-smoke] === $tag : infer+eval ==="
  $PY "${BASE[@]}" --set "output_dir=$cout" infer --mode trajectory --run-name "$tag" --adapter "$adapter" --split "$SPLIT"
  $PY "${BASE[@]}" --set "output_dir=$cout" evaluate --mode trajectory \
    --predictions "$cout/$tag/predictions_$SPLIT.jsonl" --out "$cout/$tag/metrics.json"
}

run_variant baseline   --set "training.rule_smoothing.enabled=false"
run_variant rulesmooth --set "training.rule_smoothing.enabled=true" \
                       --set "training.rule_smoothing.alpha_max=$ALPHA" \
                       --set "training.rule_smoothing.span_decay=${SPAN_DECAY:-0.5}"

echo; echo "[rs-smoke] ===== EM (chain) / EM (overall) ====="
${PYBIN:-python} - "$OUT" <<'PY'
import json, sys, glob
for tag in ["baseline","rulesmooth"]:
    fs=glob.glob(f"{sys.argv[1]}/{tag}/**/metrics.json", recursive=True)
    if not fs: print(f"  {tag}: (no metrics)"); continue
    g=json.load(open(fs[0]))["groups"]
    ov=g["overall"]["overall"]["trajectory_exact_match"]
    ch=g.get("topology",{}).get("chain",{}).get("trajectory_exact_match")
    print(f"  {tag:<10} overall EM={ov*100:.1f}  chain EM={(ch*100 if ch is not None else float('nan')):.1f}")
PY
echo "[rs-smoke] done -> $OUT"
