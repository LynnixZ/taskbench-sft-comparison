#!/usr/bin/env bash
# 32-sample smoke test: prepare tiny dataset -> train Full JSON (few steps) ->
# train Trajectory (few steps) -> inference -> evaluation. Uses a tiny random
# model so it runs in seconds on CPU. Override the model with --model-name.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/smoke.yaml}"

# 1. Ensure the official data is present.
if [ ! -f data/raw/data_huggingface/data.json ]; then
  echo "[smoke] downloading official TaskBench data..."
  bash scripts/download_data.sh data/raw
fi

# 2. Build the tiny local smoke model if missing.
if [ ! -f artifacts/tiny_model/config.json ]; then
  echo "[smoke] building tiny local model..."
  python scripts/make_tiny_model.py artifacts/tiny_model
fi

# 3. Dataset statistics (sanity).
python -m taskbench_sft.cli --config "$CONFIG" stats --out artifacts/dataset_stats_smoke.json

# 3. Run the full 4-run matrix in smoke mode:
#    builds a tiny split + token report, trains both SFT modes for a few steps,
#    runs base + SFT inference, evaluates, and writes outputs_smoke/comparison.md
python -m taskbench_sft.cli --config "$CONFIG" run-matrix --smoke

echo "[smoke] done. See outputs_smoke/comparison.md"
