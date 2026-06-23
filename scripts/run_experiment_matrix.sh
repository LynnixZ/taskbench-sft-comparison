#!/usr/bin/env bash
# Full experiment matrix (the real run; needs a GPU for the default 1.5B model).
# Produces the main comparison table in outputs/comparison.md.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/default.yaml}"

# 1. Data (download once).
if [ ! -f data/raw/data_huggingface/data.json ]; then
  bash scripts/download_data.sh data/raw
fi

# 2. Dataset statistics + stratified split + token-length report.
python -m taskbench_sft.cli --config "$CONFIG" stats
python -m taskbench_sft.cli --config "$CONFIG" split
python -m taskbench_sft.cli --config "$CONFIG" token-report

# 3. Run the 4-run matrix (base x2 prompts, SFT x2 prompts) + comparison table.
python -m taskbench_sft.cli --config "$CONFIG" run-matrix

echo "Done. Main results: outputs/comparison.md"
