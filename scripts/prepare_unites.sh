#!/usr/bin/env bash
# ============================================================================
# PART 1 of 2 -- ONLINE PREP.  Run this on the LOGIN node (unites1), which has
# internet.  It downloads EVERYTHING (python env + TaskBench data + all models)
# onto SHARED storage (/playpen-shared, visible from every Slurm compute node)
# so that PART 2 (the Slurm job) can run fully OFFLINE from the cache.
#
#   ssh <you>@unites1.cs.unc.edu
#   cd ~/taskbench-sft-comparison && git pull
#   bash scripts/prepare_unites.sh           # run inside tmux (long download)
#   # then:  sbatch scripts/job_unites.sbatch
#
# Secrets (HF_TOKEN for gated models, WANDB_API_KEY) live ONLY in ~/env.sh,
# which is gitignored -- never hard-code them here.
# ============================================================================
set -Eeuo pipefail
cd "$(dirname "$0")/.."

# --- secrets / overrides (gitignored, outside the repo) ---
[ -f "$HOME/env.sh" ] && source "$HOME/env.sh"

# --- SHARED storage seen by all compute nodes (NOT local /playpen or /data) ---
export WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
mkdir -p "$WORK_DIR" "$WORK_DIR/logs"

# Official sources + parallel Xet/hf_transfer downloads; clears any mirror/offline
# leftovers (so we actually reach huggingface.co / PyPI from the login node).
source scripts/setup_US.sh

echo "[prepare] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[prepare] WARN: HF_TOKEN not set -> the 3 gated models (Llama-2/3.2, Mistral)"
  echo "[prepare]       will be SKIPPED. Put 'export HF_TOKEN=hf_...' in ~/env.sh and"
  echo "[prepare]       accept their licenses on huggingface.co, then re-run."
fi

# Download env + data + ALL models to the shared disk (SKIP_MODELS=0).
SKIP_MODELS=0 bash scripts/prestage_all.sh

echo
echo "[prepare] ===== model cache summary ====="
cat "$WORK_DIR/prestage_models_summary.txt"
echo "[prepare] All rows should say OK. NEEDS_TOKEN = gated, set HF_TOKEN + accept license."
echo "[prepare] DONE.  Next:  sbatch scripts/job_unites.sbatch"
