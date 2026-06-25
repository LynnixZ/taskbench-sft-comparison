#!/bin/bash
# ============================================================================
# PART 2 of 2 -- the OFFLINE Slurm job.  Submit from the login node:
#
#   sbatch scripts/job_unites.sbatch
#
# Runs on a compute node (unites2-9, NO internet) and reads everything from the
# cache that PART 1 (prepare_unites.sh) staged onto /playpen-shared.  Nothing
# here touches the network -- HF_HUB_OFFLINE / WANDB_MODE=offline enforce that.
#
# Override per submission, e.g.:  sbatch -p ada --gres=gpu:4 scripts/job_unites.sbatch
# ============================================================================
#SBATCH --job-name=tb-grid
#SBATCH --partition=a100            # unites5: 8x A100 80GB (Ampere, cu121 torch OK). AVOID blackwell.
#SBATCH --gres=gpu:8                # 8 = the per-user GPU cap
#SBATCH --cpus-per-task=96
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00           # generous; the grid usually finishes in well under a day
#SBATCH --output=/playpen-shared/%u/tb_work/logs/grid-%j.out
#SBATCH --error=/playpen-shared/%u/tb_work/logs/grid-%j.err
set -Eeuo pipefail

cd "$HOME/taskbench-sft-comparison"
[ -f "$HOME/env.sh" ] && source "$HOME/env.sh"   # WANDB_API_KEY (used only by a later 'wandb sync')

export WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
source "$WORK_DIR/venv/bin/activate"

# --- OFFLINE: compute nodes have no internet. Use the pre-staged cache only. ---
# (Do NOT source setup_US.sh here -- it would UNSET HF_HUB_OFFLINE.)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline                       # sync later from unites1 (see README note)
export WANDB_DIR="${WANDB_DIR:-$WORK_DIR/wandb}" # offline logs land on shared disk -> sync-able from login node
mkdir -p "$WANDB_DIR"
export EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-grid-$SLURM_JOB_ID}"

# Use exactly the GPUs Slurm pinned to this job (appear as 0..N-1 inside the job).
NGPU=$(nvidia-smi -L | wc -l)
export GPUS="${GPUS:-$(seq -s' ' 0 $((NGPU-1)))}"
echo "[job] $(hostname)  GPUS='$GPUS'  WORK_DIR=$WORK_DIR  (offline cache)"

# Keep models on disk (DELETE_MODELS=0): the node is offline, so a requeue/rerun
# can reuse them -- deleting would make them unrecoverable here.
DELETE_MODELS="${DELETE_MODELS:-0}" bash scripts/run_grid.sh
