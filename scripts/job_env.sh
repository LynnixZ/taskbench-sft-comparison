#!/usr/bin/env bash
# SOURCE me from inside the sbatch job: OFFLINE environment for the compute node.
#   source scripts/job_env.sh
# Compute nodes (unites2-9) have NO internet -- everything reads the cache that
# prep_env.sh + prestage staged onto /playpen-shared.
# WORK_DIR must match prep_env.sh exactly (and your shared dir may differ from $USER,
# e.g. dir 'xinyu' vs $USER 'xinyuzh') -- set it to the real path if so.

export WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
source "${VENV_DIR:-$WORK_DIR/taskbench_venv}/bin/activate"

# Cache-only: never touch the network. (Do NOT source setup_US.sh -- it UNSETs these.)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline                          # sync later from unites1
export WANDB_DIR="${WANDB_DIR:-$WORK_DIR/wandb}"    # offline logs on shared disk -> sync-able
mkdir -p "$WANDB_DIR"
export EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-grid-${SLURM_JOB_ID:-local}}"

# Use exactly the GPUs Slurm pinned to this job (they appear as 0..N-1 inside it).
NGPU=$(nvidia-smi -L | wc -l)
export GPUS="${GPUS:-$(seq -s' ' 0 $((NGPU-1)))}"

echo "[job_env] $(hostname)  GPUS='$GPUS'  WORK_DIR=$WORK_DIR  (offline cache)"
