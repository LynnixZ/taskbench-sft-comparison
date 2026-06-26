#!/usr/bin/env bash
# Submit the offline grid job, choosing HOW MANY GPUs to request.
#
# Why a wrapper: #SBATCH lines are STATIC -- Slurm parses them before any shell runs,
# so they can't read a shell variable. We instead pass the resource request on the
# sbatch COMMAND LINE here, which OVERRIDES the #SBATCH defaults in job_unites.sbatch.
#
# Usage (you won't always get 8 GPUs -> set GPUS_PER_JOB):
#   GPUS_PER_JOB=4 bash scripts/submit_unites.sh
#   # DAG experiment (pass the config/models too; they propagate via sbatch --export):
#   export CONFIG=configs/experiment_dag_fulljson.yaml MODES=full_json \
#          MODELS="lmsys/vicuna-7b-v1.5 Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3"
#   GPUS_PER_JOB=4 bash scripts/submit_unites.sh
set -e

# ===== EDIT ME: how many GPUs to request (<= your per-user cap, usually 8) =====
GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
# ==============================================================================
# cu121 torch runs on Ampere/Ada (a100/a6000/ada) but NOT Blackwell ("no kernel image").
# Submit to ALL the compatible partitions at once -> Slurm runs on whichever frees up
# first, and never lands on blackwell. (Don't use the default 'all' partition: it includes
# blackwell.) Override with PARTITION=a100 to pin one type.
PARTITION="${PARTITION:-a100,ada,a6000}"
CPUS="${CPUS:-$(( GPUS_PER_JOB * 12 ))}"      # ~12 cores/GPU (Slurm default)
MEM="${MEM:-$(( GPUS_PER_JOB * 50 ))G}"       # ~50G/GPU; 8*50=400G < a100 physical (~472GiB)

# Log path follows WORK_DIR so it lands in YOUR shared dir even when it != $USER
# (the #SBATCH --output=%u default would use the login name, which can be wrong).
WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"
mkdir -p "$WORK_DIR/logs"

echo "[submit] partition=$PARTITION  gpus=$GPUS_PER_JOB  cpus=$CPUS  mem=$MEM"
echo "[submit] WORK_DIR=$WORK_DIR  REPO_DIR=${REPO_DIR:-(\$HOME default)}"
echo "[submit] CONFIG=${CONFIG:-(default)}  MODES=${MODES:-(default)}  MODELS=${MODELS:-(default 6)}"

# --export=ALL (sbatch default) carries WORK_DIR/REPO_DIR/CONFIG/MODES/MODELS into the job.
# --output/--error on the CLI override the static #SBATCH %u paths with the real WORK_DIR.
sbatch -p "$PARTITION" --gres=gpu:"$GPUS_PER_JOB" --cpus-per-task="$CPUS" --mem="$MEM" \
  --output="$WORK_DIR/logs/grid-%j.out" --error="$WORK_DIR/logs/grid-%j.err" \
  --export=ALL scripts/job_unites.sbatch
