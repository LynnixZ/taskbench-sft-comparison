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
PARTITION="${PARTITION:-a100}"               # a100 = Ampere (cu121 OK). AVOID blackwell.
CPUS="${CPUS:-$(( GPUS_PER_JOB * 12 ))}"      # ~12 cores/GPU (Slurm default)
MEM="${MEM:-$(( GPUS_PER_JOB * 50 ))G}"       # ~50G/GPU; 8*50=400G < a100 physical (~472GiB)

echo "[submit] partition=$PARTITION  gpus=$GPUS_PER_JOB  cpus=$CPUS  mem=$MEM"
echo "[submit] CONFIG=${CONFIG:-(default)}  MODES=${MODES:-(default)}  MODELS=${MODELS:-(default 6)}"

# --export=ALL (sbatch default) carries CONFIG/MODES/MODELS/etc. into the job.
sbatch -p "$PARTITION" --gres=gpu:"$GPUS_PER_JOB" --cpus-per-task="$CPUS" --mem="$MEM" \
  --export=ALL scripts/job_unites.sbatch
