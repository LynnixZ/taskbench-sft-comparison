#!/bin/bash
# PART 2 -- the OFFLINE Slurm job.
# PREFERRED: submit via the wrapper so you can pick the GPU count (you won't always
# get 8):   GPUS_PER_JOB=4 bash scripts/submit_unites.sh
# Direct submit also works (uses the #SBATCH defaults below):  sbatch scripts/job_unites.sbatch
# (#SBATCH lines are static -- they can't read a shell variable; the wrapper passes
#  --gres/--cpus/--mem on the CLI, which overrides these defaults.)
#
# ⚠️ DEPLOY CHECK (learned on UNITES): paths below assume your shared dir == $USER.
#   1. The repo + venv + cache MUST live on shared NFS (/playpen-shared), NOT $HOME:
#      /home is login-node local and INVISIBLE to compute nodes.
#   2. Your shared dir name may DIFFER from your login name (e.g. dir 'xinyu' vs
#      $USER 'xinyuzh'). If so, edit WORK_DIR in prep_env.sh/job_env.sh AND the
#      `cd` + `--output/--error` lines here to your real dir.
# ⚠️ These #SBATCH lines are a SMALL FALLBACK for a DIRECT `sbatch job_unites.sbatch`.
# The REAL request comes from submit_unites.sh, which passes --gres/--cpus/--mem on the
# sbatch CLI (those OVERRIDE the directives below). Slurm queues by the REQUEST, so we
# default to 1 GPU here -- NEVER 8 -- so a stray direct submit can't grab 8 GPUs.
#SBATCH --job-name=tb-grid
#SBATCH --partition=a100            # fallback partition; submit_unites.sh uses a100,ada,a6000 (avoid blackwell)
#SBATCH --gres=gpu:1                # FALLBACK 1 GPU. Real count = GPUS_PER_JOB via submit_unites.sh.
#SBATCH --cpus-per-task=12          # ~12 cores/GPU
#SBATCH --mem=50G                   # ~50G/GPU (whole node would be --mem=0; 480G exceeds the ~472GiB limit)
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/%u/tb_work/logs/grid-%j.out   # edit %u if your dir != login name
#SBATCH --error=/playpen-shared/%u/tb_work/logs/grid-%j.err
set -Eeuo pipefail
# REPO_DIR comes from submit_unites.sh via --export=ALL (handles shared dir != $USER).
cd "${REPO_DIR:-/playpen-shared/$USER/taskbench-sft-comparison}"   # shared NFS copy (compute-node visible); NOT $HOME

source scripts/job_env.sh                                  # 设置离线环境变量 + venv + GPUS
DELETE_MODELS="${DELETE_MODELS:-0}" bash scripts/run_grid.sh   # 跑实验
