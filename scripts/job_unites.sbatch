#!/bin/bash
# PART 2 -- the OFFLINE Slurm job. Submit from the login node:  sbatch scripts/job_unites.sbatch
# Runs on a compute node (no internet) against the cache staged by PART 1.
# Override per submission, e.g.:  sbatch -p ada --gres=gpu:4 scripts/job_unites.sbatch
#SBATCH --job-name=tb-grid
#SBATCH --partition=a100            # unites5: 8x A100 80GB (Ampere, cu121 torch OK). AVOID blackwell.
#SBATCH --gres=gpu:8                # 8 = per-user GPU cap
#SBATCH --cpus-per-task=96
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/%u/tb_work/logs/grid-%j.out
#SBATCH --error=/playpen-shared/%u/tb_work/logs/grid-%j.err
set -Eeuo pipefail
cd "$HOME/taskbench-sft-comparison"

source scripts/job_env.sh                                  # 设置离线环境变量 + venv + GPUS
DELETE_MODELS="${DELETE_MODELS:-0}" bash scripts/run_grid.sh   # 跑实验
