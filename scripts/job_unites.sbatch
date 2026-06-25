#!/bin/bash
# PART 2 -- the OFFLINE Slurm job. Submit from the login node:  sbatch scripts/job_unites.sbatch
# Runs on a compute node (no internet) against the cache staged by PART 1.
# Override per submission, e.g.:  sbatch -p ada --gres=gpu:4 scripts/job_unites.sbatch
#
# ⚠️ DEPLOY CHECK (learned on UNITES): paths below assume your shared dir == $USER.
#   1. The repo + venv + cache MUST live on shared NFS (/playpen-shared), NOT $HOME:
#      /home is login-node local and INVISIBLE to compute nodes.
#   2. Your shared dir name may DIFFER from your login name (e.g. dir 'xinyu' vs
#      $USER 'xinyuzh'). If so, edit WORK_DIR in prep_env.sh/job_env.sh AND the
#      `cd` + `--output/--error` lines here to your real dir.
#SBATCH --job-name=tb-grid
#SBATCH --partition=a100            # unites5: 8x A100 80GB (Ampere, cu121 torch OK). AVOID blackwell.
#SBATCH --gres=gpu:8                # 8 = per-user GPU cap
#SBATCH --cpus-per-task=96
#SBATCH --mem=0                     # whole-node RAM (matches gpu:8). NOTE: 480G EXCEEDS the
                                    # a100/ada physical limit (~472GiB) -> rejected. For a
                                    # partial-GPU job use a concrete value, e.g. --mem=150G.
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/%u/tb_work/logs/grid-%j.out   # edit %u if your dir != login name
#SBATCH --error=/playpen-shared/%u/tb_work/logs/grid-%j.err
set -Eeuo pipefail
cd "/playpen-shared/$USER/taskbench-sft-comparison"   # shared NFS copy (compute-node visible); NOT $HOME

source scripts/job_env.sh                                  # 设置离线环境变量 + venv + GPUS
DELETE_MODELS="${DELETE_MODELS:-0}" bash scripts/run_grid.sh   # 跑实验
