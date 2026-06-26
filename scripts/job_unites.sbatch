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
# REPO_DIR comes from submit_unites.sh via --export=ALL (handles shared dir != $USER).
cd "${REPO_DIR:-/playpen-shared/$USER/taskbench-sft-comparison}"   # shared NFS copy (compute-node visible); NOT $HOME

source scripts/job_env.sh                                  # 设置离线环境变量 + venv + GPUS
DELETE_MODELS="${DELETE_MODELS:-0}" bash scripts/run_grid.sh   # 跑实验
