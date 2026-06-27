#!/usr/bin/env bash
# SOURCE me (don't execute): the ONLINE (PART 1) environment for a US/EU box (unites1).
#   source scripts/prep_env.sh
# Self-contained: sets paths (WORK_DIR/HF_HOME) AND the network sources (official
# HF/PyPI, parallel Xet/hf_transfer). NO secrets, safe in git. Override any value by
# exporting it before sourcing. (China counterpart: prep_env_china.sh.)

# ---- paths ----
# Shared storage visible from EVERY compute node (NOT local /playpen, /data, or $HOME).
# NOTE: your shared dir name may DIFFER from $USER (e.g. dir 'xinyu' vs $USER 'xinyuzh').
# If so, set WORK_DIR to the real path here (the offline job reads from exactly here).
export WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"   # under WORK_DIR (match job_env/prestage!)
mkdir -p "$WORK_DIR" "$WORK_DIR/logs"

# ---- network: official international sources ----
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"                     # non-gated default (legacy single-model smoke)
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"       # parallel Xet downloads (newer models)
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"   # parallel classic-LFS downloads
# Clear any China-mirror / offline leftovers so we use official HF/PyPI + the Xet CDN.
unset HF_ENDPOINT HF_HUB_DISABLE_XET HF_HUB_OFFLINE TRANSFORMERS_OFFLINE PIP_INDEX_URL TASKBENCH_DATA_BASE_URL

echo "[prep_env] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME  (official HF/PyPI, Xet enabled)"
[ -n "${HF_TOKEN:-}" ] || echo "[prep_env] WARN: HF_TOKEN unset -> gated models (Llama-2/3.2, Mistral) will be SKIPPED"
[ -n "${WANDB_API_KEY:-}" ] || echo "[prep_env] NOTE: WANDB_API_KEY unset -> 'export WANDB_API_KEY=...' for online W&B (else offline)"
