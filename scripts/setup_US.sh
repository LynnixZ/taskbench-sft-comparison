#!/usr/bin/env bash
# Convenience env for a US/EU server -- official sources, NO mirrors, NO secrets.
# Safe to keep in git. Usage on any new US machine:
#   git clone https://github.com/LynnixZ/taskbench-sft-comparison.git && cd taskbench-sft-comparison
#   source scripts/setup_US.sh
#   export WANDB_API_KEY=<your reusable key>      # the ONLY secret you paste
#   # for a gated model (Llama-2) also: export HF_TOKEN=<your token>  (license accepted)
#   bash scripts/run_smoke_4090.sh
#
# Values respect anything you already exported (override per box as needed).

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"   # non-gated; for Llama set MODEL_NAME + HF_TOKEN
export WORK_DIR="${WORK_DIR:-$HOME/tb_work}"        # override to a big/scratch disk if $HOME is small
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"        # parallel Xet downloads (newer models)
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"   # parallel classic-LFS downloads (other models)

# Actively clear any China-mirror leftovers from a previous session so we use
# the official international sources (huggingface.co, PyPI, HF Xet CDN).
unset HF_ENDPOINT HF_HUB_DISABLE_XET HF_HUB_OFFLINE PIP_INDEX_URL TASKBENCH_DATA_BASE_URL

echo "[setup_us] MODEL_NAME=$MODEL_NAME  WORK_DIR=$WORK_DIR  (official HF/PyPI, Xet enabled)"
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[setup_us] NOTE: WANDB_API_KEY not set -> 'export WANDB_API_KEY=...' for online W&B (else offline)"
fi
