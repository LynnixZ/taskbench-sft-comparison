#!/usr/bin/env bash
# Convenience env for a China GPU box -- contains NO secrets, safe to keep in git.
# Usage on any new machine:
#   git clone https://github.com/LynnixZ/taskbench-sft-comparison.git && cd taskbench-sft-comparison
#   source scripts/setup_china.sh
#   export WANDB_API_KEY=<your reusable key>        # the ONLY secret you paste
#   bash scripts/run_smoke_4090.sh
#
# All values respect anything you already exported (so you can override per box).
# On a US/EU server: DON'T use this file -- just set MODEL_NAME/WANDB_API_KEY and run.

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"          # non-gated: no HF_TOKEN needed
export WORK_DIR="${WORK_DIR:-/root/autodl-tmp/tb_work}"   # big data disk (AutoDL)
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"

# --- China mirrors / network workarounds ---
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# Big Xet-backed models (e.g. Qwen3) are served from a US CDN even via hf-mirror,
# and a single stream is slow from China (~8 MB/s). Use the standard HTTP path
# (disable Xet) + hf_transfer's PARALLEL chunked download to saturate bandwidth.
# (hf_transfer is in requirements.txt and installed by the deps stage before any
#  model download.) This combo is portable -- it also helps on US servers.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"   # legacy LFS fast path
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"        # parallel Xet downloads (new models)

echo "[setup_china] MODEL_NAME=$MODEL_NAME  WORK_DIR=$WORK_DIR  HF_ENDPOINT=$HF_ENDPOINT"
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[setup_china] NOTE: WANDB_API_KEY not set -> 'export WANDB_API_KEY=...' for online W&B (else it runs offline)"
fi
