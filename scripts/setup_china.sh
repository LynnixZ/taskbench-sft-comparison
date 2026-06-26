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
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"           # under WORK_DIR (match job_env/prestage!)
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE                # PART 1 is ONLINE; drop any leaked offline flags

# --- Network: DEFAULT to AutoDL academic acceleration whenever it exists. ---
# It proxies git + githubusercontent + huggingface.co. THE KEY: disable Xet. The Xet
# client (hf-xet) does NOT honor http_proxy, so with Xet ON the big weights bypass the
# proxy and crawl (~3 MB/s) -- this bit us repeatedly. With Xet OFF, weights stream over
# plain HTTP THROUGH the proxy (fast), and hf_transfer parallelizes. No network_turbo
# (non-AutoDL box) -> fall back to hf-mirror.com.
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
  unset HF_ENDPOINT                                                 # official huggingface.co, via the proxy
  echo "[setup_china] AutoDL network_turbo ON -> official HF + git through the proxy (Xet off)"
else
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "[setup_china] no network_turbo -> hf-mirror.com for HuggingFace"
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"               # Xet bypasses the proxy -> OFF (critical)
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" # parallel HTTP download

# pip + torch: AutoDL's proxy does NOT reliably cover PyPI / pytorch.org, and these
# China mirrors are fast + reliable regardless -> always use them.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# SJTU mirrors download.pytorch.org/whl/cu121 -- same cu121 wheels (driver-safe on
# CUDA 12.x), fast from China. (Do NOT use plain Tsinghua PyPI for torch: its default
# 'torch' is the latest cu13 build, which fails on a 12.x driver.)
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu121}"

echo "[setup_china] MODEL_NAME=$MODEL_NAME  WORK_DIR=$WORK_DIR  pip=tsinghua  torch=SJTU-cu121"
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[setup_china] NOTE: WANDB_API_KEY not set -> 'export WANDB_API_KEY=...' for online W&B (else it runs offline)"
fi
