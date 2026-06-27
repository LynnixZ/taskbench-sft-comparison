#!/usr/bin/env bash
# SOURCE me (don't execute): the ONLINE (PART 1) environment for a CHINA box (AutoDL).
#   source scripts/prep_env_china.sh
# Self-contained: sets paths (WORK_DIR/HF_HOME) AND the China network sources (AutoDL
# academic acceleration if present, else hf-mirror; Tsinghua PyPI; SJTU cu121 torch).
# NO secrets, safe in git. Override any value by exporting it first. (US: prep_env.sh.)

# ---- paths ----
# China data disk (AutoDL). On US this is /playpen-shared; here a single node owns it.
export WORK_DIR="${WORK_DIR:-/root/autodl-tmp/tb_work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"   # under WORK_DIR (match job_env/prestage!)
mkdir -p "$WORK_DIR" "$WORK_DIR/logs"

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"          # non-gated: no HF_TOKEN needed
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE                # PART 1 is ONLINE; drop any leaked offline flags

# ---- network: DEFAULT to AutoDL academic acceleration whenever it exists. ----
# It proxies git + githubusercontent + huggingface.co. THE KEY: disable Xet. The Xet
# client (hf-xet) does NOT honor http_proxy, so with Xet ON the big weights bypass the
# proxy and crawl (~3 MB/s) -- this bit us repeatedly. With Xet OFF, weights stream over
# plain HTTP THROUGH the proxy (fast), and hf_transfer parallelizes. No network_turbo
# (non-AutoDL box) -> fall back to hf-mirror.com.
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
  unset HF_ENDPOINT                                                 # official huggingface.co, via the proxy
  echo "[prep_env_china] AutoDL network_turbo ON -> official HF + git through the proxy (Xet off)"
else
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "[prep_env_china] no network_turbo -> hf-mirror.com for HuggingFace"
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

echo "[prep_env_china] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME  pip=tsinghua  torch=SJTU-cu121"
[ -n "${HF_TOKEN:-}" ] || echo "[prep_env_china] NOTE: HF_TOKEN unset -> gated models skipped (fine for the smoke model)"
[ -n "${WANDB_API_KEY:-}" ] || echo "[prep_env_china] NOTE: WANDB_API_KEY unset -> online W&B needs it (else offline)"
