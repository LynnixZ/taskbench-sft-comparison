#!/usr/bin/env bash
# SOURCE me (don't execute): the ONLINE (PART 1) environment for a CHINA box (AutoDL).
#   source scripts/prep_env_china.sh
# Self-contained: sets paths (WORK_DIR/HF_HOME) AND the China network sources
# (HuggingFace via hf-mirror.com + Xet OFF; Tsinghua PyPI; SJTU cu121 torch).
# NO secrets, safe in git. Override any value by exporting it first. (US: prep_env.sh.)

# ---- paths ----
# China data disk (AutoDL). On US this is /playpen-shared; here a single node owns it.
export WORK_DIR="${WORK_DIR:-/root/autodl-tmp/tb_work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"   # under WORK_DIR (match job_env/prestage!)
mkdir -p "$WORK_DIR" "$WORK_DIR/logs"

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"          # non-gated: no HF_TOKEN needed
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE                # PART 1 is ONLINE; drop any leaked offline flags

# ---- network: HuggingFace via hf-mirror.com, ALWAYS, with Xet OFF. ----
# We do NOT use AutoDL's network_turbo for HF anymore. New huggingface_hub defaults to
# Xet for big weights, and the Xet client (hf-xet) IGNORES http_proxy -> with turbo's
# proxy ON, weights bypass it and crawl (~3 MB/s). Worse, turbo's http_proxy would route
# hf-mirror requests through the US too. So: hf-mirror.com (fast in China over plain
# HTTP, no proxy needed) + Xet OFF. Do not source /etc/network_turbo here.
# Also CLEAR any proxy (a leaked turbo / global AutoDL proxy would route hf-mirror
# through the US and crawl) -- hf-mirror must be hit DIRECTLY.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"               # Xet bypasses mirror/proxy -> OFF (critical)
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" # parallel HTTP on old hf_hub (new ignores it; the deprecation warning is harmless)
echo "[prep_env_china] HuggingFace -> hf-mirror.com (Xet off)"

# pip + torch: AutoDL's proxy does NOT reliably cover PyPI / pytorch.org, and these
# China mirrors are fast + reliable regardless -> always use them.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# SJTU mirrors download.pytorch.org/whl/cu121 -- same cu121 wheels (driver-safe on
# CUDA 12.x), fast from China. (Do NOT use plain Tsinghua PyPI for torch: its default
# 'torch' is the latest cu13 build, which fails on a 12.x driver.)
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu121}"

# Activate the isolated venv if PART 1 already built it, so EVERY later command
# (prestage / run_grid / cli) uses the venv's pinned deps -- NOT the conda base
# python (which otherwise wins and pulls a different, often-broken transformers).
VENV="${VENV_DIR:-$WORK_DIR/taskbench_venv}"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"; echo "[prep_env_china] venv ACTIVATED -> $VENV"
else
  echo "[prep_env_china] venv not built yet -> prestage_all.sh will create it (re-source after)"
fi

echo "[prep_env_china] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME  pip=tsinghua  torch=SJTU-cu121"
[ -n "${HF_TOKEN:-}" ] || echo "[prep_env_china] NOTE: HF_TOKEN unset -> gated models skipped (fine for the smoke model)"
[ -n "${WANDB_API_KEY:-}" ] || echo "[prep_env_china] NOTE: WANDB_API_KEY unset -> online W&B needs it (else offline)"
