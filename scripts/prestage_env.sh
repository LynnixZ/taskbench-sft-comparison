#!/usr/bin/env bash
# =============================================================================
# Pre-stage the environment on a node that HAS internet (e.g. the login node),
# so the GPU batch job (run_smoke_4090.sh) reuses the caches and runs fast.
#
# It builds the venv, installs deps, downloads the model snapshot, and fetches
# the TaskBench data -- all into the SAME WORK_DIR / HF_HOME / VENV_DIR that
# run_smoke_4090.sh uses, so the job skips the slow downloads.
#
# Speed in China: set mirrors before running (these are honored automatically):
#   export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   export HF_ENDPOINT=https://hf-mirror.com
# (Defaults below already point at those mirrors; override if you prefer others.)
#
# Secrets (HF_TOKEN) come from the environment and are never printed.
# =============================================================================
set -Eeuo pipefail

# ---- Mirrors (override via env). pip + huggingface_hub honor these natively. ----
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---- Paths (MUST match run_smoke_4090.sh defaults so the job reuses them) ----
WORK_DIR="${WORK_DIR:-$PWD/taskbench_smoke_work}"
HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/venv}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
export HF_HOME
mkdir -p "$WORK_DIR" "$HF_HOME"

log() { echo "[$(date -u +%H:%M:%S)] [prestage] $*"; }

log "PIP_INDEX_URL=$PIP_INDEX_URL"
log "HF_ENDPOINT=$HF_ENDPOINT"
log "WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME  VENV_DIR=$VENV_DIR"
log "MODEL_NAME=$MODEL_NAME"

# ---- 1. venv + dependencies ----
if [ ! -d "$VENV_DIR" ]; then
  log "creating venv"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Install a CUDA-matched torch FIRST (default PyPI wheel may target a newer CUDA
# than the GPU driver supports). cu121 works with driver >= 525 (CUDA 12.x).
# Override with TORCH_INDEX_URL (e.g. .../whl/cu118) for older drivers.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  log "torch already CUDA-capable: $(python -c 'import torch;print(torch.__version__)')"
else
  TORCH_SPEC="torch"
  [ -n "${TORCH_VERSION:-}" ] && TORCH_SPEC="torch==$TORCH_VERSION"
  log "installing CUDA-matched torch ($TORCH_SPEC) from $TORCH_INDEX_URL"
  pip install --force-reinstall "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
fi

log "installing requirements (via $PIP_INDEX_URL)"
pip install -r requirements.txt
pip install -e . >/dev/null 2>&1 || true
python -c "import bitsandbytes" 2>/dev/null || pip install bitsandbytes || log "WARN: bitsandbytes not installed (QLoRA falls back to LoRA)"

# ---- 2. TaskBench data ----
if [ ! -f data/raw/data_huggingface/data.json ]; then
  log "downloading TaskBench data"
  bash scripts/download_data.sh data/raw
else
  log "data already present"
fi

# ---- 3. Model snapshot into HF cache ----
if [ -z "${HF_TOKEN:-}" ]; then
  log "WARN: HF_TOKEN not set; skipping model pre-download (gated $MODEL_NAME needs a token)."
  log "      Set HF_TOKEN and re-run to pre-cache the model."
else
  log "pre-downloading model snapshot (this is the big one; via $HF_ENDPOINT)"
  python - <<PY
import os
from huggingface_hub import snapshot_download
model = os.environ["MODEL_NAME"]; tok = os.environ["HF_TOKEN"]
path = snapshot_download(
    repo_id=model,
    token=tok,
    # Skip the raw consolidated PyTorch checkpoint; transformers loads HF format.
    ignore_patterns=["original/*", "*.pth"],
)
print("model cached at:", path)
PY
fi

log "DONE. Now submit the GPU job with the SAME WORK_DIR/HF_HOME and it will reuse these caches."
log "  e.g.  WORK_DIR=$WORK_DIR HF_HOME=$HF_HOME bash scripts/run_smoke_4090.sh"
