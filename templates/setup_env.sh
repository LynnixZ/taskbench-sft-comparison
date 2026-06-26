#!/usr/bin/env bash
# ============================================================================
# PORTABLE environment bootstrap (PART 1).  Copy this single file into ANY repo.
#
# What it does, in order:
#   1. Pick the network automatically: AutoDL academic-acceleration > China
#      mirrors > official (US/EU).
#   2. Create an ISOLATED venv (NOT coupled to base/conda -- that coupling is the
#      root of the "old base torch -> pip upgrades to a cu13 wheel" mess).
#   3. Install OUR OWN cu121 torch (driver-safe on CUDA 12.x), then install the
#      requirements with torch PINNED so unpinned deps can't swap in a cu13 wheel.
#   GPU-independent: it only installs wheels (no GPU needed); GPU usability is
#   verified when you actually run (see the final hint).
#
# Usage (all knobs optional):
#   WORK_DIR=/data/myproj REQUIREMENTS=requirements.txt bash templates/setup_env.sh
#   source /data/myproj/venv/bin/activate
#
# Knobs:
#   WORK_DIR          where the venv + constraint live      (default: ./_env)
#   VENV_DIR          venv path                              (default: $WORK_DIR/venv)
#   REQUIREMENTS      requirements file to install           (default: requirements.txt)
#   REGION            auto | china | china_turbo | us        (default: auto)
#   TORCH_INDEX_URL   override the torch wheel index
#   PIP_INDEX_URL     override the pip index
#   VENV_SYSTEM_SITE  =1 to reuse base/conda packages (NOT recommended)
#
# For the OFFLINE run later (compute node / no internet) set, before running:
#   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
# ============================================================================
set -Eeuo pipefail

WORK_DIR="${WORK_DIR:-./_env}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/venv}"
REQUIREMENTS="${REQUIREMENTS:-requirements.txt}"
REGION="${REGION:-auto}"
mkdir -p "$WORK_DIR"
log(){ echo "[setup_env] $*"; }

# ---- 1. Network: auto-detect region ----
if [ "$REGION" = auto ]; then
  if [ -f /etc/network_turbo ]; then
    REGION=china_turbo
  elif curl -fsS --max-time 4 https://huggingface.co/ >/dev/null 2>&1; then
    REGION=us
  else
    REGION=china
  fi
fi
case "$REGION" in
  china_turbo)   # AutoDL academic acceleration: official HF/github via proxy
    # shellcheck disable=SC1091
    source /etc/network_turbo
    unset HF_ENDPOINT HF_HUB_DISABLE_XET
    export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu121}"
    log "region=china_turbo (official HF/github proxied; tsinghua pip; SJTU cu121 torch)" ;;
  china)         # mainland mirrors
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cu121}"
    log "region=china (hf-mirror; tsinghua pip; SJTU cu121 torch)" ;;
  us)            # official sources
    unset HF_ENDPOINT HF_HUB_DISABLE_XET PIP_INDEX_URL || true
    export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
    export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
    log "region=us (official HF/PyPI; pytorch.org cu121 torch)" ;;
  *) log "unknown REGION=$REGION (use auto|china|china_turbo|us)"; exit 1 ;;
esac

# ---- 2. Isolated venv ----
VENV_FLAGS=""; [ "${VENV_SYSTEM_SITE:-0}" = 1 ] && VENV_FLAGS="--system-site-packages"
[ -d "$VENV_DIR" ] || { log "creating venv (${VENV_FLAGS:-isolated}) at $VENV_DIR"; python3 -m venv $VENV_FLAGS "$VENV_DIR"; }
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install -q --upgrade pip wheel setuptools

# ---- 3. cu121 torch (install our own; detect a CUDA *build*, not a live GPU) ----
if ! python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
  log "installing cu121 torch from $TORCH_INDEX_URL"
  pip install torch --index-url "$TORCH_INDEX_URL"
fi
TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
log "pinning torch==$TORCH_VER (blocks a cu13 upgrade during the deps resolve)"
echo "torch==$TORCH_VER" > "$WORK_DIR/torch.constraint"

# ---- 4. Requirements (torch pinned) ----
if [ -f "$REQUIREMENTS" ]; then
  log "installing $REQUIREMENTS (torch pinned)"
  pip install -r "$REQUIREMENTS" -c "$WORK_DIR/torch.constraint"
else
  log "no $REQUIREMENTS found -- skipping (set REQUIREMENTS=...)"
fi

log "DONE. venv = $VENV_DIR"
log "VERIFY on a GPU node:  python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'   # want '...+cu121 True'"
