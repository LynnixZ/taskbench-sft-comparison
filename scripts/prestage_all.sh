#!/usr/bin/env bash
# One-shot pre-stage onto the DATA DISK: Python venv + dependencies + TaskBench
# data + the comparison models. Auto-detects gated repos -- tries each model
# WITHOUT a token; if gated and HF_TOKEN is set + license accepted it retries
# with the token, otherwise it reports "NEEDS TOKEN" and skips (so the free
# models still get cached). Run this on a node with internet (e.g. login node).
#
# China:  source scripts/setup_china.sh  (mirrors); US: nothing special.
#   export WORK_DIR=/root/autodl-tmp/tb_work HF_HOME=/root/autodl-tmp/hf_home
#   export HF_TOKEN=hf_xxx        # OPTIONAL: only to also fetch gated models
#   bash scripts/prestage_all.sh
set -Eeuo pipefail
cd "$(dirname "$0")/.."

WORK_DIR="${WORK_DIR:-/root/autodl-tmp/tb_work}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/taskbench_venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
export HF_HOME
mkdir -p "$WORK_DIR" "$HF_HOME"

# Instruct models for the comparison (override with MODELS="a b c").
DEFAULT_MODELS=(
  "Qwen/Qwen3-8B"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "lmsys/vicuna-7b-v1.5"
  "meta-llama/Llama-2-7b-chat-hf"
  "meta-llama/Llama-3.2-3B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
)
if [ -n "${MODELS:-}" ]; then read -ra MODEL_LIST <<< "$MODELS"; else MODEL_LIST=("${DEFAULT_MODELS[@]}"); fi

log() { echo "[$(date -u +%H:%M:%S)] [prestage-all] $*"; }
log "WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME"
log "PIP_INDEX_URL=${PIP_INDEX_URL:-(default PyPI)}  HF_ENDPOINT=${HF_ENDPOINT:-(default huggingface.co)}"
[ -n "${HF_TOKEN:-}" ] && log "HF_TOKEN set -> will also try gated models" || log "HF_TOKEN not set -> gated models will be skipped"

# ---- 1. venv + deps ----
# ISOLATED venv by DEFAULT (no --system-site-packages): we install OUR OWN cu121 torch
# into it, fully decoupled from the base/conda env. This is the reproducible choice and
# kills the "base torch is old -> pip upgrades it to a cu13 wheel" mess for good. It
# costs ONE torch download (~2.5GB) per fresh venv. Set VENV_SYSTEM_SITE=1 to instead
# REUSE a base torch (faster, but couples you to whatever versions base ships).
VENV_FLAGS=""; [ "${VENV_SYSTEM_SITE:-0}" = 1 ] && VENV_FLAGS="--system-site-packages"
[ -d "$VENV_DIR" ] || { log "creating venv (${VENV_FLAGS:-isolated})"; python3 -m venv $VENV_FLAGS "$VENV_DIR"; }
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null
# Ensure a CUDA-build torch, then PIN it so the requirements resolve can't UPGRADE
# torch to a newer (cu13) wheel from PyPI -- which fails on a CUDA 12.x driver. (The
# base/image torch is often 2.1; modern unpinned transformers/trl ask for a newer
# torch, so without the pin pip silently swaps in cu13.) Reuse a base CUDA torch when
# present (no multi-GB download); else install cu121 from TORCH_INDEX_URL. Installing
# wheels needs no GPU -- GPU usability is verified later in PART 2.
if ! python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
  log "no CUDA torch found -> installing cu121 torch from $TORCH_INDEX_URL"
  pip install torch --index-url "$TORCH_INDEX_URL"
fi
TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
log "pinning torch==$TORCH_VER for the requirements resolve (blocks a cu13 upgrade)"
echo "torch==$TORCH_VER" > "$WORK_DIR/torch.constraint"
log "installing requirements"
pip install -r requirements.txt -c "$WORK_DIR/torch.constraint"
pip install -e . >/dev/null 2>&1 || true
# Check INSTALLED (don't `import` -- that probes CUDA and warns/errors on a GPU-less node).
python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('bitsandbytes') else 1)" 2>/dev/null \
  || pip install bitsandbytes -c "$WORK_DIR/torch.constraint" || log "WARN: bitsandbytes install failed (needed only for PART 2 training)"

# ---- 2. TaskBench data ----
bash scripts/download_data.sh data/raw

# ---- 3. Models (auto-detect gated) ----
if [ "${SKIP_MODELS:-0}" = 1 ]; then
  log "SKIP_MODELS=1 -> not pre-downloading models (run_grid.sh fetches them on demand)"
  log "done (env + data ready)."
  exit 0
fi
log "downloading models to $HF_HOME ..."
SUMMARY="$WORK_DIR/prestage_models_summary.txt"
: > "$SUMMARY"
for model in "${MODEL_LIST[@]}"; do
  log "=== $model ==="
  status=$(MODEL_ID="$model" python - <<'PY'
import os
from huggingface_hub import snapshot_download
model = os.environ["MODEL_ID"]
token = os.environ.get("HF_TOKEN") or None
try:
    # Skip raw consolidated / GGUF; KEEP .bin (some models, e.g. vicuna, ship only .bin).
    snapshot_download(
        model, token=token,
        ignore_patterns=["original/*", "*.pth", "*.gguf", "consolidated*"],
    )
    print("OK")
except Exception as e:
    msg = str(e).lower()
    if any(s in msg for s in ("gated", "restricted", "401", "403", "awaiting", "access to model")):
        print("NEEDS_TOKEN")
    else:
        print("ERROR:" + type(e).__name__)
PY
)
  echo "$status  $model" | tee -a "$SUMMARY"
done

log "================= SUMMARY ================="
cat "$SUMMARY"
log "OK = cached on data disk; NEEDS_TOKEN = gated (set HF_TOKEN + accept license, re-run)"
log "done."
