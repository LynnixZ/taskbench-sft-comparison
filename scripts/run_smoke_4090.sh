#!/usr/bin/env bash
# =============================================================================
# Unattended NVIDIA-GPU smoke test for the TaskBench SFT comparison.
#
# Scheduler-neutral: run it directly, or as the executable of an HTCondor /
# Slurm / PBS batch job. It is fully non-interactive — it never waits for input.
#
# Pipeline:
#   read env -> clone/update repo -> create/reuse venv -> install deps
#   -> check CUDA/GPU/HF_TOKEN -> verify model access -> check W&B
#   -> tiny Node/Chain split -> Base x2 + SFT x2 (QLoRA) -> evaluate
#   -> save logs + predictions -> package results as .tar.gz
#
# On ANY failure: records the failed stage, keeps full logs, packages whatever
# results exist, and exits non-zero.
#
# Secrets (HF_TOKEN, WANDB_API_KEY) are read from the environment and never
# printed. No usernames/tokens/private paths are hard-coded.
# =============================================================================
set -Eeuo pipefail

# --------------------------------------------------------------------------- #
# 0. Configuration via environment (with safe defaults)
# --------------------------------------------------------------------------- #
# If you copy ONLY this script to the server, the script clones the code itself.
# Set EXPERIMENT_REPO_URL to your repo, OR edit DEFAULT_REPO_URL just below.
DEFAULT_REPO_URL="https://github.com/LynnixZ/taskbench-sft-comparison.git"
EXPERIMENT_REPO_URL="${EXPERIMENT_REPO_URL:-$DEFAULT_REPO_URL}"
EXPERIMENT_REPO_BRANCH="${EXPERIMENT_REPO_BRANCH:-main}"
WORK_DIR="${WORK_DIR:-$PWD/taskbench_smoke_work}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/outputs_smoke_gpu}"
HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
CONFIG_FILE="${CONFIG_FILE:-configs/smoke_gpu.yaml}"

# Stable experiment id so resubmits resume W&B runs instead of duplicating them.
EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-smoke-$(date -u +%Y%m%d)-$$}"

# W&B (key from env only; provide sensible defaults for the rest)
export WANDB_PROJECT="${WANDB_PROJECT:-taskbench-sft-smoke}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-llama2-7b-4090-smoke}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_LOG_MODEL="false"

export HF_HOME MODEL_NAME OUTPUT_DIR EXPERIMENT_RUN_ID
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$WORK_DIR" "$OUTPUT_DIR" "$HF_HOME"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/run_smoke_4090.log"

# Mirror all stdout/stderr to the main log.
exec > >(tee -a "$MAIN_LOG") 2>&1

CURRENT_STAGE="init"
STATUS_FILE="$OUTPUT_DIR/status.json"

log()   { echo "[$(date -u +%H:%M:%S)] [$CURRENT_STAGE] $*"; }
stage() { CURRENT_STAGE="$1"; echo "============================================================"; log "STAGE: $1"; }

write_status() {
  local state="$1"
  cat > "$STATUS_FILE" <<EOF
{"experiment_run_id": "$EXPERIMENT_RUN_ID", "stage": "$CURRENT_STAGE", "state": "$state", "model": "$MODEL_NAME", "timestamp_utc": "$(date -u +%FT%TZ)"}
EOF
}

package_results() {
  # Best-effort packaging of whatever exists (offline W&B dir included).
  local tarball="$WORK_DIR/results_${EXPERIMENT_RUN_ID}.tar.gz"
  log "packaging results -> $tarball"
  tar -czf "$tarball" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")" 2>/dev/null || true
  echo "$tarball" > "$OUTPUT_DIR/RESULTS_TARBALL_PATH.txt" 2>/dev/null || true
  log "results tarball: $tarball"
}

on_error() {
  local exit_code=$?
  log "ERROR: stage '$CURRENT_STAGE' failed (exit $exit_code)"
  write_status "failed"
  # Diagnostics snapshot.
  {
    echo "=== failed_stage: $CURRENT_STAGE (exit $exit_code) ==="
    echo "=== nvidia-smi ==="; nvidia-smi 2>&1 || echo "nvidia-smi unavailable"
    echo "=== pip freeze ==="; pip freeze 2>&1 || true
    echo "=== env (secrets redacted) ==="; env | grep -viE 'TOKEN|KEY|SECRET|PASSWORD' | sort || true
  } > "$LOG_DIR/diagnostics_${CURRENT_STAGE}.log" 2>&1 || true
  package_results
  exit "$exit_code"
}
trap on_error ERR

write_status "running"

# --------------------------------------------------------------------------- #
# 1. Repo: clone or update (skip if already inside the repo)
# --------------------------------------------------------------------------- #
stage "repo"
# Priority: (a) already inside a checkout -> use it; otherwise (b) clone the repo.
if [ -f "$PWD/requirements.txt" ] && [ -d "$PWD/taskbench_sft" ]; then
  log "running inside an existing checkout: $PWD"
elif [ -n "$EXPERIMENT_REPO_URL" ]; then
  REPO_DIR="$WORK_DIR/repo"
  if [ -d "$REPO_DIR/.git" ]; then
    log "updating existing clone at $REPO_DIR"
    git -C "$REPO_DIR" fetch --depth 1 origin "$EXPERIMENT_REPO_BRANCH"
    git -C "$REPO_DIR" checkout "$EXPERIMENT_REPO_BRANCH"
    git -C "$REPO_DIR" reset --hard "origin/$EXPERIMENT_REPO_BRANCH"
  else
    log "cloning $EXPERIMENT_REPO_URL (branch $EXPERIMENT_REPO_BRANCH)"
    git clone --depth 1 --branch "$EXPERIMENT_REPO_BRANCH" "$EXPERIMENT_REPO_URL" "$REPO_DIR"
  fi
  cd "$REPO_DIR"
else
  log "FATAL: no code found in \$PWD and EXPERIMENT_REPO_URL is empty."
  log "Either run this script from inside the repo, or set:"
  log "  export EXPERIMENT_REPO_URL=https://github.com/<you>/<repo>.git   # public => no auth"
  false
fi
# Sanity: the project files must be present after this stage.
if [ ! -f requirements.txt ] || [ ! -d taskbench_sft ]; then
  log "FATAL: project files (requirements.txt / taskbench_sft) not found after repo stage"; false
fi
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
log "working dir: $PWD | git commit: $GIT_COMMIT"

# --------------------------------------------------------------------------- #
# 2. Python environment: create or reuse
# --------------------------------------------------------------------------- #
stage "venv"
VENV_DIR="${VENV_DIR:-$WORK_DIR/venv}"
if [ ! -d "$VENV_DIR" ]; then
  log "creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null

# --------------------------------------------------------------------------- #
# 3. Dependencies
# --------------------------------------------------------------------------- #
stage "deps"
# The default PyPI torch wheel may be built for a newer CUDA than the GPU driver
# supports (e.g. driver 535/CUDA 12.2 cannot run a cu130 wheel). Install a
# CUDA-matched torch FIRST; override the channel with TORCH_INDEX_URL if needed.
#   driver CUDA 12.x (>=525)  -> cu121  (default)
#   driver CUDA 11.8          -> cu118  (set TORCH_INDEX_URL accordingly)
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  log "torch already CUDA-capable: $(python -c 'import torch;print(torch.__version__)')"
else
  TORCH_SPEC="torch"
  [ -n "${TORCH_VERSION:-}" ] && TORCH_SPEC="torch==$TORCH_VERSION"
  log "installing CUDA-matched torch ($TORCH_SPEC) from $TORCH_INDEX_URL (force-reinstall)"
  pip install --force-reinstall "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
fi
log "installing requirements"
pip install -r requirements.txt
pip install -e . >/dev/null 2>&1 || true
# bitsandbytes is required for QLoRA on CUDA; ensure it is present on Linux.
python -c "import bitsandbytes" 2>/dev/null || pip install bitsandbytes || log "WARN: bitsandbytes install failed (QLoRA will fall back to LoRA)"

# --------------------------------------------------------------------------- #
# 4. CUDA / GPU check
# --------------------------------------------------------------------------- #
stage "gpu_check"
nvidia-smi || { log "FATAL: nvidia-smi not available"; false; }
python - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available to torch"); sys.exit(1)
print("gpu:", torch.cuda.get_device_name(0))
PY

# --------------------------------------------------------------------------- #
# 5. HF token + model access check
# --------------------------------------------------------------------------- #
stage "hf_check"
if [ -z "${HF_TOKEN:-}" ]; then
  log "FATAL: HF_TOKEN is not set (required for gated $MODEL_NAME)"; false
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"   # used by huggingface_hub/transformers
python - <<PY
import os, sys
from huggingface_hub import HfApi
tok = os.environ["HF_TOKEN"]; model = os.environ["MODEL_NAME"]
try:
    info = HfApi().model_info(model, token=tok)
    print(f"OK: model access verified for {model} (sha {info.sha[:12]})")
except Exception as e:
    print(f"FATAL: cannot access {model}: {type(e).__name__}: {e}")
    print("Hint: accept the license at https://huggingface.co/{} and use a token with read access.".format(model))
    sys.exit(1)
PY

# --------------------------------------------------------------------------- #
# 6. W&B check (non-fatal: falls back to offline inside the code)
# --------------------------------------------------------------------------- #
stage "wandb_check"
if [ -z "${WANDB_API_KEY:-}" ]; then
  log "WARN: WANDB_API_KEY not set; switching WANDB_MODE=offline (sync later from the tarball)"
  export WANDB_MODE="offline"
else
  python - <<'PY' || echo "WARN: W&B login check failed; code will fall back to offline"
import wandb
# Does not print the key; just confirms the library is importable + configured.
print("wandb", wandb.__version__, "mode", __import__("os").environ.get("WANDB_MODE"))
PY
fi

# --------------------------------------------------------------------------- #
# 7. Data (download official TaskBench data if missing)
# --------------------------------------------------------------------------- #
stage "data"
if [ ! -f data/raw/data_huggingface/data.json ]; then
  bash scripts/download_data.sh data/raw
else
  log "official data already present"
fi

# --------------------------------------------------------------------------- #
# 8. Run the 4-setting GPU smoke (split -> base x2 -> SFT x2 -> evaluate).
#    Resumable: completed settings (metrics.json present) are skipped.
# --------------------------------------------------------------------------- #
stage "experiment"
python -m taskbench_sft.cli --config "$CONFIG_FILE" gpu-smoke \
    --train-n "${TRAIN_N:-24}" \
    --val-n "${VAL_N:-6}" \
    --test-node-n "${TEST_NODE_N:-4}" \
    --test-chain-n "${TEST_CHAIN_N:-4}"

# --------------------------------------------------------------------------- #
# 9. Package + finish
# --------------------------------------------------------------------------- #
stage "package"
write_status "succeeded"
package_results
log "DONE. Comparison table: $OUTPUT_DIR/comparison.md"
log "Results tarball path recorded in: $OUTPUT_DIR/RESULTS_TARBALL_PATH.txt"
