#!/usr/bin/env bash
# SOURCE me (don't execute): environment + paths for the ONLINE prep on unites1.
#   source scripts/prep_env.sh
# Edit WORK_DIR here if you want a different shared location.

# Shared storage visible from EVERY compute node (NOT local /playpen, /data, or $HOME).
# NOTE: your shared dir name may DIFFER from $USER (e.g. dir 'xinyu' vs $USER 'xinyuzh').
# If so, set WORK_DIR to the real path here (the offline job reads from exactly here).
export WORK_DIR="${WORK_DIR:-/playpen-shared/$USER/tb_work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
mkdir -p "$WORK_DIR" "$WORK_DIR/logs"

# Official HF/PyPI sources + parallel Xet/hf_transfer; clears mirror/offline leftovers.
source "$(dirname "${BASH_SOURCE[0]}")/setup_US.sh"

echo "[prep_env] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME"
[ -n "${HF_TOKEN:-}" ] || echo "[prep_env] WARN: HF_TOKEN unset -> gated models (Llama-2/3.2, Mistral) will be SKIPPED"
