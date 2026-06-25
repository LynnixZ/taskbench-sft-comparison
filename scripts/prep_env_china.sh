#!/usr/bin/env bash
# SOURCE me (don't execute): environment + paths for ONLINE prep on a CHINA box.
#   source scripts/prep_env_china.sh
# Mirror of prep_env.sh -- ONLY the data disk + download sources differ (mirrors).

# China data disk (AutoDL). On US this is /playpen-shared; here a single node owns it.
export WORK_DIR="${WORK_DIR:-/root/autodl-tmp/tb_work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"
mkdir -p "$WORK_DIR" "$WORK_DIR/logs"

# China mirrors (hf-mirror.com, Tsinghua PyPI) + parallel downloads.
source "$(dirname "${BASH_SOURCE[0]}")/setup_china.sh"

echo "[prep_env_china] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME"
[ -n "${HF_TOKEN:-}" ] || echo "[prep_env_china] NOTE: HF_TOKEN unset -> gated models skipped (fine for the smoke model)"
