#!/usr/bin/env bash
# AutoDL convenience: use AutoDL's ACADEMIC ACCELERATION proxy instead of mirrors.
# Gives direct (proxied) access to the OFFICIAL HuggingFace / PyPI / GitHub /
# pytorch.org -- so NO HF_ENDPOINT / PIP_INDEX_URL / TORCH_INDEX_URL juggling, and
# none of the mirror-vs-official version mismatches (e.g. the cu13 torch trap).
# SOURCE me (don't execute):   source scripts/setup_autodl.sh
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
  echo "[setup_autodl] network_turbo ON -> official HF/PyPI/GitHub/pytorch.org via proxy"
else
  echo "[setup_autodl] WARN: /etc/network_turbo not found (not an AutoDL node?) -- falling back to whatever network you have"
fi

export WORK_DIR="${WORK_DIR:-/root/autodl-tmp/tb_work}"
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_home}"

# Use OFFICIAL sources: clear any mirror/offline leftovers so nothing forces a mirror.
unset HF_ENDPOINT PIP_INDEX_URL TORCH_INDEX_URL HF_HUB_DISABLE_XET HF_HUB_OFFLINE

# Parallel downloads still help over the proxy.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "[setup_autodl] WORK_DIR=$WORK_DIR  HF_HOME=$HF_HOME  (no mirrors)"
