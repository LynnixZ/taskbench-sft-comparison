#!/usr/bin/env bash
# Download the official Microsoft JARVIS / TaskBench data for the three domains
# into data/raw/<domain>/. We fetch the *official* files and never modify the
# gold schema -- all normalization happens at load time in code.
#
# Robust to slow/blocked GitHub raw (common in China): tries several mirrors per
# file and uses the first that works. Override / prepend your own with:
#   export TASKBENCH_DATA_BASE_URL="https://raw.gitmirror.com/microsoft/JARVIS/main/taskbench"
#
# Usage: bash scripts/download_data.sh [output_dir]
set -euo pipefail

OUT_DIR="${1:-data/raw}"
DOMAINS=("data_huggingface" "data_multimedia" "data_dailylifeapis")
FILES=("data.json" "tool_desc.json" "graph_desc.json" "user_requests.json")

# Candidate base URLs, tried in order (env override first). jsDelivr is a fast
# global CDN that also works well in China, so it leads; GitHub raw is the
# canonical fallback.
BASES=()
[ -n "${TASKBENCH_DATA_BASE_URL:-}" ] && BASES+=("$TASKBENCH_DATA_BASE_URL")
BASES+=(
  "https://cdn.jsdelivr.net/gh/microsoft/JARVIS@main/taskbench"
  "https://raw.githubusercontent.com/microsoft/JARVIS/main/taskbench"
  "https://ghproxy.net/https://raw.githubusercontent.com/microsoft/JARVIS/main/taskbench"
  "https://raw.gitmirror.com/microsoft/JARVIS/main/taskbench"
)

mkdir -p "$OUT_DIR"

fetch() {
  # fetch <relative_path> <dest>; tries each base URL until one succeeds.
  local rel="$1" dest="$2" base url
  for base in "${BASES[@]}"; do
    url="$base/$rel"
    if curl -fsSL --connect-timeout 15 --max-time 120 --retry 1 "$url" -o "$dest"; then
      echo "[download] OK   $rel   (via ${base%%/*}//${base#*//})"
      return 0
    fi
    echo "[download] miss $rel   (via ${base})"
  done
  echo "[download] FATAL: all mirrors failed for $rel" >&2
  return 1
}

for domain in "${DOMAINS[@]}"; do
  mkdir -p "$OUT_DIR/$domain"
  for f in "${FILES[@]}"; do
    dest="$OUT_DIR/$domain/$f"
    [ -s "$dest" ] && { echo "[download] skip $domain/$f (exists)"; continue; }
    fetch "$domain/$f" "$dest"
  done
done

echo "[download] computing file hashes -> $OUT_DIR/SHA256SUMS.txt"
( cd "$OUT_DIR" && find . -name '*.json' -type f | sort | while read -r p; do
    shasum -a 256 "$p" 2>/dev/null || sha256sum "$p"
  done ) > "$OUT_DIR/SHA256SUMS.txt" 2>/dev/null || true
echo "[download] done."
