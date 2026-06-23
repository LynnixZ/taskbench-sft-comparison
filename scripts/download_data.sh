#!/usr/bin/env bash
# Download the official Microsoft JARVIS / TaskBench data for the three supported
# domains into data/raw/<domain>/. We deliberately fetch the *official* files and
# never modify the gold schema — all normalization happens at load time in code.
#
# Usage: bash scripts/download_data.sh [output_dir]
set -euo pipefail

OUT_DIR="${1:-data/raw}"
BASE_URL="https://raw.githubusercontent.com/microsoft/JARVIS/main/taskbench"
DOMAINS=("data_huggingface" "data_multimedia" "data_dailylifeapis")
FILES=("data.json" "tool_desc.json" "graph_desc.json" "user_requests.json")

mkdir -p "$OUT_DIR"
for domain in "${DOMAINS[@]}"; do
  mkdir -p "$OUT_DIR/$domain"
  for f in "${FILES[@]}"; do
    url="$BASE_URL/$domain/$f"
    dest="$OUT_DIR/$domain/$f"
    echo "[download] $url -> $dest"
    curl -fsSL --retry 3 --max-time 120 "$url" -o "$dest"
  done
done

echo "[download] computing file hashes -> $OUT_DIR/SHA256SUMS.txt"
( cd "$OUT_DIR" && find . -name '*.json' -type f | sort | while read -r p; do
    shasum -a 256 "$p"
  done ) > "$OUT_DIR/SHA256SUMS.txt"
echo "[download] done."
