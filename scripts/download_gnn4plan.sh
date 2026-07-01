#!/usr/bin/env bash
# Vendor GNN4TaskPlan's data (the dataset GNN4Plan/GRAFT/GTool all benchmark on)
# into our raw-dir layout, so split.mode=gnn4plan can use the FIXED split_ids.json
# test set + the exact same samples. Run on a node WITH internet (PART 1).
#
#   bash scripts/download_gnn4plan.sh [DEST]      # default DEST=data/gnn4plan
#
# Their data.json/tool_desc.json are TaskBench-format -> our loader parses them as-is
# (user_request/task_nodes/task_links + desc on tools). split_ids.json = {"test_ids":
# {"chain":[...500 ids...]}}.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-data/gnn4plan}"
BASE="https://raw.githubusercontent.com/WxxShirley/GNN4TaskPlan/main/data"
# GNN4TaskPlan dir name : our domain dir name
# (ultratool + tmdb are GNN4Plan's two NON-TaskBench benchmarks; small JSONs, harmless
#  to always vendor -- ultratool used by configs/experiment_ultratool.yaml; tmdb for baseline/transfer.)
PAIRS="huggingface:data_huggingface multimedia:data_multimedia dailylife:data_dailylifeapis ultratool:data_ultratool tmdb:data_tmdb"
FILES="data.json tool_desc.json graph_desc.json user_requests.json split_ids.json"

for pair in $PAIRS; do
  g="${pair%%:*}"; o="${pair##*:}"
  mkdir -p "$DEST/$o"
  for f in $FILES; do
    echo "[gnn4plan] $g/$f -> $DEST/$o/$f"
    curl -fsSL "$BASE/$g/$f" -o "$DEST/$o/$f"
  done
  # quick sanity: how many test chains
  n=$(python3 -c "import json;print(len(json.load(open('$DEST/$o/split_ids.json'))['test_ids']['chain']))" 2>/dev/null || echo '?')
  echo "[gnn4plan] $o: test chains = $n"
done
echo "[gnn4plan] done -> $DEST  (set data.raw_dir=$DEST + split.mode=gnn4plan)"
