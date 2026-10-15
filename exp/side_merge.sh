#!/usr/bin/env bash
# Merge the finished side-pool items per experiment: results/n5_umi_v3.json, results/p1_ant{2,4,8}_v3.json.
set -uo pipefail
R=/pscratch/sd/s/sgkim/kcj/airan_recovery
S=$R/results/side
for e in n5_umi_v3 p1_ant2_v3 p1_ant4_v3 p1_ant8_v3; do
  files=$(ls $S/${e}_s*.npz 2>/dev/null | while read f; do [ -s "${f%.npz}.json" ] && echo $f; done)
  [ -n "$files" ] || { echo "$e: no finished item"; continue; }
  echo "== $e"; python3 $R/src/merge_classical.py --inputs $files --output $R/results/$e.json
done
