#!/bin/bash
# Progress of Phase 3B jobs: finished-file counts + live worker PIDs.
cd "$(dirname "$0")/.."
d3b=$(ls .cache/scores/D3 2>/dev/null | grep -Ec "leaf0\.npz$")
d3l=$(ls .cache/scores/D3 2>/dev/null | grep -Ec "leaf(5|20)\.npz$")
oot=$(ls .cache/oot3b 2>/dev/null | grep -c "\.pkl$")
echo "D3 base score files : $d3b / 200"
echo "D3 leaf-sweep files : $d3l / 24"
echo "OOT seed checkpoints: $oot / 45"
echo "OOT final CSV       : $(ls experiments/results/phase3b_oot.csv 2>/dev/null || echo missing)"
echo "workers:"
if [ -f .cache/logs/pids.txt ]; then
  while read -r name pid; do
    if kill -0 "$pid" 2>/dev/null; then echo "  $name $pid RUNNING"; else echo "  $name $pid finished/dead"; fi
  done < .cache/logs/pids.txt
fi
echo "newest score file   : $(ls -t .cache/scores/D3 2>/dev/null | head -1)"
