#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# one CPU job at a time (coordinator, swap): TAU runs after the UrbanSound8K lane has finished; waits while swap used > 9.5 GB
P=${PY:-python}
M=$W/scripts/redesign/moe
R=$RESULTS/2026-10-25_1415_moe-t17_code-pending
while pgrep -f "t17_jobs.py --bed urban" > /dev/null; do sleep 60; done
cd $M
for f in 1 2 3 4 5; do for s in 0 1 2; do
  while [ $(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9]+).*/\1/') -gt 9500 ]; do sleep 120; done
  nice -n 10 $P t17_jobs.py --bed tau --out $R --threads 1 --seeds $s --folds $f
done; done
echo TAU_DONE > $R/logs/tau_done.txt
