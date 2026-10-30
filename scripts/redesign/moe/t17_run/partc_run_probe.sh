#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
R=$RESULTS/2026-10-25_1415_moe-t17_code-pending; P=${PY:-python}
cd $W/scripts/redesign/moe
for lb in 1e-4 3e-5; do
 COOLDOWN=20 $P t17c_train.py --arm twin --seed 0 --steps 600 --lr_body $lb --fb $R/partc/fbank --cache $RESULTS/2026-10-25_0312_moe-t12-feasibility_code-e469a35/cache --out $R/partc/probe --probe --device mps
done
echo PROBE_DONE > $R/logs/probe_done.txt
