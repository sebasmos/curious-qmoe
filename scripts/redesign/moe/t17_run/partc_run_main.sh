#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
R=$RESULTS/2026-10-25_1415_moe-t17_code-pending; P=${PY:-python}
cd $W/scripts/redesign/moe
for s in 0 1 2; do for arm in twin spec; do
 COOLDOWN=20 $P t17c_train.py --arm $arm --seed $s --steps 300 --eval_every 50 --lr_body 1e-4 --fb $R/partc/fbank --cache $RESULTS/2026-10-25_0312_moe-t12-feasibility_code-e469a35/cache --out $R/partc/runs --device mps
done; done
echo MAIN_DONE > $R/logs/main_done.txt
