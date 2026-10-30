#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# When lane 2 (ESC-50, TAU) has written all 30 job files, it takes over UrbanSound8K seed 2 then seed 1 from the other end (a finished job file is skipped).
P=${PY:-python}
M=$W/scripts/redesign/moe
R=$RESULTS/2026-10-25_1415_moe-t17_code-pending
until [ $(ls $R/esc50/raw $R/tau/raw 2>/dev/null | grep -c npz) -ge 30 ]; do sleep 60; done
sleep 30
cd $M; nice -n 10 $P t17_jobs.py --bed urban --out $R --threads 1 --seeds 2 1
