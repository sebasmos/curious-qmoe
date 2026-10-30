#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# T17 head jobs, two lanes of one thread each, nice 10. Code branch moe/t17 (worktree ~/Documents/qmoe-work/t17).
P=${PY:-python}
M=$W/scripts/redesign/moe
R=$RESULTS/2026-10-25_1415_moe-t17_code-pending
cd $M
nice -n 10 $P t17_check.py --out $R/inputs > $R/logs/check.log 2>&1 || { echo CHECK_FAILED > $R/logs/jobs_failed.txt; exit 1; }
lane1() { nice -n 10 $P t17_jobs.py --bed urban --out $R --threads 1; }
lane2() { for b in esc50 tau; do nice -n 10 $P t17_jobs.py --bed $b --out $R --threads 1; done; }
lane1 > $R/logs/jobs_lane1.log 2>&1 &
lane2 > $R/logs/jobs_lane2.log 2>&1 &
wait
echo ALL_JOBS_DONE > $R/logs/jobs_done.txt
