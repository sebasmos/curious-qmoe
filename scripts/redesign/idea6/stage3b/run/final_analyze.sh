#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Idea 6 Stage 3b: the four task analyses in parallel (run_all.sh runs them one after another; the CPU was free once
# every head had finished), then the summary. Same commands and arguments as run_all.sh.
R=$RESULTS/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db
S2=$RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba
I5=$RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38
I7=$RESULTS/2026-10-22_1105_redesign-idea7-stage1_code-e2af723
PY=${PY:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
for t in esc50 esc5 urban esc5off; do
  nice -n 10 $PY $W/scripts/redesign/idea6/stage3b/analyze_3b.py --task $t --r $R --s2 $S2 --i5 $I5 --i7 $I7 > $R/logs/analyze_$t.log 2>&1 &
done
wait
date '+%F %T analyzed' >> $R/logs/final_run.time
nice -n 10 $PY $W/scripts/redesign/idea6/stage3b/summary_3b.py --r $R --s2 $S2 > $R/logs/summary.log 2>&1
date '+%F %T summary exit '$? >> $R/logs/final_run.time
