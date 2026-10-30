#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Idea 6 Stage 3b final pass (23 Oct 2026, every extraction and head finished): score what is left, analyze the four
# tasks, summarize.
export R=$RESULTS/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db
export S2=$RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba
export I5=$RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38
export I7=$RESULTS/2026-10-22_1105_redesign-idea7-stage1_code-e2af723
export PY=${PY:-python}
date '+%F %T start' > $R/logs/final_run.time
bash $W/scripts/redesign/idea6/stage3b/run_all.sh score || exit 1
date '+%F %T scored' >> $R/logs/final_run.time
bash $W/scripts/redesign/idea6/stage3b/run_all.sh analyze || exit 1
date '+%F %T analyzed' >> $R/logs/final_run.time
bash $W/scripts/redesign/idea6/stage3b/run_all.sh summary || exit 1
date '+%F %T done' >> $R/logs/final_run.time
