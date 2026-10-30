#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Final pass (22 Oct 2026 after 15:46, every extraction and head finished): score everything not yet scored, analyze
# the four tasks, E7, summary, manifest.
export R=$RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba S2=$RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38
export I7=$RESULTS/2026-10-22_1105_redesign-idea7-stage1_code-e2af723
export S1=$RESULTS/2026-10-22_0953_redesign-idea6-stage1_code-ec70260
export PY=${PY:-python} HF_HUB_OFFLINE=1
date '+%F %T start' > $R/logs/final_run.time
bash $W/scripts/redesign/idea6/stage2/run_all.sh score analyze || exit 1
date '+%F %T scored and analyzed' >> $R/logs/final_run.time
OMP_NUM_THREADS=2 PYTHONPATH=$W nice -n 10 $PY $W/scripts/redesign/idea6/stage2/realshift.py e7 --results $R --audio-stats $I7/audio_stats > $R/logs/e7.log 2>&1 || exit 1
bash $W/scripts/redesign/idea6/stage2/run_all.sh summary || exit 1
date '+%F %T done' >> $R/logs/final_run.time
