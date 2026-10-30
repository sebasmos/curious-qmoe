#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c orchestration, scoring: P5 scores once the exact-length extraction has ended; the new CNN heads'
# scores as their runs complete (score_all.sh waits per run); then way (b) retraining for the P2 picks' head kinds.
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
S3B=${S3B:?set S3B, see REPRODUCE_PAPER.md}
export R S3B PY=${PY:-python}
cd $W
until grep -q "urban_unp2 exit 0" $R/logs/extract.done; do sleep 60; done
bash scripts/redesign/idea6/stage3c/score_all.sh unp
bash scripts/redesign/idea6/stage3c/score_all.sh cnn
echo "$(date '+%F %T') all cnn and unp scores done" >> $R/logs/score.done
