#!/bin/bash
# step 19.5 extraction: clean, five configurations, every clip except fold 1's test clips; after the calibrations finish
source /Users/cajas.sebastian/Documents/qmoe-results/2026-10-26_1409_moe-t19_code-ae334eb/scripts/env.sh
while pgrep -f "t19_calib.py" >/dev/null; do sleep 20; done
cd $W
for task in esc50 urban; do
  for try in 1 2 3 4; do
    $R/scripts/swapwait.sh
    $GPU $PY scripts/redesign/moe/t19_extract.py --task $task --configs w8a8 n4a4 n4a6 w4a6 w4a4 --conditions clean --exclude-test-fold 1 --max-minutes 80
    rc=$?; echo "$(date '+%F %T') p1 extract $task try $try exit $rc"
    [ $rc = 0 ] && break
  done
done
