#!/bin/bash
# T22 all folds and seeds (ANALYSIS_PLAN.md I.5), only after P1 and P2 pass: every fold 2..nf, seeds 0 to 2 at the task's chosen rate.
# Each GPU turn chains the missing runs; the trainer defers a seed that would start after START_BY minutes of the turn, and the loop
# re-queues until every run exists.     R=RESULTS bash t22_all_driver.sh esc50 urban tau
set -u
P=${PY:-python}; GPU=$HOME/Documents/qmoe-work/fair_gpu_run.sh
H="$(cd "$(dirname "$0")" && pwd)"; cd "$H"; RUNS=$R/runs; LOG=$R/logs/driver_all.log; START_BY=${START_BY:-40}
grep -q '"passes": true' $R/tables/p1.json || { echo "P1 has not passed: refusing" >> $LOG; exit 7; }
waitmem() { while :; do fr=$(memory_pressure | tail -1 | grep -oE '[0-9]+%' | tr -d %); [ "${fr:-0}" -ge 25 ] && break; echo "$(date '+%T') memory free ${fr}%, waiting" >> $LOG; sleep 120; done; }
diskok() { f=$(df -g $HOME | tail -1 | awk '{print $4}'); [ "$f" -ge 25 ] || { echo "$(date '+%T') free disk ${f} GB < 25, stop" >> $LOG; exit 5; }; }
for TASK in "$@"; do
  NF=$($P -c "import sys; sys.path.insert(0,'.'); from t22_common import TASKS; print(TASKS['$TASK']['nf'])")
  LR=$($P -c "import json; print(json.load(open('$RUNS/lr_choice_$TASK.json'))['chosen'])")
  for n in $(seq 1 30); do
    diskok; cmd=""
    for f in $(seq 2 $NF); do
      miss=""; for s in 0 1 2; do [ -f $RUNS/${TASK}_f${f}_s$s.json ] || miss="$miss $s"; done
      [ -n "$miss" ] && cmd="$cmd $P t22_train.py --task $TASK --fold $f --seeds$miss --lr $LR --start_by_min $START_BY --out $RUNS >> $R/logs/train_${TASK}_f$f.log 2>&1 &&"
    done
    [ -z "$cmd" ] && break
    waitmem; echo "$(date '+%F %T') queue: $TASK" >> $LOG
    COOLDOWN=30 WAIT_MAX=100000 $GPU claude-t22 bash -c "export T22_TURN_START=\$(date +%s); ${cmd% &&}"; rc=$?
    echo "$(date '+%F %T') end rc=$rc: $TASK" >> $LOG; [ $rc -ne 0 ] && exit 4
  done
  echo "$(date '+%F %T') ALL_DONE $TASK" >> $LOG
done
