#!/bin/bash
# T22 fold-1 phase (ANALYSIS_PLAN.md I.4): per task, the learning-rate probe (fold 1, seed 0, three rates), then seeds 1 and 2 at the
# chosen rate (the seed-0 run of the chosen rate is reused). Every GPU process goes through the shared FIFO queue and stays under about
# 80 minutes (the trainer defers a seed that would start after 40 minutes; the loop re-queues until every run exists).
#   R=RESULTS bash t22_p1_driver.sh esc50 urban tau
set -u
P=${PY:-python}; GPU=$HOME/Documents/qmoe-work/fair_gpu_run.sh
H="$(cd "$(dirname "$0")" && pwd)"; cd "$H"; RUNS=$R/runs; mkdir -p $RUNS $R/logs; LOG=$R/logs/driver_p1.log
waitmem() { while :; do fr=$(memory_pressure | tail -1 | grep -oE '[0-9]+%' | tr -d %); [ "${fr:-0}" -ge 25 ] && break; echo "$(date '+%T') memory free ${fr}%, waiting" >> $LOG; sleep 120; done; }
gpu() { waitmem; echo "$(date '+%F %T') queue: $1" >> $LOG; COOLDOWN=30 WAIT_MAX=100000 $GPU claude-t22 bash -c "export T22_TURN_START=\$(date +%s); $2"; rc=$?; echo "$(date '+%F %T') end rc=$rc: $1" >> $LOG; return $rc; }
diskok() { f=$(df -g $HOME | tail -1 | awk '{print $4}'); [ "$f" -ge 25 ] || { echo "$(date '+%T') free disk ${f} GB < 25, stop" >> $LOG; exit 5; }; }
for TASK in "$@"; do
  for n in 1 2 3 4; do
    diskok; todo=""
    for g in 1e-4 3e-4 1e-3; do [ -f $RUNS/${TASK}_f1_s0_lr$g.json ] || todo="$todo $g"; done
    [ -z "$todo" ] && break
    cmd=""; for g in $todo; do cmd="$cmd $P t22_train.py --task $TASK --fold 1 --seeds 0 --lr $g --tag lr$g --out $RUNS >> $R/logs/train_${TASK}_f1_probe.log 2>&1 &&"; done
    gpu "$TASK probe$todo" "${cmd% &&}" || exit 4
  done
  $P t22_pick_lr.py --runs $RUNS --task $TASK >> $LOG 2>&1 || exit 6
  LR=$($P -c "import json; print(json.load(open('$RUNS/lr_choice_$TASK.json'))['chosen'])")
  [ -f $RUNS/${TASK}_f1_s0.json ] || { cp $RUNS/${TASK}_f1_s0_lr$LR.pt $RUNS/${TASK}_f1_s0.pt; cp $RUNS/${TASK}_f1_s0_lr$LR.json $RUNS/${TASK}_f1_s0.json; }
  for n in 1 2 3 4; do
    diskok; [ -f $RUNS/${TASK}_f1_s1.json ] && [ -f $RUNS/${TASK}_f1_s2.json ] && break
    gpu "$TASK f1 s1 s2 lr$LR" "$P t22_train.py --task $TASK --fold 1 --seeds 1 2 --lr $LR --out $RUNS >> $R/logs/train_${TASK}_f1.log 2>&1" || exit 4
  done
  echo "$(date '+%F %T') FOLD1_DONE $TASK" >> $LOG
done
