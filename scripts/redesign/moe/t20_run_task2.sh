#!/bin/bash
# T20 driver, version 2 (2026-10-26 19:45): the same steps as t20_run_task.sh, but one GPU process trains and extracts TWO folds (about 55 min)
# to cut the number of lock turns when three agents share the GPU. The learning-rate choice must exist (t20_run_task.sh made it).
#   R=RESULTS bash t20_run_task2.sh TASK FIRST_FOLD
set -u
TASK=$1; FIRST=${2:-1}; P=${PY:-python}; GPU=$HOME/Documents/qmoe-work/t20_gpu_run.sh
H="$(cd "$(dirname "$0")" && pwd)"; cd "$H"; RUNS=$R/runs/$TASK; SC=$R/scores/$TASK; mkdir -p $RUNS $SC $R/logs
NF=$($P -c "import sys; sys.path.insert(0,'.'); from t20_common import TASKS; print(TASKS['$TASK']['nf'])")
waitswap() { while :; do u=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); [ "${u%.*}" -lt 9500 ] && break; fr=$(memory_pressure | tail -1 | grep -oE '[0-9]+%' | tr -d %); [ "${fr:-0}" -ge 25 ] && break; echo "$(date '+%T') swap ${u}M, waiting" >> $R/logs/driver_$TASK.log; sleep 120; done; }
LR=$($P -c "import json; print(json.load(open('$RUNS/lr_choice.json'))['chosen'])") || exit 3
step() { echo "$P t20_train.py --task $TASK --fold $1 --seeds 0 1 2 --lr_body $LR --out $RUNS >> $R/logs/train_${TASK}_f$1.log 2>&1 && { [ -f $SC/f$1_meta.json ] || $P t20_extract.py --task $TASK --fold $1 --runs $RUNS --out $SC >> $R/logs/extract_${TASK}_f$1.log 2>&1; }"; }
f=$FIRST
while [ $f -le $NF ]; do
  g=$((f + 1)); cmd="$(step $f)"; [ $g -le $NF ] && cmd="$cmd && $(step $g)"
  waitswap; echo "$(date '+%F %T') start folds $f $([ $g -le $NF ] && echo $g)" >> $R/logs/driver_$TASK.log
  COOLDOWN=${COOLDOWN:-60} WAIT_MAX=100000 $GPU bash -c "$cmd"; rc=$?
  echo "$(date '+%F %T') end rc=$rc folds $f $([ $g -le $NF ] && echo $g)" >> $R/logs/driver_$TASK.log
  [ $rc -ne 0 ] && { echo "folds $f failed" >> $R/logs/driver_$TASK.log; exit 4; }
  f=$((g + 1))
done
echo "$(date '+%F %T') TASK_DONE $TASK" >> $R/logs/driver_$TASK.log
