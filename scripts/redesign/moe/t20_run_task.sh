#!/bin/bash
# T20 driver for one task (ANALYSIS_PLAN_T20.md 3.3, 3.4): learning-rate probe on fold 1 seed 0, then per fold one GPU process for training
# (all seeds) and one for extraction, each through the shared MPS lock, swap checked before each process.
#   R=RESULTS bash t20_run_task.sh urban [first_fold]
set -u
TASK=$1; FIRST=${2:-1}; P=${PY:-python}; GPU=$HOME/Documents/qmoe-work/t20_gpu_run.sh
H="$(cd "$(dirname "$0")" && pwd)"; cd "$H"; RUNS=$R/runs/$TASK; SC=$R/scores/$TASK; mkdir -p $RUNS $SC $R/logs
NF=$($P -c "import sys; sys.path.insert(0,'.'); from t20_common import TASKS; print(TASKS['$TASK']['nf'])")
waitswap() { while :; do u=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); [ "${u%.*}" -lt 9500 ] && break; fr=$(memory_pressure | tail -1 | grep -oE '[0-9]+%' | tr -d %); [ "${fr:-0}" -ge 25 ] && break; echo "$(date '+%T') swap ${u}M, waiting" >> $R/logs/driver_$TASK.log; sleep 120; done; }
gpu() { waitswap; echo "$(date '+%F %T') start $*" >> $R/logs/driver_$TASK.log; COOLDOWN=${COOLDOWN:-60} WAIT_MAX=100000 $GPU "$@"; rc=$?; echo "$(date '+%F %T') end rc=$rc $1 $2 $3" >> $R/logs/driver_$TASK.log; return $rc; }
if [ ! -f $RUNS/lr_choice.json ]; then
  for lr in 1e-4 3e-5; do gpu $P t20_train.py --task $TASK --fold 1 --seeds 0 --lr_body $lr --tag lr$lr --out $RUNS >> $R/logs/train_${TASK}_probe.log 2>&1; done
  $P - <<PY || exit 3
import json, shutil
from pathlib import Path
d = Path("$RUNS"); r = {lr: json.load(open(d / f"f1_s0_lr{lr}.json"))["best_iv_f1"] for lr in ("1e-4", "3e-5")}
ch = "1e-4" if r["1e-4"] >= r["3e-5"] else "3e-5"
for ext in (".json", ".log", "_tail.pt", "_iv.npz"): shutil.copy(d / f"f1_s0_lr{ch}{ext}", d / f"f1_s0{ext}")
json.dump(dict(rule="best-checkpoint clean inner-validation macro F1, fold 1 seed 0, ties to 1e-4", iv_f1=r, chosen=ch), open(d / "lr_choice.json", "w"), indent=1); print("lr", r, ch)
PY
fi
LR=$($P -c "import json; print(json.load(open('$RUNS/lr_choice.json'))['chosen'])")
[ "${PROBE_ONLY:-0}" = 1 ] && exit 0
for f in $(seq $FIRST $NF); do
  gpu $P t20_train.py --task $TASK --fold $f --seeds 0 1 2 --lr_body $LR --out $RUNS >> $R/logs/train_${TASK}_f$f.log 2>&1 || { echo "train f$f failed" >> $R/logs/driver_$TASK.log; exit 4; }
  [ -f $SC/f${f}_meta.json ] || gpu $P t20_extract.py --task $TASK --fold $f --runs $RUNS --out $SC >> $R/logs/extract_${TASK}_f$f.log 2>&1 || { echo "extract f$f failed" >> $R/logs/driver_$TASK.log; exit 5; }
done
echo "$(date '+%F %T') TASK_DONE $TASK" >> $R/logs/driver_$TASK.log
