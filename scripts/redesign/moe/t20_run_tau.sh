#!/bin/bash
# T20 TAU driver: turn 1 = learning-rate probe (fold 1 seed 0, both rates) and the choice; then per fold one GPU process for training
# (all seeds, about 50 min at 1024 frames) and one for extraction. Same steps and rules as t20_run_task.sh.
#   R=RESULTS bash t20_run_tau.sh [first_fold]
set -u
TASK=tau; FIRST=${1:-1}; P=${PY:-python}; GPU=$HOME/Documents/qmoe-work/t20_gpu_run.sh
H="$(cd "$(dirname "$0")" && pwd)"; cd "$H"; RUNS=$R/runs/$TASK; SC=$R/scores/$TASK; mkdir -p $RUNS $SC $R/logs
waitswap() { while :; do u=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); [ "${u%.*}" -lt 9500 ] && break; fr=$(memory_pressure | tail -1 | grep -oE '[0-9]+%' | tr -d %); [ "${fr:-0}" -ge 25 ] && break; echo "$(date '+%T') swap ${u}M, waiting" >> $R/logs/driver_$TASK.log; sleep 120; done; }
gpu() { waitswap; echo "$(date '+%F %T') start $2 $3 $4 $5 $6 $7" >> $R/logs/driver_$TASK.log; COOLDOWN=20 WAIT_MAX=100000 $GPU "$@"; rc=$?; echo "$(date '+%F %T') end rc=$rc" >> $R/logs/driver_$TASK.log; return $rc; }
if [ ! -f $RUNS/lr_choice.json ]; then
  gpu bash -c "for lr in 1e-4 3e-5; do $P t20_train.py --task tau --fold 1 --seeds 0 --lr_body \$lr --tag lr\$lr --out $RUNS >> $R/logs/train_tau_probe.log 2>&1 || exit 1; done" || { echo "probe failed" >> $R/logs/driver_$TASK.log; exit 3; }
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
for f in $(seq $FIRST 5); do
  gpu $P t20_train.py --task tau --fold $f --seeds 0 1 2 --lr_body $LR --out $RUNS >> $R/logs/train_tau_f$f.log 2>&1 || { echo "train f$f failed" >> $R/logs/driver_$TASK.log; exit 4; }
  [ -f $SC/f${f}_meta.json ] || gpu $P t20_extract.py --task tau --fold $f --runs $RUNS --out $SC >> $R/logs/extract_tau_f$f.log 2>&1 || { echo "extract f$f failed" >> $R/logs/driver_$TASK.log; exit 5; }
done
echo "$(date '+%F %T') TASK_DONE tau" >> $R/logs/driver_$TASK.log
