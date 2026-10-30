#!/bin/bash
# T20, one GPU process (run inside the GPU wrapper): learning-rate probe (fold 1, seed 0, both rates), the choice by the plan's rule,
# then fold 1 seeds 1 and 2 at the chosen rate and the fold 1 extraction. Same steps as t20_run_task.sh, fewer lock turns.
#   R=RESULTS bash t20_first_turn.sh TASK
set -eu
TASK=$1; P=${PY:-python}; cd "$(dirname "$0")"; RUNS=$R/runs/$TASK; SC=$R/scores/$TASK; mkdir -p $RUNS $SC
for lr in 1e-4 3e-5; do $P t20_train.py --task $TASK --fold 1 --seeds 0 --lr_body $lr --tag lr$lr --out $RUNS >> $R/logs/train_${TASK}_probe.log 2>&1; done
$P - <<PY
import json, shutil
from pathlib import Path
d = Path("$RUNS"); r = {lr: json.load(open(d / f"f1_s0_lr{lr}.json"))["best_iv_f1"] for lr in ("1e-4", "3e-5")}
ch = "1e-4" if r["1e-4"] >= r["3e-5"] else "3e-5"
for ext in (".json", ".log", "_tail.pt", "_iv.npz"): shutil.copy(d / f"f1_s0_lr{ch}{ext}", d / f"f1_s0{ext}")
json.dump(dict(rule="best-checkpoint clean inner-validation macro F1, fold 1 seed 0, ties to 1e-4", iv_f1=r, chosen=ch), open(d / "lr_choice.json", "w"), indent=1); print("lr", r, ch)
PY
LR=$($P -c "import json; print(json.load(open('$RUNS/lr_choice.json'))['chosen'])")
$P t20_train.py --task $TASK --fold 1 --seeds 0 1 2 --lr_body $LR --out $RUNS >> $R/logs/train_${TASK}_f1.log 2>&1
[ -f $SC/f1_meta.json ] || $P t20_extract.py --task $TASK --fold 1 --runs $RUNS --out $SC >> $R/logs/extract_${TASK}_f1.log 2>&1
