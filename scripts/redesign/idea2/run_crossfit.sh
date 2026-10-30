#!/bin/bash
# Run the cross-fit job list two at a time (shared machine), skipping jobs
# whose output exists. A job line is "<ds> <model> <held-out folds...>"; its
# recipe is <results>/configs/<ds>.yaml (the config.yaml of the
# <ds>_singles_s0 run). "wait <pid>" holds one slot until that process ends.
#   bash run_crossfit.sh <jobs.txt> <results dir>
set -u
JOBS=$1 RES=$2
CODE=/Users/cajas.sebastian/Desktop/repositories/curious-qmoe-run-5db7e75
HERE=$(cd "$(dirname "$0")" && pwd)
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate curious-qmoe
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CODE RES HERE
one() {
  if [ "$1" = wait ]; then while kill -0 "$2" 2>/dev/null; do sleep 10; done; return 0; fi
  ds=$1 m=$2; shift 2
  tag=$(echo "$@" | tr ' ' '_')
  [ -f "$RES/crossfit/$ds/$m/ho_${tag}.json" ] && return 0
  python "$HERE/crossfit_train.py" --code "$CODE" --config "$RES/configs/$ds.yaml" --model "$m" --holdout "$@" \
    --out "$RES/crossfit/$ds" > "$RES/logs/cf_${ds}_${m}_${tag}.log" 2>&1
  echo "$? $ds $m $tag $(date +%T)" >> "$RES/logs/crossfit_runs.txt"
}
export -f one
xargs -P 2 -L 1 bash -c 'one "$@"' _ < "$JOBS"
