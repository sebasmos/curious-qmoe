#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c: way (b) scoring per (model, task, kind) once every per-fold retrained head exists.
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
S3B=${S3B:?set S3B, see REPRODUCE_PAPER.md}
PY=${PY:-python}
cd $W; export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
while read -r m t k; do
  case $t in urban) seeds="0 1 2"; folds="1 2 3 4 5 6 7 8 9 10" ;; *) seeds="0 1 2 3 4"; folds="1 2 3 4 5" ;; esac
  for s in $seeds; do for f in $folds; do until [ -f $R/outputs_int8b/${m}r_${t}_s${s}_f${f}/$k/summary.json ]; do sleep 30; done; done; done
  nice -n 10 $PY scripts/redesign/idea6/stage3c/score_int8.py --mode b --model $m --task $t --kinds $k --s3b $S3B --r $R > $R/logs/score_int8b_${m}_${t}.log 2>&1
  echo "$(date '+%F %T') score int8b $m $t $k exit $?" >> $R/logs/score.done
done < $R/logs/int8b_jobs.txt
