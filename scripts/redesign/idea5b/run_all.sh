#!/bin/bash
# Idea 5b: every step, CPU only, nice 10, 2 threads. Usage: bash run_all.sh OUT_DIR
set -e
OUT=$1
R=/Users/cajas.sebastian/Documents/qmoe-results/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p $OUT/outputs $OUT/logs
cd $HERE
for ds in esc50 esc5 urban; do
  nice -n 10 $PY cascades.py --stage2 $R --dataset $ds --out $OUT/outputs > $OUT/logs/cascades_$ds.log 2>&1 &
done
wait
for ds in esc50 esc5 urban; do
  nice -n 10 $PY -W ignore hull.py --dataset $ds --out $OUT/outputs > $OUT/logs/hull_$ds.log 2>&1 &
done
wait
nice -n 10 $PY verdict.py --out $OUT/outputs > $OUT/logs/verdict.log 2>&1
nice -n 10 $PY humility.py --stage2 $R --out $OUT/outputs > $OUT/logs/humility.log 2>&1
nice -n 10 $PY humility.py --stage2 $R --out $OUT/outputs --key theory_tuple --tag theory > $OUT/logs/humility_theory.log 2>&1
