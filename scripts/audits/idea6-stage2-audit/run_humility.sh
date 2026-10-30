#!/bin/bash
A=/Users/cajas.sebastian/Documents/qmoe-results/2026-10-22_1623_redesign-idea6-stage2-audit_code-c892c2f
PY=${PY:-python}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd $A/scripts
for t in "$@"; do
  nice -n 10 $PY -B humility.py $t > $A/logs/humility_$t.log 2>&1
  echo "$t exit $? $(date '+%T')" >> $A/logs/humility.done
done
