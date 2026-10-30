#!/bin/bash
# Auditor's recompute, one task at a time (CPU light: nice 10, 2 threads).
A=/Users/cajas.sebastian/Documents/qmoe-results/2026-10-22_1623_redesign-idea6-stage2-audit_code-c892c2f
PY=${PY:-python}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
cd $A/scripts
for t in "$@"; do
  nice -n 10 $PY -B recompute.py $t > $A/logs/recompute_$t.log 2>&1
  echo "$t exit $? $(date '+%T')" >> $A/logs/recompute.done
done
