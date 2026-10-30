#!/bin/bash
# Timing runs, one process per runtime and thread count, in sequence.
# Usage: bash scripts/run_timing.sh [threads ...]   (default 2 1 4); TAG=... appends to the output names;
# WARM and REPS override the pass counts, MAXLOAD the load gate (smoke test: WARM=1 REPS=3 TAG=_smoke MAXLOAD=100).
D=${R:-/Users/cajas.sebastian/Documents/qmoe-results/2026-10-23_1919_redesign-idea6-latency-full_code-cbf6be6}   # this run folder
PY=${PY:-python}
PYO=${PYO:-python}
cd $D
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
THREADS=${@:-2 1 4}
WARM=${WARM:-50}; REPS=${REPS:-500}; MAXLOAD=${MAXLOAD:-9}
for t in $THREADS; do
  echo "$(date '+%F %T') t$t torch start, uptime: $(uptime)" >> logs/timing.done
  OMP_NUM_THREADS=$t VECLIB_MAXIMUM_THREADS=$t MKL_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t \
    $PY scripts/time_torch.py --threads $t --warm $WARM --reps $REPS --max-load $MAXLOAD --tag "$TAG" > logs/time_torch_t$t$TAG.log 2>&1
  echo "$(date '+%F %T') t$t torch exit $?, uptime: $(uptime)" >> logs/timing.done
  OMP_NUM_THREADS=$t VECLIB_MAXIMUM_THREADS=$t \
    $PYO scripts/time_ort.py --threads $t --warm $WARM --reps $REPS --max-load $MAXLOAD --tag "$TAG" > logs/time_ort_t$t$TAG.log 2>&1
  echo "$(date '+%F %T') t$t ort exit $?, uptime: $(uptime)" >> logs/timing.done
done
echo "$(date '+%F %T') all done" >> logs/timing.done
