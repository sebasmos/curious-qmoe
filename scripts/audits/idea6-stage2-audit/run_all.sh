#!/bin/bash
# Idea 6 Stage 2 audit, every step in order. CPU only, nice 10, at most 2 threads, inputs read only
# (python -B and PYTHONDONTWRITEBYTECODE=1 so nothing is written into the worktree).
A=/Users/cajas.sebastian/Documents/qmoe-results/2026-10-22_1623_redesign-idea6-stage2-audit_code-c892c2f
PY=${PY:-python}
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
cd $A/scripts
bash check2_timeline.sh > $A/logs/check2_timeline.log 2>&1          # check 2
nice -n 10 $PY -B costs_audit.py > $A/logs/costs_audit.log 2>&1      # check 4 recount (needed by the next steps)
bash run_recompute.sh esc5 esc50 urban                               # checks 1, 6, 7, 10 per task
nice -n 10 $PY -B combine.py > $A/logs/combine.log 2>&1              # Holm over tasks, statistics, one design for all
bash run_humility.sh esc5 esc50 urban                                # check 8 and E1
nice -n 10 $PY -B check_models.py > $A/logs/check_models.log 2>&1    # checks 3, 4, 5 on real clips
nice -n 10 $PY -B check4b_device.py > $A/logs/check4b_device.log 2>&1
nice -n 10 $PY -B check9_maps.py > $A/logs/check9_maps.log 2>&1      # check 9
# check 3 (head data, patch coverage): inline script, output in logs/check3_heads.log
