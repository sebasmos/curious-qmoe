#!/usr/bin/env bash
# Idea 6 stage 1 audit: every step, CPU only, nice 10, 2 threads. Inputs are read only.
set -euo pipefail
A=/Users/cajas.sebastian/Documents/qmoe-results/2026-10-22_1101_redesign-idea6-stage1-audit_code-688a498
PY=${PY:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
cd $A
R() { nice -n 10 $PY -I "$@"; }
R scripts/check2_alignment.py outputs/check2_alignment.json   > logs/check2_alignment.log 2>&1
R scripts/check2b_ast_rows.py outputs/check2b_ast_rows.json   > logs/check2b_ast_rows.log 2>&1
R scripts/check9_linear.py outputs/check9_linear.json         > logs/check9_linear.log 2>&1   # fresh process required
R scripts/check5_costs.py outputs/check5_costs.json           > logs/check5_costs.log 2>&1
R scripts/check5_frontend_time.py outputs/check5_frontend_time.json > logs/check5_frontend_time.log 2>&1
R scripts/audio_stats.py outputs/audio_stats.csv              > logs/audio_stats.log 2>&1
for t in esc5 urban esc50; do R scripts/recompute.py --task $t --out outputs/recompute > logs/recompute_$t.log 2>&1; done
R scripts/verdict.py outputs                                  > logs/verdict.log 2>&1
R scripts/extras.py outputs                                   > logs/extras.log 2>&1
R scripts/latency.py outputs                                  > logs/latency.log 2>&1
R scripts/check8_humility.py outputs                          > logs/check8_humility.log 2>&1
