#!/bin/bash
# Every audit step, in order (CPU only, nice 10, 2 threads). Usage: bash run_all.sh
set -e
A=/Users/cajas.sebastian/Documents/qmoe-results/2026-10-22_0925_redesign-idea5-stage3-audit_code-c2c070f
R=/Users/cajas.sebastian/Documents/qmoe-results/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38
PY=${PY:-python}
WT=${WT:-$(cd "$(dirname "$0")/../../.." && pwd)}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
cd $A
for ds in esc50 esc5 urban; do nice -n 10 $PY scripts/recompute.py --stage2 $R --dataset $ds --out $A/outputs/bitops > $A/outputs/recompute_$ds.log 2>&1; done
$PY scripts/compare_to_stage2.py $R $A/outputs > $A/outputs/check1_compare.txt
nice -n 10 $PY scripts/check2_splits.py $R $A/outputs/check2_splits.json > /dev/null
nice -n 10 $PY scripts/check3_nesting.py --stage2 $R --repo $WT --out $A/outputs --n-clips 16 > $A/outputs/check3_nesting.log 2>&1
nice -n 10 $PY scripts/check3b_ties.py --stage2 $R --repo $WT --out $A/outputs --n-clips 16 > $A/outputs/check3b_ties.log 2>&1
nice -n 10 $PY scripts/check4_costs.py --stage2 $R --out $A/outputs > /dev/null 2>&1
nice -n 10 $PY scripts/stress.py --stage2 $R --out $A/outputs > $A/outputs/stress.log 2>&1
nice -n 10 $PY scripts/check7_humility.py --stage2 $R --out $A/outputs > $A/outputs/check7_humility.log 2>&1
nice -n 10 $PY scripts/check8_misc.py $R $A/outputs/check8_misc.json > /dev/null
