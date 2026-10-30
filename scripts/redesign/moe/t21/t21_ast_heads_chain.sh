#!/bin/bash
# Once the AST extraction is done: AST training CSVs and heads (CPU), then scoring of the W8A8 and AST tiers.
: "${R:?}"; D=${D:-d1}
W=$HOME/Documents/qmoe-work/t21; T=$W/scripts/redesign/moe/t21
PY=${PY:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
until grep -q "extra round .* exit 0" $R/logs/gpu_ast.done 2>/dev/null; do sleep 30; done
nice -n 10 $PY $T/t21_make_csvs.py --r $R --d $D --tiers ast >> $R/logs/make_csvs.log 2>&1
TIERS=ast bash $T/t21_train_heads.sh pool official
echo "$(date '+%F %T') ast heads done" >> $R/logs/chain.done
nice -n 10 $PY $T/t21_score.py --r $R --d $D --tiers mn10_w8a8 ast_w4g128 >> $R/logs/score.log 2>&1; rc=$?
echo "$(date '+%F %T') score w8a8 ast exit $rc" >> $R/logs/chain.done
