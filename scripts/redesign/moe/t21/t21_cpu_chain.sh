#!/bin/bash
# After the MN10 heads: int8 export, build, embed (qmoe-ort env), then (once the AST extraction is done) AST CSVs and heads.
: "${R:?}"; D=${D:-d1}
W=$HOME/Documents/qmoe-work/t21; T=$W/scripts/redesign/moe/t21
PY=${PY:-python}; PYO=${PYO:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
until grep -q "heads official done" $R/logs/train_heads.done 2>/dev/null; do sleep 30; done
for step in export build embed; do
  nice -n 10 $PYO $T/t21_int8.py $step --r $R --d $D >> $R/logs/int8_$step.log 2>&1; rc=$?; echo "$(date '+%F %T') int8 $step exit $rc" >> $R/logs/chain.done; [ $rc -ne 0 ] && exit $rc
done
until grep -q "extra round .* exit 0" $R/logs/gpu_ast.done 2>/dev/null; do sleep 30; done
nice -n 10 $PY $T/t21_make_csvs.py --r $R --d $D --tiers ast >> $R/logs/make_csvs.log 2>&1
TIERS=ast bash $T/t21_train_heads.sh pool official
echo "$(date '+%F %T') ast heads done" >> $R/logs/chain.done
