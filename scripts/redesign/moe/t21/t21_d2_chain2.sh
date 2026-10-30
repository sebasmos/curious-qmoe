#!/bin/bash
# T21 D2, second form of the chain (2026-10-27 05:40, for disk): the int8 path runs as soon as the MN10 heads exist and
# deletes each condition's mel cache once embedded; the AST part runs when the GPU extraction is done.
#   R=... bash t21_d2_chain2.sh mn|ast
: "${R:?}"; export D=d2 SEEDS="0 1 2"
W=$HOME/Documents/qmoe-work/t21; T=$W/scripts/redesign/moe/t21
PY=${PY:-python}; PYO=${PYO:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
L=$R/logs/d2_chain.done; log() { echo "$(date '+%F %T') $*" >> $L; }
case $1 in
mn)
  until [ "$(grep -c 'cpu mn done' $R/logs/cpu_mn.done)" -ge 2 ]; do sleep 60; done
  log "mn extraction: $(grep -A0 'extra exit' $R/logs/cpu_mn.done | tail -1)"
  nice -n 10 $PY $T/t21_make_csvs.py --r $R --d $D --tiers mn10 >> $R/logs/make_csvs_d2.log 2>&1 || { log "make_csvs mn10 failed"; exit 1; }
  TIERS=mn10 bash $T/t21_train_heads.sh pool official; log "mn10 heads done"
  nice -n 10 $PY $T/t21_score.py --r $R --d $D --seeds 0 1 2 --tiers mn10_w8a8 >> $R/logs/score_d2.log 2>&1; log "score w8a8 exit $?"
  for step in export build "embed --delete-cache"; do
    nice -n 10 $PYO $T/t21_int8.py $step --r $R --d $D >> $R/logs/int8_d2_${step%% *}.log 2>&1; rc=$?; log "int8 $step exit $rc"; [ $rc -ne 0 ] && exit $rc
  done
  nice -n 10 $PY $T/t21_score.py --r $R --d $D --designs pool --seeds 0 1 2 --tiers mn10_int8 >> $R/logs/score_d2.log 2>&1; log "score int8 exit $?"
  until grep -q "analyze d2 (no int8) exit" $L; do sleep 60; done
  nice -n 10 $PY $T/t21_analyze.py --r $R --d $D > $R/logs/analyze_d2_int8.log 2>&1; log "analyze d2 with int8 exit $?"
  nice -n 10 $PY $T/t21_analyze.py --r $R --combine d1 d2 > $R/logs/combine.log 2>&1; log "combine exit $?" ;;
ast)
  until grep -q "jobs pool_q pool_f extra finished" $R/logs/gpu_ast.done && grep -q "score w8a8 exit 0" $L; do sleep 60; done
  nice -n 10 $PY $T/t21_make_csvs.py --r $R --d $D --tiers ast >> $R/logs/make_csvs_d2.log 2>&1 || { log "make_csvs ast failed"; exit 1; }
  TIERS=ast bash $T/t21_train_heads.sh pool official; log "ast heads done"
  nice -n 10 $PY $T/t21_score.py --r $R --d $D --seeds 0 1 2 --tiers ast_w4g128 >> $R/logs/score_d2.log 2>&1; log "score ast exit $?"
  nice -n 10 $PY $T/t21_analyze.py --r $R --d $D > $R/logs/analyze_d2.log 2>&1; log "analyze d2 (no int8) exit $?"
  nice -n 10 $PY $T/t21_analyze.py --r $R --combine d1 d2 > $R/logs/combine_primary.log 2>&1; log "combine (primary) exit $?" ;;
esac
