#!/bin/bash
# T21 D2 end to end once the audio is ready. GPU (AST, through the shared wrapper) runs beside one CPU job at a time:
# MN10 extraction, MN10 heads, W8A8 scores, then AST heads and scores, the primary analysis, then the int8 path and the
# analysis again with it.   R=... bash t21_d2_chain.sh
: "${R:?}"; export D=d2 SEEDS="0 1 2"
W=$HOME/Documents/qmoe-work/t21; T=$W/scripts/redesign/moe/t21
PY=${PY:-python}; PYO=${PYO:-python}
export AUD=$HOME/Documents/DATASETS/FSD50K/audio_raw/dev AUDE=$HOME/Documents/DATASETS/FSD50K/audio_raw/eval
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
L=$R/logs/d2_chain.done; log() { echo "$(date '+%F %T') $*" >> $L; }
until grep -q "d2 audio ready" $R/logs/download_d2.log 2>/dev/null; do grep -qE "stop|failed" $R/logs/download_d2.log && { log "download stopped"; exit 1; }; sleep 60; done
n=$(ls $AUD | wc -l); ne=$(ls $AUDE | wc -l); log "audio $n dev, $ne eval"
[ $n -eq 26272 ] && [ $ne -eq 4396 ] || { log "audio count wrong"; exit 1; }
nohup bash $T/t21_gpu_ast.sh pool_q pool_f extra > $R/logs/gpu_ast_d2.out 2>&1 &
bash $T/t21_cpu_mn.sh; log "mn extraction: $(tail -1 $R/logs/cpu_mn.done)"
nice -n 10 $PY $T/t21_make_csvs.py --r $R --d $D --tiers mn10 >> $R/logs/make_csvs_d2.log 2>&1 || { log "make_csvs mn10 failed"; exit 1; }
TIERS=mn10 bash $T/t21_train_heads.sh pool official; log "mn10 heads done"
nice -n 10 $PY $T/t21_score.py --r $R --d $D --seeds 0 1 2 --tiers mn10_w8a8 >> $R/logs/score_d2.log 2>&1; log "score w8a8 exit $?"
until grep -q "jobs pool_q pool_f extra finished" $R/logs/gpu_ast.done 2>/dev/null && [ -f $R/emb/d2/ast_extra/fp32__clean.npy ]; do sleep 60; done
nice -n 10 $PY $T/t21_make_csvs.py --r $R --d $D --tiers ast >> $R/logs/make_csvs_d2.log 2>&1 || { log "make_csvs ast failed"; exit 1; }
TIERS=ast bash $T/t21_train_heads.sh pool official; log "ast heads done"
nice -n 10 $PY $T/t21_score.py --r $R --d $D --seeds 0 1 2 --tiers ast_w4g128 >> $R/logs/score_d2.log 2>&1; log "score ast exit $?"
nice -n 10 $PY $T/t21_analyze.py --r $R --d $D > $R/logs/analyze_d2.log 2>&1; log "analyze d2 (no int8) exit $?"
for step in export build embed; do
  nice -n 10 $PYO $T/t21_int8.py $step --r $R --d $D >> $R/logs/int8_d2_$step.log 2>&1; rc=$?; log "int8 $step exit $rc"; [ $rc -ne 0 ] && exit $rc
done
nice -n 10 $PY $T/t21_score.py --r $R --d $D --designs pool --seeds 0 1 2 --tiers mn10_int8 >> $R/logs/score_d2.log 2>&1; log "score int8 exit $?"
nice -n 10 $PY $T/t21_analyze.py --r $R --d $D > $R/logs/analyze_d2_int8.log 2>&1; log "analyze d2 with int8 exit $?"
nice -n 10 $PY $T/t21_analyze.py --r $R --combine d1 d2 > $R/logs/combine.log 2>&1; log "combine exit $?"
log "d2 chain done"
