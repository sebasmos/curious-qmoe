#!/bin/bash
# T26 Part C CPU chain for one dataset and one set of CNNs (ANALYSIS_PLAN.md C7): extraction on all.csv (clean, FP32 and
# W8A8) and eval.csv (W8A8, every shift cell and the field cells), embedding checks, head CSVs, protocol 3 heads, scores;
# with TAG=pre, the precondition. nice 10, 2 threads. Before each stage: free disk above 25 GB (else stop) and
# memory_pressure free at or above 25% (else wait).
#   R=... D=d3 SEEDS="0 1 2" MODELS="mn20" TAG=pre bash t26_cpu_chain.sh
#   R=... D=d3 SEEDS="0 1 2" MODELS="mn04 mn05 dymn04" TAG=wide bash t26_cpu_chain.sh
: "${R:?}"; D=${D:?}; SEEDS=${SEEDS:?}; MODELS=${MODELS:?}; TAG=${TAG:?}
PY=${PY:-python}
W=$HOME/Documents/qmoe-work/t26; T=$W/scripts/redesign/moe/t26
R24=$HOME/Documents/qmoe-results/2026-10-28_1827_moe-t24_code-9a51cf5
case $D in d3) AUD=$HOME/Documents/DATASETS/FSDnoisy18k/audio16k_t24 ;; d4) AUD=$HOME/Documents/DATASETS/TUT2017/audio16k_t24 ;; esac
U8C=$HOME/Documents/DATASETS/Urban8k/ast_768/urbansound8k.csv
U8A=$HOME/Documents/DATASETS/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
log() { echo "$(date '+%F %T') $*" >> $R/logs/chain_${D}_$TAG.done; }
guard() {
  local free_gb=$(df -g $HOME | awk 'NR==2{print $4}')
  if [ "$free_gb" -lt 25 ]; then log "STOP disk free ${free_gb} GB below 25"; exit 3; fi
  while [ "$(memory_pressure | awk -F': ' '/free percentage/{print $2+0}')" -lt 25 ]; do log "waiting, memory free below 25%"; sleep 120; done
}
step() { local name=$1; shift; guard; log "start $name"; nice -n 10 "$@" >> $R/logs/$name.log 2>&1; local rc=$?; log "$name exit $rc"; [ $rc -ne 0 ] && exit $rc; }
step extract_all_${D}_$TAG $PY $T/t26_extract.py --csv $R/csv/$D/all.csv --audio-dir $AUD --out-dir $R/emb/$D/cnn_all \
  --models $MODELS --configs fp32 w8a8 --conditions clean
step extract_eval_${D}_$TAG $PY $T/t26_extract.py --csv $R/csv/$D/eval.csv --audio-dir $AUD --bg-csv $U8C --bg-audio-dir $U8A \
  --out-dir $R/emb/$D/cnn_eval --models $MODELS --configs w8a8 --conditions bg:10 bg:0 pink:10 pink:0 white:10 white:0 --field
step check_emb_${D}_$TAG $PY $T/t26_check_emb.py $R $R24 $D
step make_csvs_${D}_$TAG $PY $T/t26_make_csvs.py --r $R --d $D --models $MODELS
guard; log "start heads"; R=$R D=$D SEEDS="$SEEDS" TIERS="$MODELS" bash $T/t26_train_heads.sh; log "heads done"
step score_${D}_$TAG $PY $T/t26_score.py --r $R --d $D --seeds $SEEDS --models $MODELS
if [ "$TAG" = pre ]; then
  step precondition_$D $PY $T/t26_part_c.py --r $R --d $D --seeds $SEEDS --precondition
  step recompute_precondition_$D $PY $T/t26_recompute.py --r $R --part c --d $D --seeds $SEEDS --precondition-only
fi
log "chain done"
