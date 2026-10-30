#!/bin/bash
# T25 CPU chain: CNN extraction (clean FP32 and W8A8, then every shift condition at W8A8), head CSVs, heads, scores.
# One CPU job at a time, nice 10, 2 threads. Before each stage: free disk above 25 GB and memory_pressure free at or
# above 25%, else wait (memory) or stop (disk).   R=... bash t25_cpu_chain.sh
: "${R:?}"; D=${D:-d1}
PY=${PY:-python}
W=$HOME/Documents/qmoe-work/t25; T=$W/scripts/redesign/moe/t25
R21=$HOME/Documents/qmoe-results/2026-10-26_1408_moe-t21_code-731570e
AUD=$HOME/Documents/DATASETS/FSDKaggle2018/audio_raw
U8C=$HOME/Documents/DATASETS/Urban8k/ast_768/urbansound8k.csv
U8A=$HOME/Documents/DATASETS/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
log() { echo "$(date '+%F %T') $*" >> $R/logs/chain.done; }
guard() {
  local free_gb=$(df -g $HOME | awk 'NR==2{print $4}')
  if [ "$free_gb" -lt 25 ]; then log "STOP disk free ${free_gb} GB below 25"; exit 3; fi
  while [ "$(memory_pressure | awk -F': ' '/free percentage/{print $2+0}')" -lt 25 ]; do log "waiting, memory free below 25%"; sleep 120; done
}
step() { local name=$1; shift; guard; log "start $name"; nice -n 10 "$@" >> $R/logs/$name.log 2>&1; local rc=$?; log "$name exit $rc"; [ $rc -ne 0 ] && exit $rc; }
step extract_clean $PY $T/t25_extract_cnn.py --csv $R/csv/$D/pool.csv --audio-dir $AUD --out-dir $R/emb/$D/cnn_pool \
  --models mn04 mn05 mn20 dymn04 --configs fp32 w8a8 --conditions clean
step extract_shift $PY $T/t25_extract_cnn.py --csv $R/csv/$D/pool.csv --audio-dir $AUD --bg-csv $U8C --bg-audio-dir $U8A \
  --out-dir $R/emb/$D/cnn_pool --models mn04 mn05 mn20 dymn04 --configs w8a8 \
  --conditions bg:10 bg:0 pink:10 pink:0 white:10 white:0 --field
step check_emb $PY $T/t25_check_emb.py $R $R21 $D
step make_csvs $PY $T/t25_make_csvs.py --r $R --d $D
guard; log "start heads"; R=$R D=$D bash $T/t25_train_heads.sh; log "heads done"
step score $PY $T/t25_score.py --r $R --d $D
log "chain done"
