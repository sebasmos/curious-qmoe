#!/bin/bash
# T21 MN10 extraction on the CPU (nice 10, 2 threads, one job at a time).   R=... D=d1 bash t21_cpu_mn.sh
: "${R:?}"; D=${D:-d1}
PY=${PY:-python}
W=$HOME/Documents/qmoe-work/t21; X=$W/scripts/redesign/moe/t21/t21_extract_mn.py
AUD=${AUD:-$HOME/Documents/DATASETS/FSDKaggle2018/audio_raw}; AUDE=${AUDE:-$AUD}  # D2: AUD=.../FSD50K/audio_raw/dev AUDE=.../eval
U8C=$HOME/Documents/DATASETS/Urban8k/ast_768/urbansound8k.csv
U8A=$HOME/Documents/DATASETS/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
run() { local name=$1; shift; echo "$(date '+%F %T') start $name" >> $R/logs/cpu_mn.done
  nice -n 10 $PY $X "$@" >> $R/logs/extract_mn_$name.log 2>&1; echo "$(date '+%F %T') $name exit $?" >> $R/logs/cpu_mn.done; }
run pool_q --csv $R/csv/$D/pool.csv --audio-dir $AUD --bg-csv $U8C --bg-audio-dir $U8A --out-dir $R/emb/$D/mn_pool \
  --configs w8a8 --conditions clean bg:10 bg:0 pink:10 pink:0 white:10 white:0 --field --mel-cache $R/mel_cache/$D
run pool_f --csv $R/csv/$D/pool.csv --audio-dir $AUD --out-dir $R/emb/$D/mn_pool --configs fp32 --conditions clean
run extra --csv $R/csv/$D/extra.csv --audio-dir $AUDE --out-dir $R/emb/$D/mn_extra --configs fp32 w8a8 --conditions clean
echo "$(date '+%F %T') cpu mn done" >> $R/logs/cpu_mn.done
