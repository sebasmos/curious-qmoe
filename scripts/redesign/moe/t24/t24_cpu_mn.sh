#!/bin/bash
# T24 MN10 extraction on the CPU (nice 10, 2 threads, one job at a time), T21's extractor through t24_extract.py.
#   R=... D=d3 AUD=<audio16k_t24 dir> bash t24_cpu_mn.sh [all_q all_f eval_q]
: "${R:?}"; D=${D:?}; AUD=${AUD:?}
PY=${PY:-python}
W=$HOME/Documents/qmoe-work/t24; X=$W/scripts/redesign/moe/t24/t24_extract.py
U8C=$HOME/Documents/DATASETS/Urban8k/ast_768/urbansound8k.csv
U8A=$HOME/Documents/DATASETS/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
run() { local name=$1; shift; echo "$(date '+%F %T') start $name" >> $R/logs/cpu_mn_$D.done
  nice -n 10 $PY $X mn "$@" >> $R/logs/extract_mn_${D}_$name.log 2>&1; echo "$(date '+%F %T') $name exit $?" >> $R/logs/cpu_mn_$D.done; }
for job in ${@:-all_q all_f eval_q}; do case $job in
  all_q) run all_q --csv $R/csv/$D/all.csv --audio-dir $AUD --out-dir $R/emb/$D/mn_all --configs w8a8 --conditions clean ;;
  all_f) run all_f --csv $R/csv/$D/all.csv --audio-dir $AUD --out-dir $R/emb/$D/mn_all --configs fp32 --conditions clean ;;
  eval_q) run eval_q --csv $R/csv/$D/eval.csv --audio-dir $AUD --bg-csv $U8C --bg-audio-dir $U8A --out-dir $R/emb/$D/mn_eval \
            --configs w8a8 --conditions bg:10 bg:0 pink:10 pink:0 white:10 white:0 --field ;;
esac; done
echo "$(date '+%F %T') cpu mn $* done" >> $R/logs/cpu_mn_$D.done
