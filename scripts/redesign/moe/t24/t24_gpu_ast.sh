#!/bin/bash
# T24 AST extraction driver: every MPS process goes through ~/Documents/qmoe-work/fair_gpu_run.sh (FIFO queue, tag
# claude-t24), at most MAXS seconds of work per process, memory checked before each process (T21's driver otherwise).
#   R=... D=d3 AUD=<audio16k_t24 dir> bash t24_gpu_ast.sh JOB...   JOB in: check eval_q all_f all_q
: "${R:?}"; D=${D:?}; AUD=${AUD:?}
PY=${PY:-python}
W=$HOME/Documents/qmoe-work/t24; X=$W/scripts/redesign/moe/t24/t24_extract.py
GPU="$HOME/Documents/qmoe-work/fair_gpu_run.sh claude-t24"
U8C=$HOME/Documents/DATASETS/Urban8k/ast_768/urbansound8k.csv
U8A=$HOME/Documents/DATASETS/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio
MAXS=${MAXS:-4500}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
memwait() {  # proceed only while memory_pressure reports at least 25% free and free disk is above 25 GB
  while :; do fr=$(memory_pressure | tail -1 | sed -E 's/.*: ([0-9]+)%.*/\1/'); dk=$(df -g $HOME | tail -1 | awk '{print $4}')
    [ "$fr" -ge 25 ] && [ "$dk" -gt 25 ] && return
    echo "$(date +%T) memory free ${fr}%, disk ${dk} GB, waiting" >> $R/logs/gpu_ast_$D.done; sleep 120; done; }
run() {  # name, args...; repeats until the process reports nothing left to do
  local name=$1; shift
  for i in $(seq 1 40); do
    memwait
    echo "$(date '+%F %T') start $name round $i" >> $R/logs/gpu_ast_$D.done
    $GPU nice -n 10 $PY $X ast "$@" --device mps --max-seconds $MAXS >> $R/logs/extract_ast_${D}_$name.log 2>&1
    rc=$?
    echo "$(date '+%F %T') $name round $i exit $rc" >> $R/logs/gpu_ast_$D.done
    [ $rc -eq 99 ] && continue
    [ $rc -eq 3 ] && continue
    return $rc
  done
}
for job in "$@"; do
  case $job in
    check) memwait; $GPU nice -n 10 $PY $X ast --check --check-n 32 --csv $R/csv/$D/eval.csv --audio-dir $AUD \
             --out-dir $R/emb/$D/ast_check --configs fp32 w4g128 --device mps >> $R/logs/extract_ast_${D}_check.log 2>&1
           echo "$(date '+%F %T') check exit $?" >> $R/logs/gpu_ast_$D.done ;;
    eval_q) run eval_q --csv $R/csv/$D/eval.csv --audio-dir $AUD --bg-csv $U8C --bg-audio-dir $U8A --out-dir $R/emb/$D/ast_eval \
             --configs w4g128 --conditions bg:10 bg:0 pink:10 pink:0 white:10 white:0 --field ;;
    all_f) run all_f --csv $R/csv/$D/all.csv --audio-dir $AUD --out-dir $R/emb/$D/ast_all --configs fp32 --conditions clean ;;
    all_q) run all_q --csv $R/csv/$D/all.csv --audio-dir $AUD --out-dir $R/emb/$D/ast_all --configs w4g128 --conditions clean ;;
  esac
done
echo "$(date '+%F %T') jobs $* finished" >> $R/logs/gpu_ast_$D.done
