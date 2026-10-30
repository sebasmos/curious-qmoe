#!/bin/bash
# T21 AST extraction driver: every MPS process goes through ~/Documents/qmoe-work/t21_gpu_run.sh (lock, owner claude-t21),
# at most MAXS seconds of work per process, swap checked before each process (wait while used > 9.5 GB).
#   R=... D=d1 bash t21_gpu_ast.sh JOB...   JOB in: check pool_q pool_f extra
: "${R:?}"; D=${D:-d1}
PY=${PY:-python}
W=$HOME/Documents/qmoe-work/t21; X=$W/scripts/redesign/moe/t21/t21_extract_ast.py
GPU=$HOME/Documents/qmoe-work/t21_gpu_run.sh
AUD=${AUD:-$HOME/Documents/DATASETS/FSDKaggle2018/audio_raw}; AUDE=${AUDE:-$AUD}  # D2: AUD=.../FSD50K/audio_raw/dev AUDE=.../eval
U8C=$HOME/Documents/DATASETS/Urban8k/ast_768/urbansound8k.csv
U8A=$HOME/Documents/DATASETS/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio
MAXS=${MAXS:-4500}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
swapwait() {  # coordinator rule (2026-10-26): above 9.5 GB swap, proceed while memory_pressure reports at least 25% free
  while :; do u=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); [ "${u%.*}" -lt 9500 ] && return
    fr=$(memory_pressure | tail -1 | sed -E 's/.*: ([0-9]+)%.*/\1/'); [ "$fr" -ge 25 ] && return
    echo "$(date +%T) swap used ${u}M, memory free ${fr}%, waiting"; sleep 120; done; }
run() {  # name, args...; repeats until the process reports nothing left to do
  local name=$1; shift
  for i in $(seq 1 30); do
    swapwait
    echo "$(date '+%F %T') start $name round $i" >> $R/logs/gpu_ast.done
    $GPU nice -n 10 $PY $X "$@" --device mps --max-seconds $MAXS >> $R/logs/extract_ast_$name.log 2>&1
    rc=$?
    echo "$(date '+%F %T') $name round $i exit $rc" >> $R/logs/gpu_ast.done
    [ $rc -eq 99 ] && continue
    [ $rc -eq 3 ] && continue
    return $rc
  done
}
for job in "$@"; do
  case $job in
    check) swapwait; $GPU nice -n 10 $PY $X --check --check-n 32 --csv $R/csv/$D/check.csv --audio-dir $AUD \
             --out-dir $R/emb/$D/ast_pool --configs fp32 w4g128 --device mps >> $R/logs/extract_ast_check.log 2>&1
           echo "$(date '+%F %T') check exit $?" >> $R/logs/gpu_ast.done ;;
    pool_q) run pool_q --csv $R/csv/$D/pool.csv --audio-dir $AUD --bg-csv $U8C --bg-audio-dir $U8A --out-dir $R/emb/$D/ast_pool \
             --configs w4g128 --conditions clean bg:10 bg:0 pink:10 pink:0 white:10 white:0 --field ${AVAIL:+--available-only} ;;
    pool_f) run pool_f --csv $R/csv/$D/pool.csv --audio-dir $AUD --out-dir $R/emb/$D/ast_pool --configs fp32 --conditions clean ${AVAIL:+--available-only} ;;
    extra)  run extra --csv $R/csv/$D/extra.csv --audio-dir $AUDE --out-dir $R/emb/$D/ast_extra --configs fp32 w4g128 --conditions clean ;;
  esac
done
echo "$(date '+%F %T') jobs $* finished" >> $R/logs/gpu_ast.done
