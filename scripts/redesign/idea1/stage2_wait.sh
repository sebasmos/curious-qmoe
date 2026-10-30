#!/bin/bash
# Redesign idea 1, stage 2: wait for protocol 3 single-model seeds to finish training,
# then extract each one with stage2_extract.py. Exits when every seed is extracted, or
# with status 1 when the training queue has finished and a seed is still incomplete.
#
# usage: stage2_wait.sh SINGLES_DIR OUT_DIR DS_TAG NAME SEED [SEED ...]
#   e.g. stage2_wait.sh .../2026-10-20_2127_protocol-v3-singles_code-bc4ff1c .../extract quinn quinn 0 1 2
#   reads SINGLES_DIR/outputs/<DS_TAG>_singles_s<seed>, writes OUT_DIR/<NAME>_s<seed>
set -u
singles=$1; out=$2; tag=$3; name=$4; shift 4
here="$(cd "$(dirname "$0")" && pwd)"
code=/Users/cajas.sebastian/Desktop/repositories/curious-qmoe-run-v3-bc4ff1c
models="1 2 4 8 16 bitnet esc qesc"
todo="$*"
while [ -n "$todo" ]; do
  finished=0
  grep -q "training done" "$singles/runq/progress.txt" 2>/dev/null && finished=1
  left=""
  for s in $todo; do
    run="$singles/outputs/${tag}_singles_s$s"
    ok=1
    for m in $models; do [ -f "$run/$m/summary.json" ] || ok=0; done
    if [ $ok = 1 ]; then
      echo "$(date '+%F %T') extracting $run"
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$code" \
        python "$here/stage2_extract.py" --run "$run" --out "$out/${name}_s$s" --code "$code" \
        > "$out/${name}_s$s.log" 2>&1 || { echo "extraction failed for $run"; exit 1; }
      echo "$(date '+%F %T') done ${name}_s$s"
    else
      left="$left $s"
    fi
  done
  todo="${left# }"
  [ -z "$todo" ] && break
  if [ $finished = 1 ]; then
    echo "training queue finished but seeds $todo are incomplete"; exit 1
  fi
  sleep 120
done
echo "$(date '+%F %T') all seeds of $name extracted"
