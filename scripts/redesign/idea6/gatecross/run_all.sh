#!/usr/bin/env bash
# Gate quality by precision. Reads stored score files only (no extraction, scoring or training); CPU, nice 10, 2 threads.
#   OUT=~/Documents/qmoe-results/<folder> bash scripts/redesign/idea6/gatecross/run_all.sh
set -euo pipefail
: "${OUT:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-${PY:-python}}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$OUT/logs"
nice -n 10 $PY "$HERE/gate_by_precision.py" --out "$OUT" 2>&1 | tee "$OUT/logs/gate_by_precision.log"
nice -n 10 $PY "$HERE/crosscheck_f1.py" --out "$OUT" 2>&1 | tee "$OUT/logs/crosscheck_f1.log"
nice -n 10 $PY "$HERE/make_report.py" --out "$OUT" 2>&1 | tee "$OUT/logs/make_report.log"
