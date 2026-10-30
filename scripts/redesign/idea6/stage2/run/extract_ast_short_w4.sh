#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Amendment 2: w4 (per-channel 4-bit) at the short AST length, same conditions as extract_ast_short.sh. Waits for that
# chain to finish (the GPU is shared), then runs under the battery guard.
R=$RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba
PY=${PY:-python}
E=$E
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
until grep -q "all done" $R/logs/extract_ast_short.done 2>/dev/null; do sleep 30; done
ESC=(--csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out-dir $R/emb_ast_short/esc50 --max-length 512 --cache-bg)
URB=(--csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv --out-dir $R/emb_ast_short/urban8k --max-length 400 --cache-bg)
run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $R/logs/extract_ast_short.done
  $PY scripts/redesign/idea6/stage2/extract_ast_short.py "$@" --configs w4 --device mps --batch-size 16 --chunk 256 > $R/logs/extract_ast_short_$name.log 2>&1 &
  local pid=$!
  bash $GUARD $pid >> $R/logs/power_guard.log 2>&1 &
  wait $pid
  echo "$(date '+%F %T') $name exit $?" >> $R/logs/extract_ast_short.done
}
run w4_esc50 "${ESC[@]}" --conditions clean bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0 bg:20 pink:20 white:20
run w4_urban "${URB[@]}" --conditions clean bg:10 bg:0 --field
echo "$(date '+%F %T') w4 done" >> $R/logs/extract_ast_short.done
