#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Idea 6 Stage 2, after the Stage 1 audit (coordinator, 22 Oct 2026 ~11:45): static ASTs on unpadded input, 512 frames
# for ESC-50 and 400 for UrbanSound8K, configurations fp32 w8g64 w4g128 n4 w3g64 (the coordinator's list), on idea 5
# Stage 2's audio. Order: ESC-50 clean and the nine field cells in full; UrbanSound8K clean, background 10 and 0 dB in
# full and the other field cells for the clips the field assignments need; ESC-50 at 20 dB. Battery guard as before.
R=$RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba
PY=${PY:-python}
E=$E
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
CFG="fp32 w8g64 w4g128 n4 w3g64"
ESC=(--csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out-dir $R/emb_ast_short/esc50 --max-length 512 --cache-bg)
URB=(--csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv --out-dir $R/emb_ast_short/urban8k --max-length 400 --cache-bg)
run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $R/logs/extract_ast_short.done
  $PY scripts/redesign/idea6/stage2/extract_ast_short.py "$@" --configs $CFG --device mps --batch-size 16 --chunk 256 > $R/logs/extract_ast_short_$name.log 2>&1 &
  local pid=$!
  bash $GUARD $pid >> $R/logs/power_guard.log 2>&1 &
  wait $pid
  echo "$(date '+%F %T') $name exit $?" >> $R/logs/extract_ast_short.done
}
run esc50_main "${ESC[@]}" --conditions clean bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0
run urban "${URB[@]}" --conditions clean bg:10 bg:0 --field
run esc50_20db "${ESC[@]}" --conditions bg:20 pink:20 white:20
echo "$(date '+%F %T') all done" >> $R/logs/extract_ast_short.done
