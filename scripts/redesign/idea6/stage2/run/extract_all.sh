#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Idea 6 Stage 2 cheap-tier extraction: EfficientNet-B3 pooled features, FP32 and W8A8, on idea 5 Stage 2's audio.
# Clean first on both datasets (the heads train on it), then the noisy grid: ESC-50 at 10/5/0 dB, 20 dB; UrbanSound8K
# background 10 and 0 dB (where the AST tier exists in full), then the rest of its grid. Each python process runs under
# idea 5 Stage 2's battery guard (pause on battery below 15%, resume on AC or above 25%).
R=$RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba
PY=${PY:-python}
E=$E
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ESC=(--csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out-dir $R/emb/esc50)
URB=(--csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv --out-dir $R/emb/urban8k)
run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $R/logs/extract.done
  $PY scripts/redesign/idea6/stage2/extract_eff.py "$@" --configs fp32 w8a8 --device mps --batch-size 64 --chunk 256 > $R/logs/extract_$name.log 2>&1 &
  local pid=$!
  bash $GUARD $pid >> $R/logs/power_guard.log 2>&1 &
  wait $pid
  echo "$(date '+%F %T') $name exit $?" >> $R/logs/extract.done
}
run esc50_clean "${ESC[@]}" --conditions clean
run urban_clean "${URB[@]}" --conditions clean
run esc50_noisy "${ESC[@]}" --conditions bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0 bg:20 pink:20 white:20
run urban_noisy "${URB[@]}" --conditions bg:10 bg:0 bg:5 pink:10 pink:5 pink:0 white:10 white:5 white:0 bg:20 pink:20 white:20
echo "$(date '+%F %T') all done" >> $R/logs/extract.done
