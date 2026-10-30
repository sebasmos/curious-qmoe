#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Idea 6 Stage 3b: EfficientAT mn04 and mn10 (AudioSet), FP32 and W8A8, pooled embeddings, on Stage 2's audio. Clean
# first on both datasets (the heads train on it), then every noisy condition on both (the CNN is a cheap tier, so its
# humility curve needs every condition). MPS, battery guard as in Stage 2. Code at de710c5.
R=$RESULTS/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db
PY=${PY:-python}
E=$E
WT=$QW/efficientat_release-v0.0.1
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ESC=(--csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out-dir $R/emb_cnn/esc50)
URB=(--csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv --out-dir $R/emb_cnn/urban8k)
NOISY="bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0 bg:20 pink:20 white:20"
run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $R/logs/extract_cnn.done
  nice -n 10 $PY scripts/redesign/idea6/stage3b/extract_cnn.py "$@" --weights-dir $WT --models mn04 mn10 --configs fp32 w8a8 --device mps --batch-size 64 --chunk 256 > $R/logs/extract_cnn_$name.log 2>&1 &
  local pid=$!
  bash $GUARD $pid >> $R/logs/power_guard.log 2>&1 &
  wait $pid
  echo "$(date '+%F %T') $name exit $?" >> $R/logs/extract_cnn.done
}
run esc50_clean "${ESC[@]}" --conditions clean
run urban_clean "${URB[@]}" --conditions clean
run esc50_noisy "${ESC[@]}" --conditions $NOISY
run urban_noisy "${URB[@]}" --conditions $NOISY
echo "$(date '+%F %T') all done" >> $R/logs/extract_cnn.done
