#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c extraction (code 9c9cbf1). Four processes on MPS, 2 CPU threads each, nice 10:
#   esc50:      mn05 mn20 dymn04, FP32 and W8A8, clean and all 12 noisy conditions (5 s clips, no padding)
#   urban_a/b:  the same on UrbanSound8K padded to 64,587 samples (Stage 3b amendment 1); every background
#               condition in process a so bg_plan.json has one writer
#   urban_unp:  P5, mn10 FP32 and W8A8 on UrbanSound8K at exact lengths (--tail-cpu), clean and all 12 noisy
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
PY=${PY:-python}
E=${E:-$HOME/Documents/DATASETS}
WT=${WT:?set WT to the EfficientAT release v0.0.1 weights folder}
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ESC=(--csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out-dir $R/emb_cnn/esc50)
URB=(--csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv)
NEW="--models mn05 mn20 dymn04"
ALL="clean bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0 bg:20 pink:20 white:20"
run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $R/logs/extract.done
  nice -n 10 $PY scripts/redesign/idea6/stage3c/extract_cnn.py "$@" --weights-dir $WT --configs fp32 w8a8 --device mps --batch-size 64 --chunk 256 > $R/logs/extract_$name.log 2>&1
  echo "$(date '+%F %T') $name exit $?" >> $R/logs/extract.done
}
run esc50 "${ESC[@]}" $NEW --conditions $ALL &
run urban_a "${URB[@]}" --out-dir $R/emb_cnn/urban8k --pad-to-samples 64587 $NEW --conditions clean bg:10 bg:5 bg:0 bg:20 pink:10 pink:20 &
run urban_b "${URB[@]}" --out-dir $R/emb_cnn/urban8k --pad-to-samples 64587 $NEW --conditions pink:5 pink:0 white:10 white:5 white:0 white:20 &
run urban_unp "${URB[@]}" --out-dir $R/emb_cnn/urban8k_unpadded --tail-cpu --models mn10 --conditions $ALL &
wait
echo "$(date '+%F %T') all done" >> $R/logs/extract.done
