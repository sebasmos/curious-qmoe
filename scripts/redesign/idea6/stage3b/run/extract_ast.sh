#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Idea 6 Stage 3b: static ASTs at 256 and 384 input frames (first frames), configurations n4 and w4g128 on every
# condition, fp32 on clean only (the heads train on it), with Stage 2's extraction and quantization code
# (extract_ast_short.py, QWave/backbone_quant.py, QWave/ast_frames.py) and Stage 2's audio. ESC-50: clean and all 12
# noisy conditions; UrbanSound8K: clean, background 10 and 0 dB in full and the other field cells for the clips the
# field assignments need (Stage 2's short-AST set: the AST tier exists only there). Then the second 256-frame window
# of each ESC-50 clip (frames 256 to 511) for the 2-window mean. MPS, battery guard as in Stage 2.
R=$RESULTS/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db
PY=${PY:-python}
E=$E
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
ESCA=(--csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --cache-bg)
URBA=(--csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv --cache-bg)
NOISY="bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0 bg:20 pink:20 white:20"
run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $R/logs/extract_ast.done
  nice -n 10 $PY scripts/redesign/idea6/stage2/extract_ast_short.py "$@" --device mps --batch-size 16 --chunk 256 > $R/logs/extract_ast_$name.log 2>&1 &
  local pid=$!
  bash $GUARD $pid >> $R/logs/power_guard.log 2>&1 &
  wait $pid
  echo "$(date '+%F %T') $name exit $?" >> $R/logs/extract_ast.done
}
for L in 256 384; do
  run esc50_${L}_clean "${ESCA[@]}" --out-dir $R/emb_ast/esc50_$L --max-length $L --configs fp32 n4 w4g128 --conditions clean
  run urban_${L}_clean "${URBA[@]}" --out-dir $R/emb_ast/urban8k_$L --max-length $L --configs fp32 n4 w4g128 --conditions clean
done
for L in 256 384; do
  run esc50_${L}_noisy "${ESCA[@]}" --out-dir $R/emb_ast/esc50_$L --max-length $L --configs n4 w4g128 --conditions $NOISY
  run urban_${L}_noisy "${URBA[@]}" --out-dir $R/emb_ast/urban8k_$L --max-length $L --configs n4 w4g128 --conditions bg:10 bg:0 --field
done
run esc50_256o256_clean "${ESCA[@]}" --out-dir $R/emb_ast/esc50_256o256 --max-length 256 --frame-offset 256 --configs fp32 n4 w4g128 --conditions clean
run esc50_256o256_noisy "${ESCA[@]}" --out-dir $R/emb_ast/esc50_256o256 --max-length 256 --frame-offset 256 --configs n4 w4g128 --conditions $NOISY
echo "$(date '+%F %T') all done" >> $R/logs/extract_ast.done
