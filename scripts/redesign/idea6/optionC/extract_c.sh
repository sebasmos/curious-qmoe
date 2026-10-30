#!/bin/bash
# Idea 6 option C: two more AST configurations at the clips' real length (512 frames on ESC-50, 400 on UrbanSound8K):
# n3g64 (nested 3-bit child of w8g64, group 64) and w3g32 (direct 3-bit, group 32), on Stage 2's audio (same CSVs,
# background clips, offsets and field assignments: extract_ast_short.py with --cache-bg reproduces Stage 2's
# bg_plan). w3g64 at 512/400 (Stage 2) and n4 at 256 (Stage 3b) already exist and are reused. Two MPS processes
# (ESC-50 and UrbanSound8K) at once, 2 CPU threads each. Conditions as Stage 2: ESC-50 clean and the twelve noisy
# cells; UrbanSound8K clean, background 10 and 0 dB in full, the other field cells for the clips the assignments need.
#   R=results folder bash scripts/redesign/idea6/optionC/extract_c.sh
set -u
: "${R:?set R to the results folder}"
W="$(cd "$(dirname "$0")/../../../.." && pwd)"
PY=${PY:-python}
E=$HOME/Documents/DATASETS
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1
mkdir -p $R/logs $R/emb_ast_short/esc50 $R/emb_ast_short/urban8k
CFG="n3g64 w3g32"
ESC=(--csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out-dir $R/emb_ast_short/esc50 --max-length 512 --cache-bg)
URB=(--csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv --out-dir $R/emb_ast_short/urban8k --max-length 400 --cache-bg)
run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $R/logs/extract.done
  nice -n 10 $PY scripts/redesign/idea6/stage2/extract_ast_short.py "$@" --configs $CFG --device mps --batch-size 16 --chunk 256 --threads 2 > $R/logs/extract_$name.log 2>&1
  echo "$(date '+%F %T') $name exit $?" >> $R/logs/extract.done
}
run esc50 "${ESC[@]}" --conditions clean bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0 bg:20 pink:20 white:20 &
run urban "${URB[@]}" --conditions clean bg:10 bg:0 --field &
wait
echo "$(date '+%F %T') all done" >> $R/logs/extract.done
