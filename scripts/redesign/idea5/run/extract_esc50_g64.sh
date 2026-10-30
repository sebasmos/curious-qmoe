#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Group size 64 at 3 bits (group 128 lost 4.5 F1 points at 3 bits on clean ESC-50): clean ESC-50.
# Each condition pass waits while a Stage 1 python process runs (shared GPU, one extraction at a time).
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate curious-qmoe
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$E
COMMON=(--csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out-dir $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/emb/esc50 --device mps --batch-size 16 --chunk 256 --wait-for '^[^ ]*python[0-9.]* .*extract_conditions' '^[^ ]*python[0-9.]* .*redesign-idea5-stage1' '^[^ ]*python[0-9.]* .*curious-qmoe-redesign-idea5-where-it-matters')
date '+%Y-%m-%dT%H:%M:%S%z' > $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/logs/extract_esc50_g64.start
python scripts/redesign/idea5/extract_stage2.py "${COMMON[@]}" --configs w8g64 w3g64 n3g64 --conditions clean
