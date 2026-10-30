#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Phase A: ESC-50 clean audio under every configuration, then the uint8-P ablation (W8, W4).
# Each condition pass waits while a Stage 1 python process runs (shared GPU, one extraction at a time).
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate curious-qmoe
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$E
COMMON=(--csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out-dir $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/emb/esc50 --device mps --batch-size 16 --chunk 256 --wait-for '^[^ ]*python[0-9.]* .*extract_conditions' '^[^ ]*python[0-9.]* .*redesign-idea5-stage1' '^[^ ]*python[0-9.]* .*curious-qmoe-redesign-idea5-where-it-matters')
date '+%Y-%m-%dT%H:%M:%S%z' > $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/logs/extract_esc50_clean.start
python scripts/redesign/idea5/extract_stage2.py "${COMMON[@]}" --configs fp32 w8 w6 w5 w4 w3 w8g128 w4g128 w3g128 w2g128 n4 n3 n4g128 n3g128 n2g128 --conditions clean
python scripts/redesign/idea5/extract_stage2.py "${COMMON[@]}" --configs w8 w4 --conditions clean --p-bits 8 --name-suffix _p8
