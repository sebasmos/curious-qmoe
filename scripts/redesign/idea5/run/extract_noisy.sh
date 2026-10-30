#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Phase B: ESC-50 under the nine field-mixture cells (bg, pink, white at 10, 5, 0 dB), 12 configurations
#   (W3 per channel, W2 group and nested 2-bit collapse on clean audio and are left out; group 64 replaces
#   group 128 at 3 bits after the clean pass: W3g64 0.925 vs W3g128 0.906 macro F1 on ESC-50 50 classes).
# Phase C: UrbanSound8K clean, background 10 and 0 dB, and the field-mixture cells for the clips that need them.
# Phase D: ESC-50 at 20 dB (bg, pink, white) for the humility curve.
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate curious-qmoe
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$E
GUARD=(--wait-for '^[^ ]*python[0-9.]* .*extract_conditions' '^[^ ]*python[0-9.]* .*redesign-idea5-stage1' '^[^ ]*python[0-9.]* .*curious-qmoe-redesign-idea5-where-it-matters')
ESC=(--csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out-dir $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/emb/esc50 --device mps --batch-size 16 --chunk 256)
URB=(--csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv --out-dir $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/emb/urban8k --device mps --batch-size 16 --chunk 256 --cache-bg)
ESC_CFG="fp32 w8 w6 w5 w4 n4 w8g128 w4g128 n4g128 w8g64 w3g64 n3g64"
URB_CFG="fp32 w8 w5 w4 n4 w8g128 w4g128 n4g128 w8g64 w3g64 n3g64"
date '+%Y-%m-%dT%H:%M:%S%z' > $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/logs/extract_noisy.start
python scripts/redesign/idea5/extract_stage2.py "${ESC[@]}" "${GUARD[@]}" --configs $ESC_CFG --conditions bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0 || exit 1
echo "phase B done $(date)"
python scripts/redesign/idea5/extract_stage2.py "${URB[@]}" "${GUARD[@]}" --configs $URB_CFG --conditions clean bg:10 bg:0 --field || exit 1
echo "phase C done $(date)"
python scripts/redesign/idea5/extract_stage2.py "${ESC[@]}" "${GUARD[@]}" --configs $ESC_CFG --conditions bg:20 pink:20 white:20 || exit 1
echo "phase D done $(date)"
