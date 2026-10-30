#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Phase D started in parallel with the UrbanSound8K W6 job (owner, 22 Oct 2026), same command as extract_esc50_20db.sh.
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate curious-qmoe
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$E
date '+%Y-%m-%dT%H:%M:%S%z' > $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/logs/extract_esc50_20db.start
python scripts/redesign/idea5/extract_stage2.py --csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out-dir $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/emb/esc50 --device mps --batch-size 16 --chunk 256 --configs fp32 w8 w6 w5 w4 n4 w8g128 w4g128 n4g128 w8g64 w3g64 n3g64 --conditions bg:20 pink:20 white:20 &
pid=$!
bash $GUARD $pid >> $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/logs/power_guard.log 2>&1 &
wait $pid
echo "esc50 20 dB done $(date)"
