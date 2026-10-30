#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# W6 on UrbanSound8K (clean, background 10 and 0 dB, field cells): the static front needs it, because cascades
# costing between W6 and W8 were otherwise compared with W5 (Phase C ran the coordinator's list, which had no W6).
# Waits for the Phase C process to end; runs under the battery guard.
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate curious-qmoe
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
E=$E
while kill -0 22333 2>/dev/null; do sleep 30; done
date '+%Y-%m-%dT%H:%M:%S%z' > $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/logs/extract_urban_w6.start
python scripts/redesign/idea5/extract_stage2.py --csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv --out-dir $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/emb/urban8k --device mps --batch-size 16 --chunk 256 --cache-bg --configs w6 --conditions clean bg:10 bg:0 --field &
pid=$!
bash $GUARD $pid >> $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/logs/power_guard.log 2>&1 &
wait $pid
echo "urban w6 done $(date)"
