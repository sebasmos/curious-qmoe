#!/usr/bin/env bash
# Idea 6 option B: clean embeddings of every TAU 2020 Mobile segment (all devices, the universe CSV), two MPS
# processes: the AST at its native 1024 frames (fp32, w8g64, w4g128, n4; Stage 2's extractor, one input length) and
# EfficientAT mn10 and mn20 at FP32 and W8A8 (Stage 3c's extractor; every clip is 10 s, one graph per model).
#   R=... bash scripts/redesign/idea6/optionB/extract_optb.sh [ast|cnn]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}"
PY=${PY:-${PY:-python}}
CSV=$HOME/Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv
AUDIO=$HOME/Documents/DATASETS/TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968/TAU-urban-acoustic-scenes-2020-mobile-development/audio
WT=$HOME/Documents/qmoe-weights/efficientat_release-v0.0.1
mkdir -p "$R/logs" "$R/emb_ast/tau" "$R/emb_cnn/tau"
for step in "${@:-ast cnn}"; do
  case $step in
    ast) cd "$ROOT" && nice -n 5 "$PY" scripts/redesign/idea6/stage2/extract_ast_short.py --csv "$CSV" --audio-dir "$AUDIO" \
           --out-dir "$R/emb_ast/tau" --configs fp32 w8g64 w4g128 n4 --conditions clean --max-length 1024 \
           --device mps --batch-size 16 --chunk 256 --threads 2 > "$R/logs/extract_ast.log" 2>&1
         echo "$(date '+%F %T') extract ast exit $?" >> "$R/logs/extract.done" ;;
    cnn) cd "$ROOT" && nice -n 5 "$PY" scripts/redesign/idea6/stage3c/extract_cnn.py --csv "$CSV" --audio-dir "$AUDIO" \
           --out-dir "$R/emb_cnn/tau" --weights-dir "$WT" --models mn10 mn20 --configs fp32 w8a8 --conditions clean \
           --device mps --batch-size 64 --chunk 256 --threads 2 --no-audio-cache > "$R/logs/extract_cnn.log" 2>&1
         echo "$(date '+%F %T') extract cnn exit $?" >> "$R/logs/extract.done" ;;
  esac
done
