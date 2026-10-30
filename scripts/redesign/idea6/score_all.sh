#!/usr/bin/env bash
# Score every cheap-tier head: the protocol 3 singles (ESC-50 5 classes and UrbanSound8K,
# kinds esc, 8, 4, bitnet) and this stage's ESC-50 50-class heads.
#   R=/path/results P3=/path/protocol-v3-singles/outputs bash scripts/redesign/idea6/score_all.sh [esc5|urban|esc50 ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${R:?}" "${P3:?}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT"
for task in "${@:-esc5 urban esc50}"; do
  runs=()
  case $task in
    esc5)  for s in 0 1 2 3 4; do for k in esc 8 4 bitnet; do runs+=("$P3/esc_singles_s$s/$k"); done; done ;;
    urban) for s in 0 1 2; do for k in esc 8 4 bitnet; do runs+=("$P3/urban_singles_s$s/$k"); done; done ;;
    esc50) for s in 0 1 2 3 4; do for k in esc 8 4 bitnet; do runs+=("$R/outputs/effesc50_s$s/$k"); done; done ;;
  esac
  nice -n 10 python "$ROOT/scripts/redesign/idea6/score_eff.py" --runs "${runs[@]}" --out "$R/scores/$task"
done
