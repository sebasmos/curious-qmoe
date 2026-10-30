#!/usr/bin/env bash
# Idea 6 Stage 3c: training CSVs for the heads of mn05, mn20 and dymn04 from the clean FP32 embeddings (Stage 2's
# make_csv.py), and the official-category copies for esc5off (Stage 2's derive_off5.py). Waits for each clean file.
#   R=... bash scripts/redesign/idea6/stage3c/make_csvs.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}"
PY=${PY:-$(command -v python)}
E=$HOME/Documents/DATASETS
S2=$ROOT/scripts/redesign/idea6/stage2
mkdir -p $R/csv
make() {  # emb file, dataset (esc50|urban8k), csv name
  local emb=$1 ds=$2 out=$R/csv/$3 meta
  case $ds in esc50) meta=$E/ESC-50/ast_768/esc-50.csv ;; urban8k) meta=$E/Urban8k/ast_768/urbansound8k.csv ;; esac
  until [ -f "$emb" ] && [ -f "${emb%.npy}.json" ]; do sleep 30; done
  if [ ! -f "$out" ]; then
    nice -n 10 $PY $S2/make_csv.py --emb "$emb" --meta $meta --out "$out.tmp.csv" && mv "$out.tmp.csv" "$out"
  fi
  if [ $ds = esc50 ] && [ ! -f "${out%.csv}_off5.csv" ]; then
    nice -n 10 $PY $S2/derive_off5.py "$out" "${out%.csv}_off5.tmp.csv" && mv "${out%.csv}_off5.tmp.csv" "${out%.csv}_off5.csv"
  fi
  echo "$(date +%T) $out"
}
for m in mn05 mn20 dymn04; do make $R/emb_cnn/esc50/${m}_fp32__clean.npy esc50 esc-50_${m}_fp32.csv; done
for m in mn05 mn20 dymn04; do make $R/emb_cnn/urban8k/${m}_fp32__clean.npy urban8k urbansound8k_${m}_fp32.csv; done
echo "$(date +%T) all CSVs"
