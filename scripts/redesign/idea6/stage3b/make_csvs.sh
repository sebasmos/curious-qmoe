#!/usr/bin/env bash
# Idea 6 Stage 3b: training CSVs for every new head from the clean FP32 embeddings (Stage 2's make_csv.py), and the
# official-category copies for esc5off (Stage 2's derive_off5.py). Waits for each clean file to appear.
#   R=... bash scripts/redesign/idea6/stage3b/make_csvs.sh
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
make $R/emb_cnn/esc50/mn10_fp32__clean.npy esc50 esc-50_mn10_fp32.csv
make $R/emb_cnn/esc50/mn04_fp32__clean.npy esc50 esc-50_mn04_fp32.csv
make $R/emb_ast/esc50_256/fp32__clean.npy esc50 esc-50_ast256_fp32.csv
make $R/emb_ast/urban8k_256/fp32__clean.npy urban8k urbansound8k_ast256_fp32.csv
make $R/emb_cnn/urban8k/mn10_fp32__clean.npy urban8k urbansound8k_mn10_fp32.csv
make $R/emb_cnn/urban8k/mn04_fp32__clean.npy urban8k urbansound8k_mn04_fp32.csv
make $R/emb_ast/esc50_384/fp32__clean.npy esc50 esc-50_ast384_fp32.csv
make $R/emb_ast/urban8k_384/fp32__clean.npy urban8k urbansound8k_ast384_fp32.csv
until [ -f $R/emb_ast/esc50_256o256/fp32__clean.json ]; do sleep 30; done
nice -n 10 $PY $ROOT/scripts/redesign/idea6/stage3b/make_mean2.py --first $R/emb_ast/esc50_256 --second $R/emb_ast/esc50_256o256 --out $R/emb_ast/esc50_256x2
make $R/emb_ast/esc50_256x2/fp32__clean.npy esc50 esc-50_astm2_fp32.csv
echo "$(date +%T) all CSVs"
