#!/usr/bin/env bash
# Idea 6 option B driver after the extractions: training CSVs as the clean FP32 embeddings appear, every head
# (CNN jobs first, the AST jobs wait for their CSV), then the universe scores. CPU only, nice 10, 2 threads per job,
# P head jobs at a time.
#   R=... nohup bash scripts/redesign/idea6/optionB/pipeline_optb.sh [P] &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}"
P=${1:-2}
PY=${PY:-python}
OB=$ROOT/scripts/redesign/idea6/optionB
META=$HOME/Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv
LOG=$R/logs/pipeline.log
mkdir -p "$R/csv" "$R/logs"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT"
csv() {  # emb npy, csv name
  until [ -f "$1" ] && [ -f "${1%.npy}.json" ]; do sleep 60; done
  [ -f "$R/csv/$2" ] || { nice -n 10 $PY $OB/make_train_csv.py --emb "$1" --meta $META --out "$R/csv/$2.tmp.csv" && mv "$R/csv/$2.tmp.csv" "$R/csv/$2"; }
  echo "$(date '+%F %T') csv $2" >> "$LOG"
}
csv "$R/emb_cnn/tau/mn10_fp32__clean.npy" tau_mn10_fp32.csv
csv "$R/emb_cnn/tau/mn20_fp32__clean.npy" tau_mn20_fp32.csv
( csv "$R/emb_ast/tau/fp32__clean.npy" tau_ast_fp32.csv ) &
JOBS=$R/logs/head_jobs.txt
{ for pre in mn10 mn20; do for s in 0 1 2; do for k in esc 4 8 bitnet; do echo "$pre $s $k"; done; done; done
  for s in 0 1 2; do for k in esc 4; do echo "ast $s $k"; done; done; } > $JOBS
echo "$(date '+%F %T') heads start, P=$P" >> "$LOG"
R=$R JOBS=$JOBS bash $OB/train_heads.sh $P
wait
echo "$(date '+%F %T') heads done: $(grep -c 'exit 0' $R/logs/train_heads.done) of 30 exit 0" >> "$LOG"
runs() { local pre=$1; shift; for s in 0 1 2; do for k in "$@"; do echo "$R/outputs/${pre}_tau_s$s/$k"; done; done; }
for m in mn10 mn20; do
  nice -n 10 $PY $OB/score_optb.py --runs $(runs $m esc 4 8 bitnet) --emb-dir $R/emb_cnn/tau --configs ${m}_fp32 ${m}_w8a8 --meta $META --out $R/scores >> $R/logs/score_$m.log 2>&1
  echo "$(date '+%F %T') score $m exit $?" >> "$LOG"
done
nice -n 10 $PY $OB/score_optb.py --runs $(runs ast esc 4) --emb-dir $R/emb_ast/tau --configs fp32 w8g64 w4g128 n4 --meta $META --out $R/scores >> $R/logs/score_ast.log 2>&1
echo "$(date '+%F %T') score ast exit $?" >> "$LOG"
echo "$(date '+%F %T') pipeline done" >> "$LOG"
