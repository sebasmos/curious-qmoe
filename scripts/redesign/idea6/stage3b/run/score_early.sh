#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Idea 6 Stage 3b: score the systems whose heads and embeddings are complete (ESC-50 CNN and 256-frame AST), while
# the remaining heads train. Same commands as run_all.sh's score step.
R=$RESULTS/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db
PY=${PY:-python}
SC=$W/scripts/redesign/idea6/stage2/score_eff.py
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
runs() { local pre=$1 t=$2; shift 2; for s in 0 1 2 3 4; do for k in "$@"; do echo "$R/outputs/${pre}_${t}_s$s/$k"; done; done; }
for t in esc50 esc5 esc5off; do
  for m in mn10 mn04; do
    dim=$([ $m = mn10 ] && echo 960 || echo 384)
    nice -n 10 $PY $SC --runs $(runs $m $t esc 8 4 bitnet) --emb-dir $R/emb_cnn/esc50 --out $R/scores_cnn/$t --dim $dim --configs ${m}_fp32 ${m}_w8a8 >> $R/logs/score_cnn_${t}.log 2>&1 || echo "cnn $m $t failed" >> $R/logs/score_early.done
  done
  nice -n 10 $PY $SC --runs $(runs ast256 $t esc 4) --emb-dir $R/emb_ast/esc50_256 --out $R/scores_ast256/$t --dim 768 --configs n4 w4g128 >> $R/logs/score_ast256_${t}.log 2>&1 || echo "ast256 $t failed" >> $R/logs/score_early.done
  echo "$(date '+%F %T') $t scored" >> $R/logs/score_early.done
done
