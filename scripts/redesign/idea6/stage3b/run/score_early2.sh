#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Idea 6 Stage 3b: second early scoring pass (UrbanSound8K CNN and 256-frame AST, ESC-50 384-frame AST where every
# head is done). Same commands as run_all.sh's score step; run_all.sh skips what is already scored.
R=$RESULTS/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db
PY=${PY:-python}
SC=$W/scripts/redesign/idea6/stage2/score_eff.py
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W
runs() { local pre=$1 t=$2 n=$3; shift 3; for s in $(seq 0 $n); do for k in "$@"; do echo "$R/outputs/${pre}_${t}_s$s/$k"; done; done; }
for m in mn10 mn04; do
  dim=$([ $m = mn10 ] && echo 960 || echo 384)
  nice -n 10 $PY $SC --runs $(runs $m urban 2 esc 8 4 bitnet) --emb-dir $R/emb_cnn/urban8k --out $R/scores_cnn/urban --dim $dim --configs ${m}_fp32 ${m}_w8a8 >> $R/logs/score_cnn_urban.log 2>&1 || echo "cnn $m urban failed" >> $R/logs/score_early.done
done
nice -n 10 $PY $SC --runs $(runs ast256 urban 2 esc 4) --emb-dir $R/emb_ast/urban8k_256 --out $R/scores_ast256/urban --dim 768 --configs n4 w4g128 >> $R/logs/score_ast256_urban.log 2>&1 || echo "ast256 urban failed" >> $R/logs/score_early.done
for t in esc50 esc5; do
  nice -n 10 $PY $SC --runs $(runs ast384 $t 4 esc 4) --emb-dir $R/emb_ast/esc50_384 --out $R/scores_ast384/$t --dim 768 --configs n4 w4g128 >> $R/logs/score_ast384_${t}.log 2>&1 || echo "ast384 $t failed" >> $R/logs/score_early.done
done
echo "$(date '+%F %T') second early pass scored" >> $R/logs/score_early.done
