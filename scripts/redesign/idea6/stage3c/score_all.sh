#!/usr/bin/env bash
# Idea 6 Stage 3c scoring (Stage 2's score_eff.py, unchanged, incremental):
#   cnn   the new CNN heads (mn05, mn20, dymn04; kinds esc 8 4 bitnet) on their embedding files -> R/scores_cnn/<task>
#   unp   Stage 3b's mn10 UrbanSound8K heads on the exact-length embeddings -> R/scores_unp/urban (P5)
# Each head's run directory must be complete (summary.json); the scorer re-checks every fold's F1 against it.
#   R=... S3B=... bash scripts/redesign/idea6/stage3c/score_all.sh [cnn] [unp]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}" "${S3B:?}"
PY=${PY:-$(command -v python)}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT"
SC=$ROOT/scripts/redesign/idea6/stage2/score_eff.py
ds() { case $1 in urban) echo urban8k ;; *) echo esc50 ;; esac; }
seeds() { case $1 in urban) echo "0 1 2" ;; *) echo "0 1 2 3 4" ;; esac; }
runs() { local root=$1 pre=$2 t=$3; shift 3; for s in $(seeds $t); do for k in "$@"; do echo "$root/outputs/${pre}_${t}_s$s/$k"; done; done; }
dim() { case $1 in mn05) echo 480 ;; mn20) echo 1920 ;; dymn04) echo 384 ;; mn10) echo 960 ;; esac; }
for step in "${@:-cnn unp}"; do
  case $step in
    cnn)
      for t in esc50 esc5 urban esc5off; do
        for m in mn05 mn20 dymn04; do
          for r in $(runs $R $m $t esc 8 4 bitnet); do until [ -f $r/summary.json ]; do sleep 30; done; done
          nice -n 10 $PY $SC --runs $(runs $R $m $t esc 8 4 bitnet) --emb-dir $R/emb_cnn/$(ds $t) --out $R/scores_cnn/$t \
            --dim $(dim $m) --configs ${m}_fp32 ${m}_w8a8 >> $R/logs/score_cnn_${t}.log 2>&1
          echo "$(date '+%F %T') score cnn $m $t exit $?" >> $R/logs/score.done
        done
      done ;;
    unp)
      nice -n 10 $PY $SC --runs $(runs $S3B mn10 urban esc 8 4 bitnet) --emb-dir $R/emb_cnn/urban8k_unpadded --out $R/scores_unp/urban \
        --dim 960 --configs mn10_fp32 mn10_w8a8 >> $R/logs/score_unp_urban.log 2>&1
      echo "$(date '+%F %T') score unp urban exit $?" >> $R/logs/score.done ;;
  esac
done
