#!/usr/bin/env bash
# Idea 6 Stage 3b: score every new head on every new embedding file (Stage 2's score_eff.py, unchanged), then analyze
# each task and summarize. Scoring is incremental (files already scored are skipped).
#   R=results S2=idea 6 Stage 2 folder I5=idea 5 Stage 2 folder I7=idea 7 stage 1 folder
#   [NO_MEAN2=1] bash scripts/redesign/idea6/stage3b/run_all.sh [score] [analyze] [summary]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}" "${S2:?}" "${I5:?}" "${I7:?}"
PY=${PY:-$(command -v python)}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT"
H=$ROOT/scripts/redesign/idea6/stage3b
SC=$ROOT/scripts/redesign/idea6/stage2/score_eff.py
steps=${*:-score analyze summary}
ds() { case $1 in urban) echo urban8k ;; *) echo esc50 ;; esac; }
seeds() { case $1 in urban) echo "0 1 2" ;; *) echo "0 1 2 3 4" ;; esac; }
runs() {  # prefix task kinds...
  local pre=$1 t=$2; shift 2
  for s in $(seeds $t); do for k in "$@"; do echo "$R/outputs/${pre}_${t}_s$s/$k"; done; done
}
for step in $steps; do
  case $step in
    score)
      for t in esc50 esc5 urban esc5off; do
        for m in mn10 mn04; do
          dim=$([ $m = mn10 ] && echo 960 || echo 384)
          nice -n 10 $PY $SC --runs $(runs $m $t esc 8 4 bitnet) --emb-dir $R/emb_cnn/$(ds $t) --out $R/scores_cnn/$t \
            --dim $dim --configs ${m}_fp32 ${m}_w8a8 >> $R/logs/score_cnn_${t}.log 2>&1
        done
        for L in 256 384; do
          nice -n 10 $PY $SC --runs $(runs ast$L $t esc 4) --emb-dir $R/emb_ast/$(ds $t)_$L --out $R/scores_ast$L/$t \
            --dim 768 --configs n4 w4g128 >> $R/logs/score_ast${L}_${t}.log 2>&1
        done
        if [ $t != urban ] && [ -z "${NO_MEAN2:-}" ]; then
          nice -n 10 $PY $SC --runs $(runs astm2 $t esc 4) --emb-dir $R/emb_ast/esc50_256x2 --out $R/scores_astm2/$t \
            --dim 768 --configs n4 w4g128 >> $R/logs/score_astm2_${t}.log 2>&1
        fi
      done ;;
    analyze)
      for t in esc50 esc5 urban esc5off; do
        nice -n 10 $PY $H/analyze_3b.py --task $t --r $R --s2 $S2 --i5 $I5 --i7 $I7 ${NO_MEAN2:+--no-mean2} \
          > $R/logs/analyze_${t}.log 2>&1
      done ;;
    summary)
      nice -n 10 $PY $H/summary_3b.py --r $R --s2 $S2 > $R/logs/summary.log 2>&1 ;;
  esac
done
