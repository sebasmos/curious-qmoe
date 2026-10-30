#!/usr/bin/env bash
# Idea 6 Stage 2: score every head on every embedding file, then analyze each task and summarize.
#   R=results folder S2=idea 5 Stage 2 folder I7=idea 7 stage 1 folder S1=idea 6 Stage 1 folder
#   bash scripts/redesign/idea6/stage2/run_all.sh [score] [analyze] [summary]
# Tasks esc50, esc5, urban (pre-registered) and esc5off (amendment 3; its 1024-frame AST heads are scored here on
# idea 5 Stage 2's embeddings into scores_ast1024/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}" "${S2:?}" "${I7:?}" "${S1:?}"
PY=${PY:-$(command -v python)}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT"
H=$ROOT/scripts/redesign/idea6/stage2
steps=${*:-score analyze summary}
emb() { case $1 in urban) echo urban8k ;; *) echo esc50 ;; esac; }
asts() { case $1 in esc5off) echo $R/scores_ast1024/esc5off ;; *) echo $S2/scores/$1 ;; esac; }
stats() { case $1 in urban) echo urban ;; *) echo esc50 ;; esac; }
seeds() { case $1 in urban) echo "0 1 2" ;; *) echo "0 1 2 3 4" ;; esac; }
for step in $steps; do
  for t in esc50 esc5 urban esc5off; do
    case $step in
      score)
        runs=(); for s in $(seeds $t); do for k in esc 8 4 bitnet; do runs+=("$R/outputs/effp_${t}_s$s/$k"); done; done
        nice -n 10 $PY $H/score_eff.py --runs "${runs[@]}" --emb-dir $R/emb/$(emb $t) --out $R/scores/$t \
          >> $R/logs/score_${t}.log 2>&1
        runs=(); for s in $(seeds $t); do for k in esc 4; do runs+=("$R/outputs/astshort_${t}_s$s/$k"); done; done
        nice -n 10 $PY $H/score_eff.py --runs "${runs[@]}" --emb-dir $R/emb_ast_short/$(emb $t) --out $R/scores_ast_short/$t \
          --dim 768 --configs fp32 w8g64 w4g128 n4 w3g64 w4 >> $R/logs/score_short_${t}.log 2>&1
        if [ $t = esc5off ]; then
          runs=(); for s in $(seeds $t); do for k in esc 4; do runs+=("$R/outputs/astoff1024_${t}_s$s/$k"); done; done
          nice -n 10 $PY $H/score_eff.py --runs "${runs[@]}" --emb-dir $S2/emb/esc50 --out $R/scores_ast1024/$t \
            --dim 768 --configs '*' >> $R/logs/score_ast1024_${t}.log 2>&1
        fi ;;
      analyze)
        nice -n 10 $PY $H/analyze.py --task $t --eff-scores $R/scores/$t --ast-scores $(asts $t) \
          --eff-emb $R/emb/$(emb $t) --ast-emb $S2/emb/$(emb $t) --ast-short-scores $R/scores_ast_short/$t \
          --ast-short-emb $R/emb_ast_short/$(emb $t) --audio-stats $I7/audio_stats/$(stats $t) --out $R/analysis/$t \
          > $R/logs/analyze_${t}.log 2>&1 ;;
    esac
  done
  if [ $step = summary ]; then
    nice -n 10 $PY $H/summary.py --analysis $R/analysis --stage1 $S1/analysis --out $R/analysis > $R/logs/summary.log 2>&1
  fi
done
