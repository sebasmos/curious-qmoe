#!/bin/bash
# Part B scoring with Stage 2's score_eff.py (unchanged) on the nine new conditions, existing heads only.
# Waits for each extraction to finish. R=... bash score_final.sh esc|urban
: "${R:?}"
source "$(dirname "$0")/env.sh"
cd $W
SC=scripts/redesign/idea6/stage2/score_eff.py
waitfor() { until grep -q "$1 exit 0" $R/logs/extract.done 2>/dev/null; do sleep 30; done; }
runs() { local root=$1 pre=$2 t=$3 seeds=$4; shift 4; for s in $seeds; do for k in "$@"; do echo "$root/outputs/${pre}_${t}_s$s/$k"; done; done; }
score() { local name=$1; shift; nice -n 10 $PY $SC "$@" >> $R/logs/score_$name.log 2>&1; echo "$(date '+%F %T') score $name exit $?" >> $R/logs/score.done; }
case $1 in
  esc)
    waitfor esc_cnn
    score esc_mn20 --runs $(runs $S3C mn20 esc50 "0 1 2 3 4" esc 8 4 bitnet) --emb-dir $R/emb_cnn/esc50 --out $R/scores/esc50 --dim 1920 --configs mn20_w8a8
    score esc_mn10 --runs $(runs $S3B mn10 esc50 "0 1 2 3 4" esc 8 4 bitnet) --emb-dir $R/emb_cnn/esc50 --out $R/scores/esc50 --dim 960 --configs mn10_w8a8
    waitfor esc_ast
    score esc_ast --runs $(runs $S2 astshort esc50 "0 1 2 3 4" esc 4) --emb-dir $R/emb_ast/esc50_512 --out $R/scores/esc50 --dim 768 --configs w4g128 ;;
  urban)
    waitfor urban_cnn_unp
    score urban_mn10 --runs $(runs $S3C mn10unp urban "0 1 2" esc | sed "s#/outputs/#/outputs_unp/#") $(runs $UNP mn10unp urban "0 1 2" 8 4 bitnet) \
      --emb-dir $R/emb_cnn/urban8k_unpadded --out $R/scores/urban --dim 960 --configs mn10_w8a8
    waitfor urban_ast
    score urban_ast --runs $(runs $S2 astshort urban "0 1 2" esc 4) --emb-dir $R/emb_ast/urban8k_400 --out $R/scores/urban --dim 768 --configs w4g128
    waitfor urban_cnn_pad
    score urban_mn20 --runs $(runs $S3C mn20 urban "0 1 2" esc 8 4 bitnet) --emb-dir $R/emb_cnn/urban8k --out $R/scores/urban --dim 1920 --configs mn20_w8a8 ;;
esac
