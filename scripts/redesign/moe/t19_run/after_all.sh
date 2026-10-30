#!/bin/bash
# after the descriptive GPU extras are extracted and rescored: rerun the same analysis (no rule changes) so their rows appear, refresh tables
source "$(dirname "$0")/env.sh"; cd "$W"
until grep -q "rescore done" $R/logs/after_gpu2.log 2>/dev/null; do sleep 120; done
nice -n 10 $PY scripts/redesign/moe/t19_check_emb.py > $R/logs/emb_check_final.log 2>&1
nice -n 10 $PY scripts/redesign/moe/t19_analyze.py --cheap n4a4 > $R/logs/analysis_full_3.log 2>&1
cp $R/analysis/statics.csv $R/analysis/cascade_all_points.csv $R/tables/
echo "final analysis rerun with descriptive extras done $(date '+%F %T')" >> $R/logs/analysis_full_3.log
