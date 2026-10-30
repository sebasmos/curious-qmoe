#!/bin/bash
# start the heads of each task once its GPU jobs (job file lines 1-3 for esc50 and urban, 7-8 for tau) are done
source "$(dirname "$0")/env.sh"
JF=$R/scripts/jobs_full.txt
need() { sed -n "$1p" $JF; }
for spec in "esc50 1 2 3" "urban 4 5 6" "tau 7 8"; do
  set -- $spec; t=$1; shift
  for n in "$@"; do until grep -qxF -- "$(need $n)" $JF.done 2>/dev/null; do sleep 60; done; done
  bash $R/scripts/run_heads.sh $t > $R/logs/heads_$t.log 2>&1
done
