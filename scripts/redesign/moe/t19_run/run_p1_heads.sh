#!/bin/bash
# step 19.5 heads: fold 1 only, no test rows scored, two CPU lanes at nice 10
source /Users/cajas.sebastian/Documents/qmoe-results/2026-10-26_1409_moe-t19_code-ae334eb/scripts/env.sh; export PY R
cd $W
for task in esc50 urban; do
  until grep -q "p1 extract $task try [0-9] exit 0" $R/logs/gpu3_p1_extract.log; do sleep 30; done
  case $task in esc50) seeds="0 1 2 3 4";; *) seeds="0 1 2";; esac
  for c in w8a8 n4a4 n4a6 w4a6 w4a4; do for s in $seeds; do echo "$task $c $s"; done; done |     xargs -P 2 -L 1 bash -c 'nice -n 10 $PY scripts/redesign/moe/t19_heads.py --task $0 --config $1 --seed $2 --folds 1 --no-test > $R/logs/p1heads_$0_$1_s$2.log 2>&1; echo "$0 $1 $2 exit $?"'
done
echo "p1 heads done $(date '+%F %T')"
