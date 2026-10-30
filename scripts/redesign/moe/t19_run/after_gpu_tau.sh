#!/bin/bash
# TAU heads as soon as the TAU statics (job line 7) are extracted; the SR passes (line 8) are scored once they exist.
source "$(dirname "$0")/env.sh"; export PY R
cd "$W"
JF=$R/scripts/jobs_full.txt
until grep -qxF -- "$(sed -n 7p $JF)" $JF.done 2>/dev/null; do sleep 60; done
bash $R/scripts/run_heads.sh tau > $R/logs/heads_tau.log 2>&1
until grep -qxF -- "$(sed -n 8p $JF)" $JF.done 2>/dev/null; do sleep 60; done
for c in n4a4sr1 n4a4sr2; do for s in 0 1 2; do echo "tau $c $s n4a4"; done; done |
  xargs -P 2 -L 1 bash -c 'nice -n 10 $PY scripts/redesign/moe/t19_heads.py --task $0 --config $1 --seed $2 --transfer $3 > $R/logs/heads/${0}_${1}_s${2}_at_$3.log 2>&1; echo "$0 $1 $2 $3 exit $?"' >> $R/logs/heads_tau.log
echo "tau SR scored $(date '+%F %T')" >> $R/logs/heads_tau.log
