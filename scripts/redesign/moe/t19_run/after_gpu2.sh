#!/bin/bash
# once every GPU job is done and the esc50 and urban heads exist, rescore the files that arrived after the heads were
# trained (w8a8 noise, n4a4 corruptions) with the already trained heads. Two CPU lanes, nice 10.
source "$(dirname "$0")/env.sh"; export PY R
cd "$W"
until grep -q "all jobs done" $R/logs/gpu4_full.log && grep -q "heads urban done" $R/logs/heads_urban.log 2>/dev/null; do sleep 120; done
for t in esc50 esc5; do for s in 0 1 2 3 4; do echo "$t w8a8 $s w8a8"; echo "$t n4a4 $s n4a4"; echo "$t n4a4 $s w8a8"; done; done > $R/logs/rescore_jobs.txt
for s in 0 1 2; do echo "urban w8a8 $s w8a8"; echo "urban n4a4 $s n4a4"; echo "urban n4a4 $s w8a8"; done >> $R/logs/rescore_jobs.txt
xargs -P 2 -L 1 bash -c 'nice -n 10 $PY scripts/redesign/moe/t19_heads.py --task $0 --config $1 --seed $2 --transfer $3 > $R/logs/heads/rescore_${0}_${1}_s${2}_at_$3.log 2>&1; echo "$0 $1 $2 $3 exit $?"' < $R/logs/rescore_jobs.txt
rm -f $R/logs/rescore_jobs.txt
echo "rescore done $(date '+%F %T')"
