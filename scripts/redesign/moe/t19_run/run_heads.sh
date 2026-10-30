#!/bin/bash
# step 19.8 for one task: own heads of every static (new and legacy), the w8a8 head transferred to every other new static, the SR passes
# scored by the cheap tier's heads. Two CPU lanes, nice 10. Usage: run_heads.sh TASK (esc50 runs esc50 and esc5)
source "$(dirname "$0")/env.sh"; export PY R
cd "$W"
T=$1
case $T in esc50) tasks="esc50 esc5"; seeds="0 1 2 3 4"; leg="fp32 w8g64 w4g128 n4 w4 w3g64";;
           urban) tasks="urban"; seeds="0 1 2"; leg="fp32 w8g64 w4g128 n4 w4 w3g64";;
           tau) tasks="tau"; seeds="0 1 2"; leg="fp32 w8g64 w4g128 n4";; esac
NEW="w8a8 n4a4 n4a6 w4a6 w4a4 w8a6 w8a4 w6a6 w4a8 n4a8 mp6 mp8"
run() { xargs -P 2 -L 1 bash -c '[ -z "$3" ] && [ -f $R/scores/$0/${1/:/_}/s$2/summary.json ] && [ $(grep -c test_f1_clean $R/scores/$0/${1/:/_}/s$2/summary.json) -ge 5 ] && { echo "$0 $1 $2 skip (done)"; exit 0; }; nice -n 10 $PY scripts/redesign/moe/t19_heads.py --task $0 --config $1 --seed $2 ${3:+--transfer $3} > $R/logs/heads/${0}_${1/:/_}_s${2}${3:+_at_$3}.log 2>&1; echo "$0 $1 $2 $3 exit $?"'; }
mkdir -p $R/logs/heads
for t in $tasks; do for c in $NEW $(for l in $leg; do echo L:$l; done); do for s in $seeds; do echo "$t $c $s"; done; done; done | run
for t in $tasks; do for c in $NEW n4a4sr1 n4a4sr2; do [ $c = w8a8 ] && continue; for s in $seeds; do
  case $c in n4a4sr*) echo "$t $c $s n4a4";; *) echo "$t $c $s w8a8";; esac; done; done; done | run
echo "heads $T done $(date '+%F %T')"
