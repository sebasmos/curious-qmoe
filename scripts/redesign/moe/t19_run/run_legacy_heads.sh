#!/bin/bash
# own heads on the stored legacy statics (no GPU needed), run early while the GPU extraction goes on
source "$(dirname "$0")/env.sh"; export PY R
cd "$W"; mkdir -p $R/logs/heads
{ for t in esc50 esc5; do for l in fp32 w8g64 w4g128 n4 w4 w3g64; do for s in 0 1 2 3 4; do echo "$t L:$l $s"; done; done; done
  for l in fp32 w8g64 w4g128 n4 w4 w3g64; do for s in 0 1 2; do echo "urban L:$l $s"; done; done
  for l in fp32 w8g64 w4g128 n4; do for s in 0 1 2; do echo "tau L:$l $s"; done; done; } |
xargs -P 2 -L 1 bash -c 'nice -n 10 $PY scripts/redesign/moe/t19_heads.py --task $0 --config $1 --seed $2 > $R/logs/heads/${0}_${1/:/_}_s${2}.log 2>&1; echo "$0 $1 $2 exit $?"'
echo "legacy heads done $(date '+%F %T')"
