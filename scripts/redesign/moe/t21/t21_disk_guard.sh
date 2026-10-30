#!/bin/bash
# Disk guard (coordinator, 2026-10-27): if free space reaches 27 GB, delete the rest of the D2 mel cache (the D2 int8
# secondary then covers only the conditions already embedded). Ends when the cache is gone.   R=... bash t21_disk_guard.sh
: "${R:?}"
while [ -d $R/mel_cache/d2 ] && ls $R/mel_cache/d2/*.npz >/dev/null 2>&1; do
  f=$(df -g $HOME | tail -1 | awk '{print $4}')
  if [ "$f" -le 27 ]; then echo "$(date '+%F %T') free $f GB: mel cache deleted, D2 int8 stops" >> $R/logs/d2_chain.done; rm -rf $R/mel_cache/d2; break; fi
  sleep 60
done
