#!/bin/bash
# Logs swap use and memory pressure once a minute until R/logs/swap_watch.stop exists.
: "${R:?}"
while [ ! -f $R/logs/swap_watch.stop ]; do
  echo "$(date '+%F %T') $(sysctl -n vm.swapusage | tr -s ' ') | free $(memory_pressure | grep 'free percentage' | grep -o '[0-9]*%')" >> $R/logs/swap.log
  sleep 60
done
