#!/bin/bash
# wait while swap used exceeds 9.5 GB AND memory_pressure reports under 25% free (coordinator rule of 26 Oct, about 21:00;
# swap alone counts stale pages). Checked every 60 s, at most WAITS times.
for i in $(seq 1 ${WAITS:-60}); do
  u=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
  awk "BEGIN{exit !($u < 9728)}" && exit 0
  fp=$(memory_pressure | tail -1 | sed -E 's/.*: ([0-9]+)%.*/\1/')
  [ "${fp:-0}" -ge 25 ] && exit 0
  echo "$(date '+%F %T') swap used ${u}M, memory free ${fp}%, waiting" >&2; sleep 60
done
exit 0
