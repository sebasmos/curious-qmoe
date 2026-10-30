#!/bin/bash
# GPU job runner: each line of the job file is the argument list of t19_extract.py; each runs through the T19 lock wrapper
# (one MPS process at a time, at most --max-minutes ${MAXMIN:-80} per process; MAXMIN=600 from 27 Oct 15:45, GPU no longer shared) and is repeated until it exits 0 (exit 3 = time limit,
# resume). Lines already done are recorded in <jobfile>.done. Usage: run_jobs.sh JOBFILE
source "$(dirname "$0")/env.sh"
JF=$1
touch "$JF.done"
cd "$W"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  grep -qxF -- "$line" "$JF.done" && continue
  for try in $(seq 1 100); do
    "$R/scripts/swapwait.sh"
    echo "$(date '+%F %T') start: $line (try $try)"
    $GPU $PY scripts/redesign/moe/t19_extract.py $line --max-minutes ${MAXMIN:-80}
    rc=$?
    echo "$(date '+%F %T') exit $rc: $line"
    [ $rc = 0 ] && { echo "$line" >> "$JF.done"; break; }
    [ $rc = 3 ] || [ $rc = 99 ] || { echo "job failed with $rc, stopping"; exit 1; }
  done
done < "$JF"
echo "$(date '+%F %T') all jobs done"
