#!/bin/bash
# lockrun.sh NAME CMD... : take the GPU lock (mkdir is atomic), write NAME into it, run CMD, release (even on failure or signal). Retries every 60 s.
LOCK=$HOME/Documents/qmoe-work/.mps-lock
NAME=$1; shift
until mkdir "$LOCK" 2>/dev/null; do echo "$(date '+%F %T') lock held by $(cat $LOCK/owner 2>/dev/null), waiting" >&2; sleep 60; done
echo "$NAME $$ $(date '+%F %T')" > "$LOCK/owner"
trap 'rm -f "$LOCK/owner"; rmdir "$LOCK"' EXIT INT TERM HUP
"$@"
