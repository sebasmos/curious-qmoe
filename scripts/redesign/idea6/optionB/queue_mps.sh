#!/usr/bin/env bash
# Idea 6 option B: run the two MPS extractions (CNN first, then the AST) only when fewer than two extraction
# processes of any session are on the MPS device (the machine-wide rule: at most two MPS processes). Polls every
# minute; logs to R/logs/queue_mps.log.
#   R=... nohup bash scripts/redesign/idea6/optionB/queue_mps.sh &
set -u
: "${R:?}"
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
LOG=$R/logs/queue_mps.log
count() { pgrep -f "python.*extract_(ast_short|cnn|stage2|eff|c)\.py" | wc -l | tr -d ' '; }
wait_slot() { while [ "$(count)" -ge 2 ]; do sleep 60; done; }
wait_slot; echo "$(date '+%F %T') cnn starts, $(count) other MPS extraction processes" >> "$LOG"
R=$R bash "$ROOT/scripts/redesign/idea6/optionB/extract_optb.sh" cnn &
sleep 90
wait_slot; echo "$(date '+%F %T') ast starts, $(count) other MPS extraction processes" >> "$LOG"
R=$R bash "$ROOT/scripts/redesign/idea6/optionB/extract_optb.sh" ast &
wait
echo "$(date '+%F %T') both extractions exited" >> "$LOG"
