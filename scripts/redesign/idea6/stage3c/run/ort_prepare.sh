#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c: ORT export and label-free calibration-method choice (ESC-50 fold 1) for the MN widths given.
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
S3B=${S3B:?set S3B, see REPRODUCE_PAPER.md}
PYO=${PYO:-python}
cd $W; export PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
for m in "$@"; do
  echo "$(date '+%F %T') start ort export/select $m" >> $R/logs/extract.done
  nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py export --model $m --r $R --s3b $S3B > $R/logs/ort_export_$m.log 2>&1
  nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py select --model $m --r $R --s3b $S3B > $R/logs/ort_select_$m.log 2>&1
  echo "$(date '+%F %T') ort export/select $m exit $?" >> $R/logs/extract.done
done
