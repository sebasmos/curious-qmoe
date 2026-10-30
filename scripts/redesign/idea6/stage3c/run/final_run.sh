#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c: analysis of the four tasks (two at a time), then the summary and the manifest.
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
S3B=${S3B:?set S3B, see REPRODUCE_PAPER.md}
S2=${S2:?set S2, see REPRODUCE_PAPER.md}
I5=${I5:?set I5, see REPRODUCE_PAPER.md}
I7=${I7:?set I7, see REPRODUCE_PAPER.md}
PY=${PY:-python}
cd $W; export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
H=scripts/redesign/idea6/stage3c
echo "$(date '+%F %T') start analysis" >> $R/logs/score.done
for pair in "esc50 esc5" "urban esc5off"; do
  for t in $pair; do
    nice -n 10 $PY $H/analyze_3c.py --task $t --r $R --s3b $S3B --s2 $S2 --i5 $I5 --i7 $I7 > $R/logs/analyze_$t.log 2>&1 &
  done; wait
done
for t in esc50 esc5 urban esc5off; do echo "$(date '+%F %T') analyze $t: $(tail -1 $R/logs/analyze_$t.log)" >> $R/logs/score.done; done
nice -n 10 $PY $H/summary_3c.py --r $R --s3b $S3B > $R/logs/summary.log 2>&1
echo "$(date '+%F %T') summary exit $?" >> $R/logs/score.done
