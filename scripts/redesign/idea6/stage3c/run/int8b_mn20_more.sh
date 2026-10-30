#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c: way (b) heads and scoring for the mn20 P3 picks on the five groups (FP32 head) and the official
# categories (4-bit head).
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
PY=${PY:-python}
cd $W; export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 R PY
printf "mn20 esc5 esc\nmn20 esc5off 4\n" > $R/logs/int8b_jobs_mn20_more.txt
MAXRUN=13 bash scripts/redesign/idea6/stage3c/train_int8_heads.sh 6 < $R/logs/int8b_jobs_mn20_more.txt > $R/logs/train_int8_heads_launcher_mn20_more.log 2>&1
nice -n 10 $PY scripts/redesign/idea6/stage3c/score_int8.py --mode b --model mn20 --task esc5 --kinds esc --s3b $R --r $R > $R/logs/score_int8b_mn20_esc5.log 2>&1
echo "$(date '+%F %T') score int8b mn20 esc5 esc exit $?" >> $R/logs/score.done
nice -n 10 $PY scripts/redesign/idea6/stage3c/score_int8.py --mode b --model mn20 --task esc5off --kinds 4 --s3b $R --r $R > $R/logs/score_int8b_mn20_esc5off.log 2>&1
echo "$(date '+%F %T') score int8b mn20 esc5off 4 exit $?" >> $R/logs/score.done
