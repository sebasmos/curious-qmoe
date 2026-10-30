#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c: real-int8 mn20 (the ESC-50/50 P3 pick's backbone) on ESC-50: export, method choice, per-fold builds,
# embeddings, way (a) scoring on the three ESC-50 tasks, way (b) heads for the pick's kind, way (b) scoring.
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
S3B=${S3B:?set S3B, see REPRODUCE_PAPER.md}
PY=${PY:-python}
PYO=${PYO:-python}
cd $W; export PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
m=mn20
echo "$(date '+%F %T') start ort mn20 (ESC-50/50 P3 pick)" >> $R/logs/extract.done
nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py export --model $m --r $R --s3b $R > $R/logs/ort_export_$m.log 2>&1
nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py select --model $m --r $R --s3b $R > $R/logs/ort_select_$m.log 2>&1
nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py build --model $m --ds esc50 --r $R --s3b $R > $R/logs/ort_build_${m}_esc50.log 2>&1
nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py embed --model $m --ds esc50 --r $R --s3b $R > $R/logs/ort_embed_${m}_esc50.log 2>&1
echo "$(date '+%F %T') ort build/embed $m esc50 exit $?" >> $R/logs/extract.done
for t in esc50 esc5 esc5off; do
  nice -n 10 $PY scripts/redesign/idea6/stage3c/score_int8.py --mode a --model $m --task $t --s3b $R --r $R > $R/logs/score_int8a_${m}_${t}.log 2>&1
  echo "$(date '+%F %T') score int8a $m $t exit $?" >> $R/logs/extract.done
done
export R PY
printf "mn20 esc50 4\n" > $R/logs/int8b_jobs_mn20.txt
MAXRUN=13 bash scripts/redesign/idea6/stage3c/train_int8_heads.sh 5 < $R/logs/int8b_jobs_mn20.txt > $R/logs/train_int8_heads_launcher_mn20.log 2>&1
nice -n 10 $PY scripts/redesign/idea6/stage3c/score_int8.py --mode b --model $m --task esc50 --kinds 4 --s3b $R --r $R > $R/logs/score_int8b_${m}_esc50.log 2>&1
echo "$(date '+%F %T') score int8b $m esc50 4 exit $?" >> $R/logs/score.done
