#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c orchestration, ESC-50 side: once the ESC-50 extraction ends, resume the mel cache, run the ORT
# export/select for mn10 and mn04, build and embed the per-fold int8 models on ESC-50 once the mels exist, then
# score way (a) on the three ESC-50 tasks.
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
S3B=${S3B:?set S3B, see REPRODUCE_PAPER.md}
PY=${PY:-python}
PYO=${PYO:-python}
cd $W; export PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
until grep -q "esc50 exit" $R/logs/extract.done; do sleep 30; done
pkill -CONT -f ort_mels.py; echo "$(date '+%F %T') mels esc50 resumed" >> $R/logs/extract.done
bash $HERE/ort_prepare.sh mn10 mn04
until grep -q "mels esc50 exit 0" $R/logs/extract.done; do sleep 30; done
for m in mn10 mn04; do
  echo "$(date '+%F %T') start ort build/embed $m esc50" >> $R/logs/extract.done
  nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py build --model $m --ds esc50 --r $R --s3b $S3B > $R/logs/ort_build_${m}_esc50.log 2>&1
  nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py embed --model $m --ds esc50 --r $R --s3b $S3B > $R/logs/ort_embed_${m}_esc50.log 2>&1
  echo "$(date '+%F %T') ort build/embed $m esc50 exit $?" >> $R/logs/extract.done
  for t in esc50 esc5 esc5off; do
    nice -n 10 $PY scripts/redesign/idea6/stage3c/score_int8.py --mode a --model $m --task $t --s3b $S3B --r $R > $R/logs/score_int8a_${m}_${t}.log 2>&1
    echo "$(date '+%F %T') score int8a $m $t exit $?" >> $R/logs/extract.done
  done
done
