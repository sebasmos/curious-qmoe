#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c: ORT int8 mn10 on UrbanSound8K as soon as the mel cache exists (the after_urban.sh steps then find
# their outputs done and skip).
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
S3B=${S3B:?set S3B, see REPRODUCE_PAPER.md}
PY=${PY:-python}
PYO=${PYO:-python}
cd $W; export PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
until grep -q "mels urban8k exit 0" $R/logs/extract.done; do sleep 30; done
m=mn10
echo "$(date '+%F %T') start ort build/embed $m urban8k (early)" >> $R/logs/extract.done
nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py build --model $m --ds urban8k --r $R --s3b $S3B > $R/logs/ort_build_${m}_urban8k.log 2>&1
nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py embed --model $m --ds urban8k --r $R --s3b $S3B > $R/logs/ort_embed_${m}_urban8k.log 2>&1
echo "$(date '+%F %T') ort build/embed $m urban8k exit $? (early)" >> $R/logs/extract.done
nice -n 10 $PY scripts/redesign/idea6/stage3c/score_int8.py --mode a --model $m --task urban --s3b $S3B --r $R > $R/logs/score_int8a_${m}_urban.log 2>&1
echo "$(date '+%F %T') score int8a $m urban exit $? (early)" >> $R/logs/extract.done
