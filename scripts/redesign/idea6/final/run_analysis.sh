#!/bin/bash
# Waits for the scores / heads, then runs the analyses. R=... bash run_analysis.sh B|A
: "${R:?}"
source "$(dirname "$0")/env.sh"
cd $W
X=scripts/redesign/idea6/final
case $1 in
  B)
    until grep -q 'score urban_mn20 exit' $R/logs/score.done 2>/dev/null; do sleep 30; done
    for t in esc50 urban; do
      nice -n 10 $PY $X/analyze_corrupt.py --task $t --r $R --s2 $S2 --s3b $S3B --s3c $S3C --unp $UNP --i7 $I7 > $R/logs/analyze_b_$t.log 2>&1
      echo "$(date '+%F %T') analyze_b $t exit $?" >> $R/logs/analysis.done
    done
    nice -n 10 $PY $X/check_repro.py --r $R --s3c $S3C --unp $UNP --s3b $S3B --out $R/logs/check_repro.json > $R/logs/check_repro.log 2>&1
    echo "$(date '+%F %T') check_repro exit $?" >> $R/logs/analysis.done ;;
  A)
    until grep -q 'train_xds done' $R/logs/train_xds.done 2>/dev/null; do sleep 30; done
    nice -n 10 $PY $X/realshift_final.py e6 --r $R --s2 $S2 --s3b $S3B --s3c $S3C --i5 $I5 --i7 $I7 > $R/logs/e6.log 2>&1
    echo "$(date '+%F %T') e6 exit $?" >> $R/logs/analysis.done
    nice -n 10 $PY $X/realshift_final.py e7 --r $R --s2 $S2 --s3b $S3B --s3c $S3C --unp $UNP --i7 $I7 > $R/logs/e7.log 2>&1
    echo "$(date '+%F %T') e7 exit $?" >> $R/logs/analysis.done ;;
esac
