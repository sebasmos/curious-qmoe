#!/usr/bin/env bash
# Idea 6 option C: once every head is scored, analyze the three tasks in the priced view (analysis/<task>) and the
# unpriced view (analysis_unpriced/<task>, S4), then the summary and the manifest. CPU, nice 10, 2 threads, one task
# at a time per view (the two views run side by side).
#   R=... bash scripts/redesign/idea6/optionC/run_analysis.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}"
Q=$HOME/Documents/qmoe-results
S3C=$Q/2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1; N=$Q/2026-10-23_1918_redesign-idea6-stage3c-unpadded_code-cbf6be6
S3B=$Q/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db; S2=$Q/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba
I5=$Q/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38; I7=$Q/2026-10-22_1105_redesign-idea7-stage1_code-e2af723
PY=${PY:-${PY:-python}}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT"
H=$ROOT/scripts/redesign/idea6/optionC
until grep -q "all scored" $R/logs/score.done 2>/dev/null; do sleep 60; done
echo "start $(date '+%F %T')" >> $R/logs/analysis.done
view() {  # price outdir
  for t in esc50 esc5 urban; do
    nice -n 10 $PY $H/analyze_c.py --task $t --r $R --s3c $S3C --unp $N --s3b $S3B --s2 $S2 --i5 $I5 --i7 $I7 \
      --price $1 --out $R/$2/$t > $R/logs/analyze_${2}_$t.log 2>&1
    echo "$2 $t exit $? $(date '+%F %T')" >> $R/logs/analysis.done
  done
}
view 1024 analysis & view 0 analysis_unpriced & wait
nice -n 10 $PY $H/summary_c.py --r $R --unpriced $R/analysis_unpriced > $R/logs/summary.log 2>&1
echo "summary exit $? $(date '+%F %T')" >> $R/logs/analysis.done
nice -n 10 $PY $H/write_manifest.py $R --inputs $S3C $N $S3B $S2 $I5 $I7 > $R/logs/manifest.log 2>&1
echo "manifest exit $? $(date '+%F %T')" >> $R/logs/analysis.done
