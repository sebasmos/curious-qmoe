#!/bin/bash
# Idea 7 stage 1, every step on CPU (nice 10, 2 threads): audio statistics, deferral analysis per task, statistics.
# Usage: bash scripts/redesign/idea7/run_all.sh OUT
set -e
OUT=$1
W=$(cd "$(dirname "$0")/../../.." && pwd)
P=${PY:-python}
S2=/Users/cajas.sebastian/Documents/qmoe-results/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38
E=/Users/cajas.sebastian/Documents/DATASETS
EA=$E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio
UA=$E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
mkdir -p "$OUT/logs"
cd "$W"
t0=$(date +%s)
nice -n 10 $P scripts/redesign/idea7/audio_stats.py --csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $EA \
  --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --bg-audio-dir $UA --bg-plan $S2/emb/esc50/bg_plan.json \
  --conditions clean bg:20 bg:10 bg:5 bg:0 pink:20 pink:10 pink:5 pink:0 white:20 white:10 white:5 white:0 \
  --out "$OUT/audio_stats/esc50" > "$OUT/logs/audio_stats_esc50.log" 2>&1
nice -n 10 $P scripts/redesign/idea7/audio_stats.py --csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $UA \
  --bg-csv $E/ESC-50/ast_768/esc-50.csv --bg-audio-dir $EA --bg-plan $S2/emb/urban8k/bg_plan.json \
  --conditions clean bg:10 bg:0 --field --out "$OUT/audio_stats/urban" > "$OUT/logs/audio_stats_urban.log" 2>&1
for t in esc50 esc5 urban; do
  a=$t; [ $t = esc5 ] && a=esc50
  nice -n 10 $P scripts/redesign/idea7/defer.py --stage2 $S2 --task $t --audio-stats "$OUT/audio_stats/$a" \
    --out "$OUT/outputs/$t" > "$OUT/logs/defer_$t.log" 2>&1
done
nice -n 10 $P scripts/redesign/idea7/stats.py --out "$OUT/outputs" > "$OUT/logs/stats.log" 2>&1
echo "wall seconds $(( $(date +%s) - t0 ))" | tee "$OUT/logs/timing.txt"
