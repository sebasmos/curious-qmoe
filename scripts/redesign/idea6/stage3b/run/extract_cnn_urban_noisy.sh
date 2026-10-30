#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Idea 6 Stage 3b: UrbanSound8K CNN noisy conditions in two parallel processes (their CPU front end was the bottleneck
# at about 14 clips/s), every background condition in one process so bg_plan.json has one writer. Replaces the
# urban_noisy step of extract_cnn_resume.sh, whose wrapper was stopped while esc50_noisy ran (that python process was
# left running; its exit is recorded below when it ends).
R=$RESULTS/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db
PY=${PY:-python}
E=$E
WT=$QW/efficientat_release-v0.0.1
ESC_PID=$1
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
URB=(--csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv --out-dir $R/emb_cnn/urban8k --pad-to-samples 64587)
run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $R/logs/extract_cnn.done
  nice -n 10 $PY scripts/redesign/idea6/stage3b/extract_cnn.py "$@" --weights-dir $WT --models mn04 mn10 --configs fp32 w8a8 --device mps --batch-size 64 --chunk 256 > $R/logs/extract_cnn_$name.log 2>&1 &
  local pid=$!
  bash $GUARD $pid >> $R/logs/power_guard.log 2>&1 &
  wait $pid
  echo "$(date '+%F %T') $name exit $?" >> $R/logs/extract_cnn.done
}
run urban_noisy_a "${URB[@]}" --conditions bg:10 bg:5 bg:0 bg:20 pink:10 pink:20 &
run urban_noisy_b "${URB[@]}" --conditions pink:5 pink:0 white:10 white:5 white:0 white:20 &
while kill -0 $ESC_PID 2>/dev/null; do sleep 15; done
grep -q "^white__20db: 2000 rows" $R/logs/extract_cnn_esc50_noisy.log && echo "$(date '+%F %T') esc50_noisy ended (process left running after its wrapper was stopped; log complete)" >> $R/logs/extract_cnn.done
wait
echo "$(date '+%F %T') all done" >> $R/logs/extract_cnn.done
