#!/bin/bash
# T23 step 4 (ANALYSIS_PLAN.md C8): clean FP32 and W8A8 embeddings of all three tasks, one GPU process per task through the shared FIFO queue.
R=$(cat ~/Documents/qmoe-work/t23-R.txt); W=$HOME/Documents/qmoe-work/t23; PY=${PY:-python}
G=$HOME/Documents/qmoe-work/fair_gpu_run.sh; X=$W/scripts/redesign/moe/t23/t23_extract.py
export WAIT_MAX=999999
for ds in esc50 urban tau; do
  free=$(df -g ~ | tail -1 | awk '{print $4}'); [ "$free" -lt 25 ] && { echo "$(date '+%F %T') disk $free GB, stop" >> $R/logs/gpu_clean.done; exit 1; }
  cd $HOME/Documents/qmoe-work/t23-scratch && $G claude-t23 nice -n 5 $PY -I $X --dataset $ds --conditions clean --fp32 --out-dir $R/emb/$ds --device mps > $R/logs/extract_${ds}_clean.log 2>&1
  echo "$(date '+%F %T') extract $ds clean exit $?" >> $R/logs/gpu_clean.done
done
