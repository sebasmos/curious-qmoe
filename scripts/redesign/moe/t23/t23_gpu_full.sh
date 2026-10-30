#!/bin/bash
# T23 step 5 (only after P1 and P2 pass): W8A8 embeddings under every stored condition (12 noise cells, 9 corruptions) for ESC-50 and UrbanSound8K.
# Each GPU process goes through the shared FIFO queue and stops itself after 75 minutes (exit 3, partial saved); it is then queued again.
R=$(cat ~/Documents/qmoe-work/t23-R.txt); W=$HOME/Documents/qmoe-work/t23; PY=${PY:-python}
G=$HOME/Documents/qmoe-work/fair_gpu_run.sh; X=$W/scripts/redesign/moe/t23/t23_extract.py
export WAIT_MAX=999999
NOISE="bg:20 bg:10 bg:5 bg:0 pink:20 pink:10 pink:5 pink:0 white:20 white:10 white:5 white:0"
CORR="reverb:0.3 reverb:0.6 reverb:1.2 clip:0.5 clip:0.25 clip:0.1 lowpass:4000 lowpass:2000 lowpass:1000"
for job in "esc50|$NOISE" "esc50|$CORR" "urban|$NOISE" "urban|$CORR"; do
  ds=${job%%|*}; conds=${job#*|}; tag=$(echo $conds | cut -c1-5)
  for try in 1 2 3 4 5 6; do
    free=$(df -g ~ | tail -1 | awk '{print $4}'); [ "$free" -lt 25 ] && { echo "$(date '+%F %T') disk $free GB, stop" >> $R/logs/gpu_full.done; exit 1; }
    cd $HOME/Documents/qmoe-work/t23-scratch && $G claude-t23 nice -n 5 $PY -I $X --dataset $ds --conditions $conds --out-dir $R/emb/$ds --device mps --max-minutes 75 >> $R/logs/extract_${ds}_${tag}.log 2>&1
    rc=$?; echo "$(date '+%F %T') extract $ds $tag try $try exit $rc" >> $R/logs/gpu_full.done
    [ $rc -eq 3 ] || break
  done
done
