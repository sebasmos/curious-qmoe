#!/bin/bash
# T23 full run, step 1 (amendment 1): fine-tuned tails for every remaining (fold, seed), three seeds per GPU process, through the shared queue.
# t23_finetune.py stops itself before a seed that could pass 75 minutes (exit 3) and is queued again; finished seeds are skipped.
R=$(cat ~/Documents/qmoe-work/t23-R.txt); W=$HOME/Documents/qmoe-work/t23; PY=${PY:-python}
G=$HOME/Documents/qmoe-work/fair_gpu_run.sh; export WAIT_MAX=999999
for job in esc50:2 esc50:3 esc50:4 esc50:5 urban:2 urban:3 urban:4 urban:5 urban:6 urban:7 urban:8 urban:9 urban:10 tau:2 tau:3 tau:4 tau:5; do
  ds=${job%%:*}; f=${job#*:}
  for try in 1 2 3 4; do
    [ -f $R/ft/$ds/f${f}_s0/train_log.json ] && [ -f $R/ft/$ds/f${f}_s1/train_log.json ] && [ -f $R/ft/$ds/f${f}_s2/train_log.json ] && break
    free=$(df -g ~ | tail -1 | awk '{print $4}'); [ "$free" -lt 25 ] && { echo "$(date '+%F %T') disk $free GB, stop" >> $R/logs/ft_all.done; exit 1; }
    cd $HOME/Documents/qmoe-work/t23-scratch && $G claude-t23 nice -n 5 $PY -I $W/scripts/redesign/moe/t23/t23_finetune.py --task $ds --fold $f --seeds 0 1 2 --out $R/ft --device mps --max-minutes 75 >> $R/logs/ft_${ds}.log 2>&1
    rc=$?; echo "$(date '+%F %T') ft $ds f$f try $try exit $rc" >> $R/logs/ft_all.done
    [ $rc -eq 3 ] || break
  done
done
echo "$(date '+%F %T') all fine-tunes done" >> $R/logs/ft_all.done
