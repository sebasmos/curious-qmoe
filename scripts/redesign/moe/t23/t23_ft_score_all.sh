#!/bin/bash
# T23 full run, step 3 (amendment 1): per task, once every head of the task exists, the alignment gate (pre), then the condition passes on the
# GPU queue (t23_ft_score.py stops itself before a condition that could pass 75 minutes, exit 3, and is queued again; finished conditions skip).
R=$(cat ~/Documents/qmoe-work/t23-R.txt); W=$HOME/Documents/qmoe-work/t23; PY=${PY:-python}
G=$HOME/Documents/qmoe-work/fair_gpu_run.sh; export WAIT_MAX=999999; S=$W/scripts/redesign/moe/t23
NOISE="bg:20 bg:10 bg:5 bg:0 pink:20 pink:10 pink:5 pink:0 white:20 white:10 white:5 white:0"
CORR="reverb:0.3 reverb:0.6 reverb:1.2 clip:0.5 clip:0.25 clip:0.1 lowpass:4000 lowpass:2000 lowpass:1000"
for ds in "$@"; do
  case $ds in esc50) nf=5; conds="clean $NOISE $CORR";; urban) nf=10; conds="clean $NOISE $CORR";; tau) nf=5; conds="clean";; esac
  until [ $(ls $R/outputs_ft/wg_${ds}_s*_f*/esc/scaler.pkl 2>/dev/null | wc -l) -ge $((3 * nf)) ]; do sleep 120; done
  cd $W && OMP_NUM_THREADS=2 nice -n 10 $PY $S/t23_align.py --r $R --stage pre --tasks $ds > $R/logs/align_pre_$ds.log 2>&1
  rc=$?; echo "$(date '+%F %T') align pre $ds exit $rc" >> $R/logs/score_all.done; [ $rc -eq 0 ] || exit 1
  for try in 1 2 3 4 5 6; do
    free=$(df -g ~ | tail -1 | awk '{print $4}'); [ "$free" -lt 25 ] && { echo "$(date '+%F %T') disk $free GB, stop" >> $R/logs/score_all.done; exit 1; }
    cd $HOME/Documents/qmoe-work/t23-scratch && $G claude-t23 nice -n 5 $PY -I $S/t23_ft_score.py --r $R --task $ds --conditions $conds --device mps --max-minutes 75 >> $R/logs/score_$ds.log 2>&1
    rc=$?; echo "$(date '+%F %T') score $ds try $try exit $rc" >> $R/logs/score_all.done
    [ $rc -eq 3 ] || break
  done
  [ $rc -eq 0 ] || exit 1
done
echo "$(date '+%F %T') scoring done for $*" >> $R/logs/score_all.done
