#!/bin/bash
# T23 code test of t23_ft_score.py on ESC-50 fold 1 (tails and heads of the P1 re-check), three conditions, written to the scratch folder only.
R=$(cat ~/Documents/qmoe-work/t23-R.txt); W=$HOME/Documents/qmoe-work/t23; PY=${PY:-python}
G=$HOME/Documents/qmoe-work/fair_gpu_run.sh; export WAIT_MAX=999999; D=$HOME/Documents/qmoe-work/t23-scratch/dry_score; rm -rf $D; mkdir -p $D
cd $HOME/Documents/qmoe-work/t23-scratch && $G claude-t23 nice -n 5 $PY -I $W/scripts/redesign/moe/t23/t23_ft_score.py --r $R --task esc50 --conditions clean bg:0 reverb:0.3 --dry-out $D > $D/log.txt 2>&1
echo "dry exit $?" >> $D/log.txt
