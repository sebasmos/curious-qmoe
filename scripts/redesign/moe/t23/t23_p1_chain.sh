#!/bin/bash
# T23 P1 (ANALYSIS_PLAN.md C8): fold-1 heads (CPU) for the tasks whose clean embeddings exist, then the P1 table. Usage: t23_p1_chain.sh esc50 urban tau
R=$(cat ~/Documents/qmoe-work/t23-R.txt); W=$HOME/Documents/qmoe-work/t23; PY=${PY:-python}
jobs=""
for ds in "$@"; do case $ds in esc50) S="0 1 2 3 4";; *) S="0 1 2";; esac; for s in $S; do jobs="$jobs $ds:$s:1"; done; done
R=$R OUTS=outputs_p1 bash $W/scripts/redesign/moe/t23/t23_heads.sh 1 $jobs
cd $W && OMP_NUM_THREADS=2 nice -n 10 $PY scripts/redesign/moe/t23/t23_p1.py --r $R --outs outputs_p1 > $R/logs/p1.log 2>&1; echo "p1 exit $?"
