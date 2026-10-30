#!/bin/bash
# T23 scores (ANALYSIS_PLAN.md C4), CPU nice 10, 2 threads. Refuses to run unless inputs/alignment.json has all_ok true (C7).
R=$(cat ~/Documents/qmoe-work/t23-R.txt); W=$HOME/Documents/qmoe-work/t23; PY=${PY:-python}
$PY -c "import json,sys; sys.exit(0 if json.load(open('$R/inputs/alignment.json'))['all_ok'] else 1)" || { echo "alignment not all_ok, refusing to score"; exit 1; }
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=$W/scripts/redesign/idea6/stage2/score_eff.py
runs=""; for s in 0 1 2 3 4; do runs="$runs $R/outputs/wg_esc50_s$s/esc"; done
nice -n 10 $PY $S --runs $runs --emb-dir $R/emb/esc50 --out $R/scores/esc50 --dim 2048 --configs wg > $R/logs/score_esc50.log 2>&1 || exit 1
runs=""; for s in 0 1 2; do runs="$runs $R/outputs/wg_urban_s$s/esc"; done
nice -n 10 $PY $S --runs $runs --emb-dir $R/emb/urban --out $R/scores/urban --dim 2048 --configs wg > $R/logs/score_urban.log 2>&1 || exit 1
runs=""; for s in 0 1 2; do runs="$runs $R/outputs/wg_tau_s$s/esc"; done
nice -n 10 $PY $W/scripts/redesign/idea6/optionB/score_optb.py --runs $runs --emb-dir $R/emb/tau --configs wg_fp32 wg \
  --meta $HOME/Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv --out $R/scores/tau > $R/logs/score_tau.log 2>&1 || exit 1
echo "scores done $(date '+%F %T')"
