#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Scoring of the ESC-50 tasks on the files present at 13:35 (score_eff.py skips files already scored, so the final
# run_all.sh pass only adds the rest).
export OMP_NUM_THREADS=2 PYTHONPATH=$W
PY=${PY:-python}
H=$W/scripts/redesign/idea6/stage2
for t in esc50 esc5 esc5off; do
  runs=(); for s in 0 1 2 3 4; do for k in esc 8 4 bitnet; do runs+=("$RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/outputs/effp_${t}_s$s/$k"); done; done
  nice -n 10 $PY $H/score_eff.py --runs "${runs[@]}" --emb-dir $RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/emb/esc50 --out $RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/scores/$t >> $RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/logs/score_${t}.log 2>&1
  runs=(); for s in 0 1 2 3 4; do for k in esc 4; do runs+=("$RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/outputs/astshort_${t}_s$s/$k"); done; done
  nice -n 10 $PY $H/score_eff.py --runs "${runs[@]}" --emb-dir $RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/emb_ast_short/esc50 --out $RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/scores_ast_short/$t --dim 768 --configs fp32 w8g64 w4g128 n4 w3g64 w4 >> $RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/logs/score_short_${t}.log 2>&1
done
runs=(); for s in 0 1 2 3 4; do for k in esc 4; do runs+=("$RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/outputs/astoff1024_esc5off_s$s/$k"); done; done
nice -n 10 $PY $H/score_eff.py --runs "${runs[@]}" --emb-dir $RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/emb/esc50 --out $RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/scores_ast1024/esc5off --dim 768 --configs '*' >> $RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/logs/score_ast1024_esc5off.log 2>&1
echo "done $(date)" >> $RESULTS/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/logs/score_esc_partial.log
