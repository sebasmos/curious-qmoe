#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Score every finished head run on every embedding file (existing scores are kept), then the cascade analysis,
# front.py and tables per dataset. Usage: analyze_all.sh esc50|urban|esc5
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate curious-qmoe
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
R=$RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38
ds=$1
case $ds in
  esc50|esc5) emb=$R/emb/esc50; seeds="0 1 2 3 4";;
  urban) emb=$R/emb/urban8k; seeds="0 1 2";;
esac
STATIC="fp32 w8 w6 w5 w4 n4 w8g128 w4g128 n4g128 w8g64 w3g64 n3g64 w3g128 n3g128 w3 n3 w2g128 n2g128"
CHEAP="w4 n4 w4g128 n4g128 w3g64 n3g64"
runs=(); tags=()
for s in $seeds; do
  t=ast${ds}_s$s
  if [ -f $R/outputs/$t/4/summary.json ] && [ -f $R/outputs/$t/esc/summary.json ]; then runs+=($R/outputs/$t/esc $R/outputs/$t/4); tags+=($t); fi
done
python scripts/redesign/idea5/score_stage2.py --runs "${runs[@]}" --emb-dir $emb --out $R/scores/$ds || exit 1
python scripts/redesign/idea5/analyze_stage2.py --scores $R/scores/$ds --emb-dir $emb --runs-root $R/outputs   --tags "${tags[@]}" --dataset $ds --cheap $CHEAP --configs $STATIC --out $R/analysis/$ds || exit 1
python scripts/redesign/idea5/report_stage2.py $R/analysis/$ds
