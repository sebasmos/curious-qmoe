#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Heads trained on the W4 (per channel, 8-bit activations) backbone's clean ESC-50 embeddings, seed 0, protocol 3:
# ESC-50 5 classes (the brief) and 50 classes (extra). FP32 head only.
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate curious-qmoe
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
R=$RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38
for spec in "esc5 class_id" "esc50 label"; do
  set -- $spec
  tag=ast$1headw4_s0
  python scripts/benchmark.py --config-path $W/config --config-name esc50     experiment.datasets.esc.normalization_type=standard experiment.device=cpu experiment.track_emissions=false     experiment.num_threads=2 experiment.seed=0 experiment.datasets.esc.csv=$R/csv/esc-50_ast_w4.csv     experiment.datasets.esc.label_col=$2 experiment.cross_validation.inner_split_groups=recording     "experiment.models_to_run=['esc']" experiment.metadata.tag=$tag     experiment.output_root=$R/outputs hydra.run.dir=$R/outputs/$tag > $R/logs/train_$tag.log 2>&1 &
done
wait
echo done >> $R/logs/train_cheaphead.done
