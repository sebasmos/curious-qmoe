#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Heads for idea 5 stage 2, seeds 3 and 4 of ESC-50 (coordinator update: ESC-50 seeds 0 to 4), protocol 3
# (inner split grouped by recording), trained on the clean FP32-backbone AST CSVs. Two jobs at a time.
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate curious-qmoe
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
R=$RESULTS/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38
job() {
  ds=$1; seed=$2
  case $ds in
    esc5)  cfgname=esc50;   csv=$E/ESC-50/ast_768/esc-50.csv; extra="experiment.datasets.esc.label_col=class_id";;
    esc50) cfgname=esc50;   csv=$E/ESC-50/ast_768/esc-50.csv; extra="experiment.datasets.esc.label_col=label";;
    urban) cfgname=urban8k; csv=$E/Urban8k/ast_768/urbansound8k.csv; extra="";;
  esac
  tag=ast${ds}_s${seed}
  python scripts/benchmark.py --config-path $W/config --config-name $cfgname     experiment.datasets.esc.normalization_type=standard experiment.device=cpu experiment.track_emissions=false     experiment.num_threads=2 experiment.seed=$seed experiment.datasets.esc.csv=$csv $extra     experiment.cross_validation.inner_split_groups=recording     "experiment.models_to_run=['esc','4']" experiment.metadata.tag=$tag     experiment.output_root=$R/outputs hydra.run.dir=$R/outputs/$tag > $R/logs/train_$tag.log 2>&1
  echo "$tag exit $?" >> $R/logs/train_heads.done
}
export -f job
export W R
printf "%s\n" "esc50 3" "esc50 4" "esc5 3" "esc5 4" | xargs -P 2 -L 1 bash -c 'job $0 $1'
