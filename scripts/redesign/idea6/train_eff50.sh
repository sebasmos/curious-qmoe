#!/usr/bin/env bash
# Idea 6 stage 1: EfficientNet-B3 heads for ESC-50 with its 50 classes (label), the
# cheap tier on the task protocol 3's singles did not cover. Protocol 3 recipe
# (scripts/redesign/singles_v3.sh flags, inner split grouped by recording), kinds
# esc (FP32), 8, 4 and bitnet, seeds 0 to 4, one job per (seed, kind), P at a time,
# CPU only under nice -n 10 with 2 threads each.
#
#   R=/path/results bash scripts/redesign/idea6/train_eff50.sh [P]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${R:?set R to the results folder}"
P=${1:-3}
CSV=${ESC_EFF_CSV:-$HOME/Documents/DATASETS/ESC-50/efficientnet_1536/esc-50.csv}
mkdir -p "$R/outputs" "$R/logs"
job() {
  local seed=$1 k=$2 tag="effesc50_s$1"
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 python scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name esc50 experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$CSV" experiment.datasets.esc.label_col=label \
    "experiment.models_to_run=['$k']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}_$k.log" 2>&1
  echo "$tag $k exit $? $(date +%H:%M:%S)" >> "$R/logs/train_eff50.done"
}
export -f job
export ROOT R CSV
for s in 0 1 2 3 4; do for k in bitnet 4 8 esc; do echo "$s $k"; done; done | xargs -P "$P" -L 1 bash -c 'job $0 $1'
