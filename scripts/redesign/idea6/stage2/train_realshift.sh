#!/usr/bin/env bash
# Idea 6 Stage 2, amendment 2 (E6): FP32 heads on the four shared classes of each dataset, for the EfficientNet
# (rebuilt FP32) and AST (clean FP32, 1024 frames) tiers, protocol 3, seeds 0 to 2. Tags xds_<eff|ast>_<dataset>_s<seed>.
#   R=... [DATASETS="urban8k esc50"] bash scripts/redesign/idea6/stage2/train_realshift.sh [P]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?set R to the results folder}"
P=${1:-4}
PY=${PY:-$(command -v python)}
job() {
  local tier=$1 ds=$2 seed=$3 cfg extra tag="xds_$1_$2_s$3"
  # urban8k.yaml has no label_col key and reads class_id by default
  case $ds in esc50) cfg=esc50; extra="experiment.datasets.esc.label_col=class_id" ;; urban8k) cfg=urban8k; extra="" ;; esac
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name $cfg experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$R/csv/realshift/${tier}_${ds}_shared.csv" $extra \
    "experiment.models_to_run=['esc']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}.log" 2>&1
  echo "$tag exit $? $(date +%H:%M:%S)" >> "$R/logs/train_realshift.done"
}
export -f job
export ROOT R PY
for t in eff ast; do for d in ${DATASETS:-urban8k esc50}; do for s in 0 1 2; do echo "$t $d $s"; done; done; done \
  | xargs -P "$P" -L 1 bash -c 'job $0 $1 $2'
