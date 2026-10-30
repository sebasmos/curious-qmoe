#!/usr/bin/env bash
# Idea 6 Stage 2, amendment 3: every head of the secondary task esc5off (ESC-50, official five categories,
# target // 10) on derived CSVs, protocol 3 recipe, seeds 0 to 4, CPU, nice 10, 2 threads:
#   cheap   effp_esc5off_s<seed>/{esc,8,4,bitnet}   rebuilt FP32 EfficientNet embeddings
#   AST     astoff1024_esc5off_s<seed>/{esc,4}      clean FP32 AST, 1024 frames
#   AST     astshort_esc5off_s<seed>/{esc,4}        clean FP32 AST, 512 frames
#   R=... EFF_CSV=... AST_CSV=... SHORT_CSV=... bash scripts/redesign/idea6/stage2/train_esc5off.sh [P]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}" "${EFF_CSV:?}" "${AST_CSV:?}" "${SHORT_CSV:?}"
P=${1:-3}
PY=${PY:-$(command -v python)}
job() {
  local prefix=$1 seed=$2 k=$3 csv tag="$1_esc5off_s$2"
  case $prefix in effp) csv=$EFF_CSV ;; astoff1024) csv=$AST_CSV ;; astshort) csv=$SHORT_CSV ;; esac
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name esc50 experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$csv" experiment.datasets.esc.label_col=class_id \
    "experiment.models_to_run=['$k']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}_$k.log" 2>&1
  echo "$tag $k exit $? $(date +%H:%M:%S)" >> "$R/logs/train_esc5off.done"
}
export -f job
export ROOT R PY EFF_CSV AST_CSV SHORT_CSV
for s in 0 1 2 3 4; do
  for k in esc 8 4 bitnet; do echo "effp $s $k"; done
  for k in esc 4; do echo "astoff1024 $s $k"; echo "astshort $s $k"; done
done | xargs -P "$P" -L 1 bash -c 'job $0 $1 $2'
