#!/usr/bin/env bash
# Idea 6 Stage 2, amendment 1: heads for the short-input AST (512 frames on ESC-50, 400 on UrbanSound8K), trained on
# its clean FP32 embeddings with the protocol 3 recipe of idea 5 Stage 2's AST heads (scripts/benchmark.py, inner
# split grouped by recording), kinds esc (FP32) and 4. Tags astshort_<task>_s<seed>.
#   R=... ESC_CSV=... URBAN_CSV=... [TASKS="urban esc50 esc5"] bash scripts/redesign/idea6/stage2/train_ast_short_heads.sh [P]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?set R to the results folder}"
TASKS=${TASKS:-"urban esc50 esc5"}
P=${1:-4}
PY=${PY:-$(command -v python)}
mkdir -p "$R/outputs" "$R/logs"
job() {
  local task=$1 seed=$2 k=$3 cfg csv extra tag="astshort_$1_s$2"
  case $task in
    esc50) cfg=esc50; csv=$ESC_CSV; extra="experiment.datasets.esc.label_col=label" ;;
    esc5)  cfg=esc50; csv=$ESC_CSV; extra="experiment.datasets.esc.label_col=class_id" ;;
    urban) cfg=urban8k; csv=$URBAN_CSV; extra="" ;;
  esac
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name $cfg experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$csv" $extra \
    "experiment.models_to_run=['$k']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}_$k.log" 2>&1
  echo "$tag $k exit $? $(date +%H:%M:%S)" >> "$R/logs/train_ast_short_heads.done"
}
export -f job
export ROOT R PY ESC_CSV="${ESC_CSV:-}" URBAN_CSV="${URBAN_CSV:-}"
for t in $TASKS; do
  case $t in urban) seeds="0 1 2" ;; *) seeds="0 1 2 3 4" ;; esac
  for s in $seeds; do for k in esc 4; do echo "$t $s $k"; done; done
done | xargs -P "$P" -L 1 bash -c 'job $0 $1 $2'
