#!/usr/bin/env bash
# Reviewer extras, tasks B and D: heads trained with the protocol 3 recipe (scripts/benchmark.py, flags of Stage 3b and 3c),
# CPU, nice 10, 2 threads per job, P jobs at a time. Reads the stored embedding CSVs, writes only under $R.
#   B: Q1 and Q2 heads on the clean FP32 embeddings of MN10 and MN20, ESC-50 (50 classes), seeds 0 to 4
#      (the Q4, Q8, FP32 and BitNet heads of the same CSVs already exist in the Stage 3b and 3c folders).
#   D: FP32 head trained on the clean embeddings of AST-Q4 group-128 (1024 frames), ESC-50 (50 classes), seeds 0 to 4.
#   R=... S3B=... S3C=... bash train_heads.sh B|D [P]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?set R to the results folder}"
PART=${1:?B or D}
P=${2:-2}
PY=${PY:-$(command -v python)}
mkdir -p "$R/outputs" "$R/logs"
job() {
  local csv=$1 tag=$2 k=$3 seed=$4
  [ -f "$R/outputs/$tag/$k/summary.json" ] && return
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name esc50 experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$csv" experiment.datasets.esc.label_col=label \
    "experiment.models_to_run=['$k']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}_$k.log" 2>&1
  echo "$tag $k exit $? $(date +%H:%M:%S)" >> "$R/logs/train_heads.done"
}
export -f job
export ROOT R PY
joblist() {
  if [ "$PART" = B ]; then
    for s in 0 1 2 3 4; do for k in 1 2; do
      echo "$S3B/csv/esc-50_mn10_fp32.csv mn10_esc50_s$s $k $s"
      echo "$S3C/csv/esc-50_mn20_fp32.csv mn20_esc50_s$s $k $s"
    done; done
  else
    for s in 0 1 2 3 4; do echo "$R/csv/esc-50_ast_w4g128.csv astesc50headw4g128_s$s esc $s"; done
  fi
}
joblist | xargs -P "$P" -L 1 bash -c 'job $0 $1 $2 $3'
