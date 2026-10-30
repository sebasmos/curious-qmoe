#!/usr/bin/env bash
# Idea 6 option B: heads on the clean FP32 device-A embeddings of TAU 2020 Mobile, protocol 3 recipe
# (scripts/benchmark.py with the Stage 2/3b flags; urban8k config with 5 location-grouped folds and the location id
# as the recording token), CPU, nice 10, 2 threads, P jobs at a time. Runs: mn10 and mn20 with kinds esc 8 4 bitnet,
# the AST (fp32 clean embeddings) with kinds esc 4; seeds 0 to 2. Tags <prefix>_tau_s<seed>.
#   R=... [JOBS=file] bash scripts/redesign/idea6/optionB/train_heads.sh [P]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}"
P=${1:-4}
PY=${PY:-${PY:-python}}
mkdir -p "$R/outputs" "$R/logs"
job() {
  local prefix=$1 seed=$2 k=$3 csv=$R/csv/tau_${1}_fp32.csv tag="$1_tau_s$2"
  [ -f "$R/outputs/$tag/$k/summary.json" ] && return
  mkdir -p "$R/outputs/.locks"
  mkdir "$R/outputs/.locks/${tag}_$k" 2>/dev/null || return
  until [ -f "$csv" ]; do sleep 30; done
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name urban8k experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$csv" experiment.cross_validation.n_splits=5 \
    "experiment.models_to_run=['$k']" experiment.cross_validation.inner_split_groups=recording \
    experiment.cross_validation.recording_token=2 \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}_$k.log" 2>&1
  echo "$tag $k exit $? $(date +%H:%M:%S)" >> "$R/logs/train_heads.done"
}
export -f job
export ROOT R PY
joblist() {
  for s in 0 1 2; do for k in esc 4; do echo "ast $s $k"; done; done
  for pre in mn10 mn20; do for s in 0 1 2; do for k in esc 4 8 bitnet; do echo "$pre $s $k"; done; done; done
}
if [ -n "${JOBS:-}" ]; then cat "$JOBS"; else joblist; fi | xargs -P "$P" -L 1 bash -c 'job $0 $1 $2'
