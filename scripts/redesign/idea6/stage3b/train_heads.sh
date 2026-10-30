#!/usr/bin/env bash
# Idea 6 Stage 3b: every head of the new systems, protocol 3 recipe (scripts/benchmark.py with the flags of Stage 2's
# train_heads.sh: inner split grouped by recording, standard scaling), CPU, nice 10, 2 threads, P jobs at a time.
#   mn10, mn04   kinds esc 8 4 bitnet   clean FP32 EfficientAT embeddings          csv/<ds>_<prefix>_fp32.csv
#   ast256, ast384, astm2  kinds esc 4  clean FP32 AST at 256 / 384 frames, two-window mean (ESC-50 only)
# Tasks esc50 (label), esc5 (class_id, five fixed groups), esc5off (derived CSV, target // 10), urban; seeds 0-4 on the
# ESC-50 tasks, 0-2 on UrbanSound8K. Tags <prefix>_<task>_s<seed>. A job waits until its CSV exists.
#   R=... [JOBS=file] bash scripts/redesign/idea6/stage3b/train_heads.sh [P]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?set R to the results folder}"
P=${1:-5}
PY=${PY:-$(command -v python)}
mkdir -p "$R/outputs" "$R/logs"
job() {
  local prefix=$1 task=$2 seed=$3 k=$4 cfg csv extra tag="$1_$2_s$3"
  case $task in
    esc50)   cfg=esc50;   csv=$R/csv/esc-50_${prefix}_fp32.csv; extra="experiment.datasets.esc.label_col=label" ;;
    esc5)    cfg=esc50;   csv=$R/csv/esc-50_${prefix}_fp32.csv; extra="experiment.datasets.esc.label_col=class_id" ;;
    esc5off) cfg=esc50;   csv=$R/csv/esc-50_${prefix}_fp32_off5.csv; extra="experiment.datasets.esc.label_col=class_id" ;;
    urban)   cfg=urban8k; csv=$R/csv/urbansound8k_${prefix}_fp32.csv; extra="" ;;
  esac
  [ -f "$R/outputs/$tag/$k/summary.json" ] && return
  # a lock per head, so a second launcher never runs a head the first one holds
  mkdir -p "$R/outputs/.locks"
  mkdir "$R/outputs/.locks/${tag}_$k" 2>/dev/null || return
  until [ -f "$csv" ]; do sleep 30; done
  # at most MAXRUN heads on the machine, whichever launcher started them
  while [ "$(pgrep -f 'scripts/benchmark.py' | wc -l)" -ge "${MAXRUN:-99}" ]; do sleep 20; done
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name $cfg experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$csv" $extra \
    "experiment.models_to_run=['$k']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}_$k.log" 2>&1
  echo "$tag $k exit $? $(date +%H:%M:%S)" >> "$R/logs/train_heads.done"
}
export -f job
export ROOT R PY MAXRUN
joblist() {
  local s t k
  # ordered by when each CSV exists: ESC-50 CNN and 256-frame AST first, UrbanSound8K CNN (long jobs) next
  for pre in mn10 mn04; do
    for t in esc50 esc5 esc5off; do for s in 0 1 2 3 4; do for k in bitnet 4 8 esc; do echo "$pre $t $s $k"; done; done; done
  done
  for t in esc50 esc5 esc5off; do for s in 0 1 2 3 4; do for k in 4 esc; do echo "ast256 $t $s $k"; done; done; done
  for pre in mn10 mn04; do for s in 0 1 2; do for k in bitnet 4 8 esc; do echo "$pre urban $s $k"; done; done; done
  for s in 0 1 2; do for k in 4 esc; do echo "ast256 urban $s $k"; done; done
  for t in esc50 esc5 esc5off urban; do
    case $t in urban) seeds="0 1 2" ;; *) seeds="0 1 2 3 4" ;; esac
    for s in $seeds; do for k in 4 esc; do echo "ast384 $t $s $k"; done; done
  done
  for t in esc50 esc5 esc5off; do for s in 0 1 2 3 4; do for k in 4 esc; do echo "astm2 $t $s $k"; done; done; done
}
if [ -n "${JOBS:-}" ]; then cat "$JOBS"; else joblist; fi | xargs -P "$P" -L 1 bash -c 'job $0 $1 $2 $3'
