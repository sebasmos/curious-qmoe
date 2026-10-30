#!/usr/bin/env bash
# Idea 6 Stage 3c follow-up (exact-length UrbanSound8K): the 8-bit, 4-bit and BitNet heads of mn10 trained on the
# exact-length clean FP32 embeddings (protocol 3 recipe, scripts/benchmark.py with Stage 3b's flags), seeds 0-2.
# The FP32 heads already exist (Stage 3c post hoc, outputs_unp/mn10unp_urban_s<seed>/esc). CPU, nice 10, 2 threads,
# P jobs at a time. Then every head kind is scored on the exact-length FP32 and W8A8 embedding files.
#   R=... R3C=... bash scripts/redesign/idea6/stage3c/train_unp_heads.sh [P]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?set R to the results folder}" "${R3C:?set R3C to the Stage 3c results folder}"
P=${1:-3}
PY=${PY:-$(command -v python)}
mkdir -p "$R/outputs" "$R/logs"
job() {
  local seed=$1 k=$2 tag="mn10unp_urban_s$1" csv=$R/csv/urbansound8k_mn10unp_fp32.csv
  [ -f "$R/outputs/$tag/$k/summary.json" ] && return
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name urban8k experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$csv" \
    "experiment.models_to_run=['$k']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}_$k.log" 2>&1
  echo "$tag $k exit $? $(date '+%F %T')" >> "$R/logs/train_heads.done"
}
export -f job
export ROOT R PY
echo "start $(date '+%F %T')" >> "$R/logs/train_heads.done"
for s in 0 1 2; do for k in bitnet 4 8; do echo "$s $k"; done; done | xargs -P "$P" -L 1 bash -c 'job $0 $1'
runs=""
for s in 0 1 2; do
  runs="$runs $R3C/outputs_unp/mn10unp_urban_s$s/esc"
  for k in bitnet 4 8; do runs="$runs $R/outputs/mn10unp_urban_s$s/$k"; done
done
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/redesign/idea6/stage2/score_eff.py \
  --runs $runs --emb-dir $R3C/emb_cnn/urban8k_unpadded --out $R/scores_unpr/urban --dim 960 --configs mn10_fp32 mn10_w8a8 \
  > "$R/logs/score_unpr_urban.log" 2>&1
echo "score exit $? $(date '+%F %T')" >> "$R/logs/train_heads.done"
