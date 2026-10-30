#!/usr/bin/env bash
# T15/T18 heads on the clean FP32-head recipe (protocol 3, kind esc): one scripts/benchmark.py run per (view, dataset, seed), CPU, nice 10, 2 threads.
#   R=results dir  bash t15_heads.sh P "view:dataset:seed[:onlyfold]" ...      (dataset urban|esc50|tau)
# Training CSVs are built from the clean FP32-backbone embeddings R/emb/<dataset>/<view>_fp32__clean.npy (repo convention: heads on FP32 embeddings, applied to W8A8 ones) by stage 2's make_csv.py (TAU: option B's make_train_csv.py, device-A rows).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${R:?}"
P=${1:-3}; shift
PY=${PY:-python}
DS=$HOME/Documents/DATASETS
mkdir -p "$R/outputs" "$R/logs" "$R/csv"
job() {
  IFS=: read -r view ds seed only <<< "$1"
  local tag="${view}_${ds}_s${seed}" csv="$R/csv/${ds}_${view}.csv" extra="" cfg=urban8k meta
  [ -f "$R/outputs/$tag/esc/summary.json" ] && return
  while [ -d "$csv.lock" ]; do sleep 5; done
  case $ds in
    urban) meta=$DS/Urban8k/ast_768/urbansound8k.csv ;;
    esc50) meta=$DS/ESC-50/ast_768/esc-50.csv; cfg=esc50; extra="experiment.datasets.esc.label_col=label" ;;
    tau)   meta=$DS/TAU2020Mobile/ast_768/tau2020mobile.csv; extra="experiment.cross_validation.n_splits=5 experiment.cross_validation.recording_token=2" ;;
  esac
  if [ ! -f "$csv" ]; then
    mkdir "$csv.lock" 2>/dev/null && {
      if [ $ds = tau ]; then
        PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/redesign/idea6/optionB/make_train_csv.py" --emb "$R/emb/$ds/${view}_fp32__clean.npy" --meta "$meta" --out "$csv"
      else
        PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/redesign/idea6/stage2/make_csv.py" --emb "$R/emb/$ds/${view}_fp32__clean.npy" --meta "$meta" --out "$csv"
      fi > "$R/logs/csv_${ds}_${view}.log" 2>&1; rmdir "$csv.lock"; }
    until [ -f "$csv" ] && [ ! -d "$csv.lock" ]; do sleep 5; done
  fi
  local fold_arg=""
  [ -n "${only:-}" ] && fold_arg="+experiment.cross_validation.only_folds=[$only]"
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name $cfg experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$csv" $extra $fold_arg \
    "experiment.models_to_run=['esc']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}.log" 2>&1
  echo "$tag exit $? $(date '+%F %T')" >> "$R/logs/train_heads.done"
}
export -f job
export ROOT R PY DS
printf '%s\n' "$@" | xargs -P "$P" -I{} bash -c 'job {}'
