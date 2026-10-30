#!/usr/bin/env bash
# T23 heads (ANALYSIS_PLAN.md C4): protocol 3 `esc` head, one scripts/benchmark.py run per (dataset, seed), CPU, nice 10, 2 threads; T15's t15_heads.sh
# settings. Training CSVs from the clean FP32 embeddings R/emb/<ds>/wg_fp32__clean.npy (stage 2 make_csv.py; TAU: option B make_train_csv.py, device A).
#   R=results OUTS=outputs_p1|outputs bash t23_heads.sh P "dataset:seed[:onlyfold]" ...
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}"; OUTS=${OUTS:-outputs}
P=${1:-2}; shift
PY=${PY:-${PY:-python}}
DS=$HOME/Documents/DATASETS
mkdir -p "$R/$OUTS" "$R/logs" "$R/csv"
job() {
  IFS=: read -r ds seed only <<< "$1"
  local tag="wg_${ds}_s${seed}" csv="$R/csv/${ds}_wg.csv" extra="" cfg=urban8k meta
  [ -f "$R/$OUTS/$tag/esc/summary.json" ] && return
  while [ -d "$csv.lock" ]; do sleep 5; done
  case $ds in
    urban) meta=$DS/Urban8k/ast_768/urbansound8k.csv ;;
    esc50) meta=$DS/ESC-50/ast_768/esc-50.csv; cfg=esc50; extra="experiment.datasets.esc.label_col=label" ;;
    tau)   meta=$DS/TAU2020Mobile/ast_768/tau2020mobile.csv; extra="experiment.cross_validation.n_splits=5 experiment.cross_validation.recording_token=2" ;;
  esac
  if [ ! -f "$csv" ]; then
    mkdir "$csv.lock" 2>/dev/null && {
      if [ $ds = tau ]; then
        PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/redesign/idea6/optionB/make_train_csv.py" --emb "$R/emb/$ds/wg_fp32__clean.npy" --meta "$meta" --out "$csv.tmp.csv"
      else
        PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/redesign/idea6/stage2/make_csv.py" --emb "$R/emb/$ds/wg_fp32__clean.npy" --meta "$meta" --out "$csv.tmp.csv"
      fi > "$R/logs/csv_${ds}_wg.log" 2>&1 && mv "$csv.tmp.csv" "$csv"; rmdir "$csv.lock"; }
    until [ -f "$csv" ] && [ ! -d "$csv.lock" ]; do sleep 5; done
  fi
  local fold_arg=""
  [ -n "${only:-}" ] && fold_arg="+experiment.cross_validation.only_folds=[$only]"
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name $cfg experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$csv" $extra $fold_arg \
    "experiment.models_to_run=['esc']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/$OUTS" "hydra.run.dir=$R/$OUTS/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${OUTS}_${tag}.log" 2>&1
  echo "$tag exit $? $(date '+%F %T')" >> "$R/logs/train_heads_${OUTS}.done"
}
export -f job
export ROOT R PY DS OUTS
printf '%s\n' "$@" | xargs -P "$P" -I{} bash -c 'job {}'
