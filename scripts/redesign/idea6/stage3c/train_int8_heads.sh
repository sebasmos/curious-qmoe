#!/usr/bin/env bash
# Idea 6 Stage 3c, P4 way (b): heads retrained per outer fold on fold f's real-int8 clean embeddings (every row of
# emb_int8/<ds>/<model>_int8_f<f>__clean.npy, CSV via Stage 2's make_csv.py), protocol 3 recipe, restricted to fold
# f with cross_validation.only_folds. Jobs "<model> <task> <kind>" from stdin or JOBS; seeds 0-4 (ESC-50), 0-2
# (UrbanSound8K); folds 1-5 or 1-10. Tags <model>r_<task>_s<seed>_f<fold> under R/outputs_int8b.
#   R=... [MAXRUN=n] bash scripts/redesign/idea6/stage3c/train_int8_heads.sh [P] < jobs
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?}"
P=${1:-4}
PY=${PY:-$(command -v python)}
E=$HOME/Documents/DATASETS
S2=$ROOT/scripts/redesign/idea6/stage2
mkdir -p "$R/outputs_int8b" "$R/logs" "$R/csv_int8"
csv_of() {  # model ds fold -> CSV path, built if missing
  local m=$1 ds=$2 f=$3 meta emb out
  case $ds in esc50) meta=$E/ESC-50/ast_768/esc-50.csv; out=$R/csv_int8/esc-50_${m}_int8_f$f.csv ;;
              urban8k) meta=$E/Urban8k/ast_768/urbansound8k.csv; out=$R/csv_int8/urbansound8k_${m}_int8_f$f.csv ;; esac
  emb=$R/emb_int8/$ds/${m}_int8_f${f}__clean.npy
  until [ -f "$emb" ] && [ -f "${emb%.npy}.json" ]; do sleep 30; done
  if [ ! -f "$out" ]; then
    mkdir "$out.lock" 2>/dev/null && { nice -n 10 $PY $S2/make_csv.py --emb "$emb" --meta $meta --out "$out.tmp.csv" > /dev/null && mv "$out.tmp.csv" "$out"; rmdir "$out.lock"; }
    until [ -f "$out" ]; do sleep 10; done
    if [ $ds = esc50 ] && [ ! -f "${out%.csv}_off5.csv" ]; then
      nice -n 10 $PY $S2/derive_off5.py "$out" "${out%.csv}_off5.tmp.csv" > /dev/null && mv "${out%.csv}_off5.tmp.csv" "${out%.csv}_off5.csv"
    fi
  fi
  echo "$out"
}
job() {
  local m=$1 task=$2 k=$3 seed=$4 f=$5 cfg ds csv extra tag="${1}r_$2_s$4_f$5"
  case $task in
    esc50)   cfg=esc50; ds=esc50; extra="experiment.datasets.esc.label_col=label" ;;
    esc5)    cfg=esc50; ds=esc50; extra="experiment.datasets.esc.label_col=class_id" ;;
    esc5off) cfg=esc50; ds=esc50; extra="experiment.datasets.esc.label_col=class_id" ;;
    urban)   cfg=urban8k; ds=urban8k; extra="" ;;
  esac
  [ -f "$R/outputs_int8b/$tag/$k/summary.json" ] && return
  mkdir -p "$R/outputs_int8b/.locks"
  mkdir "$R/outputs_int8b/.locks/${tag}_$k" 2>/dev/null || return
  csv=$(csv_of $m $ds $f)
  [ $task = esc5off ] && csv=${csv%.csv}_off5.csv
  until [ -f "$csv" ]; do sleep 10; done
  while [ "$(pgrep -f 'scripts/benchmark.py' | wc -l)" -ge "${MAXRUN:-99}" ]; do sleep 20; done
  cd "$ROOT" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name $cfg experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$csv" $extra "+experiment.cross_validation.only_folds=[$f]" \
    "experiment.models_to_run=['$k']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs_int8b" "hydra.run.dir=$R/outputs_int8b/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_int8b_${tag}_$k.log" 2>&1
  echo "$tag $k exit $? $(date +%H:%M:%S)" >> "$R/logs/train_int8_heads.done"
}
export -f job csv_of
export ROOT R PY MAXRUN E S2
expand() {  # model task kind -> one line per seed and fold
  local m=$1 t=$2 k=$3 seeds folds
  case $t in urban) seeds="0 1 2"; folds="1 2 3 4 5 6 7 8 9 10" ;; *) seeds="0 1 2 3 4"; folds="1 2 3 4 5" ;; esac
  for s in $seeds; do for f in $folds; do echo "$m $t $k $s $f"; done; done
}
export -f expand
while read -r m t k; do expand $m $t $k; done | xargs -P "$P" -L 1 bash -c 'job $0 $1 $2 $3 $4'
