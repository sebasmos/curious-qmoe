#!/usr/bin/env bash
# Idea 6 option C: heads trained on the 3-bit AST tiers' own clean embeddings (w3g64 from Stage 2's emb_ast_short,
# n3g64 and w3g32 from this run), protocol 3 recipe (scripts/benchmark.py with Stage 2's flags for the short AST
# heads), kinds esc and 4, seeds 0-4 on the ESC-50 tasks and 0-2 on UrbanSound8K; tags own_<cfg>_<task>_s<seed>.
# CPU, nice 10, 2 threads, P jobs at once; the esc heads of every tier first, then the 4-bit heads. Each job waits
# for its CSV, which is built as soon as the tier's clean embedding file exists. Then the scoring: every own head
# on its own tier's files (scores_own/), Stage 2's transferred astshort heads on the new tiers' files (scores_tr/).
#   R=... S2=... bash scripts/redesign/idea6/optionC/train_own_heads.sh [P]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
: "${R:?set R to the results folder}" "${S2:?set S2 to the idea 6 Stage 2 folder}"
P=${1:-3}
PY=${PY:-${PY:-python}}
E=$HOME/Documents/DATASETS
mkdir -p "$R/outputs" "$R/logs" "$R/csv"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT"
emb_dir() { case $1 in w3g64) echo $S2/emb_ast_short/$2 ;; *) echo $R/emb_ast_short/$2 ;; esac; }
csv_of() { case $2 in esc50) echo $R/csv/esc-50_ast_$1_short.csv ;; urban8k) echo $R/csv/urbansound8k_ast_$1_short.csv ;; esac; }
make_csv() {  # cfg ds
  local emb=$(emb_dir $1 $2)/$1__clean.npy out=$(csv_of $1 $2) meta
  case $2 in esc50) meta=$E/ESC-50/ast_768/esc-50.csv ;; urban8k) meta=$E/Urban8k/ast_768/urbansound8k.csv ;; esac
  until [ -f "$emb" ] && [ -f "${emb%.npy}.json" ]; do sleep 30; done
  if [ ! -f "$out" ]; then
    nice -n 10 $PY $ROOT/scripts/redesign/idea6/stage2/make_csv.py --emb "$emb" --meta $meta --out "$out.tmp.csv" \
      >> $R/logs/make_csvs.log 2>&1 && mv "$out.tmp.csv" "$out"
  fi
}
job() {  # cfg task seed kind
  local cfg=$1 task=$2 seed=$3 k=$4 tag="own_$1_$2_s$3" ds ycfg csv extra
  case $task in esc50) ds=esc50; ycfg=esc50; extra="experiment.datasets.esc.label_col=label" ;;
                esc5)  ds=esc50; ycfg=esc50; extra="experiment.datasets.esc.label_col=class_id" ;;
                urban) ds=urban8k; ycfg=urban8k; extra="" ;; esac
  csv=$(csv_of $cfg $ds)
  until [ -f "$csv" ]; do sleep 30; done
  [ -f "$R/outputs/$tag/$k/summary.json" ] && return
  cd "$ROOT" && nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$ROOT/config" --config-name $ycfg experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$csv" $extra \
    "experiment.models_to_run=['$k']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}_$k.log" 2>&1
  echo "$tag $k exit $? $(date '+%F %T')" >> "$R/logs/train_heads.done"
}
export -f job csv_of
export ROOT R S2 PY
echo "start $(date '+%F %T')" >> "$R/logs/train_heads.done"
for cfg in w3g64 n3g64 w3g32; do for ds in esc50 urban8k; do make_csv $cfg $ds; done; done &
for k in esc 4; do for cfg in w3g64 n3g64 w3g32; do
  for t in urban esc50 esc5; do case $t in urban) seeds="0 1 2" ;; *) seeds="0 1 2 3 4" ;; esac
    for s in $seeds; do echo "$cfg $t $s $k"; done; done; done; done | xargs -P "$P" -L 1 bash -c 'job $0 $1 $2 $3'
wait
echo "heads done $(date '+%F %T')" >> "$R/logs/train_heads.done"
# scoring: own heads on their tier's files, transferred heads on the new tiers' files
until grep -q "all done" $R/logs/extract.done 2>/dev/null; do sleep 60; done
for t in urban esc50 esc5; do
  case $t in urban) ds=urban8k; seeds="0 1 2" ;; *) ds=esc50; seeds="0 1 2 3 4" ;; esac
  for cfg in w3g64 n3g64 w3g32; do
    runs=""; for s in $seeds; do for k in esc 4; do runs="$runs $R/outputs/own_${cfg}_${t}_s$s/$k"; done; done
    nice -n 10 $PY $ROOT/scripts/redesign/idea6/stage2/score_eff.py --runs $runs --emb-dir $(emb_dir $cfg $ds) \
      --out $R/scores_own/$t --dim 768 --configs $cfg >> $R/logs/score_own_$t.log 2>&1
    echo "score own $cfg $t exit $? $(date '+%F %T')" >> "$R/logs/score.done"
  done
  runs=""; for s in $seeds; do for k in esc 4; do runs="$runs $S2/outputs/astshort_${t}_s$s/$k"; done; done
  nice -n 10 $PY $ROOT/scripts/redesign/idea6/stage2/score_eff.py --runs $runs --emb-dir $R/emb_ast_short/$ds \
    --out $R/scores_tr/$t --dim 768 --configs n3g64 w3g32 >> $R/logs/score_tr_$t.log 2>&1
  echo "score transferred $t exit $? $(date '+%F %T')" >> "$R/logs/score.done"
done
echo "all scored $(date '+%F %T')" >> "$R/logs/score.done"
