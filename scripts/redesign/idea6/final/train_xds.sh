#!/bin/bash
# Part A (E6) heads: FP32 head (`esc`) on the four shared classes of one dataset, protocol 3 recipe, seeds 0 to 2,
# for mn20, mn10 (exact-length clean FP32 embeddings) and the 512-frame AST. Tags xds_<mn20|mn10|ast512>_<dataset>_s<seed>.
# Waits for the part A extraction, builds the training CSVs, trains P heads at a time (CPU, nice 10, 2 threads).
#   R=... bash train_xds.sh [P]
: "${R:?}"
source "$(dirname "$0")/env.sh"
P=${1:-3}
until grep -q "part A extraction done" $R/logs/extract.done 2>/dev/null; do sleep 30; done
cd $W
nice -n 10 $PY scripts/redesign/idea6/final/realshift_final.py build-csv --r $R --s2 $S2 --s3b $S3B --s3c $S3C > $R/logs/build_csv.log 2>&1 || exit 1
job() {
  local tier=$1 ds=$2 seed=$3 cfg extra tag="xds_$1_$2_s$3"
  case $ds in esc50) cfg=esc50; extra="experiment.datasets.esc.label_col=class_id" ;; urban8k) cfg=urban8k; extra="" ;; esac
  cd "$W" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$W" nice -n 10 "$PY" scripts/benchmark.py \
    --config-path "$W/config" --config-name $cfg experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$seed \
    "experiment.datasets.esc.csv=$R/csv/xds_${tier}_${ds}.csv" $extra \
    "experiment.models_to_run=['esc']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs" "hydra.run.dir=$R/outputs/$tag" experiment.metadata.tag=$tag \
    > "$R/logs/train_${tag}.log" 2>&1
  echo "$tag exit $? $(date +%H:%M:%S)" >> "$R/logs/train_xds.done"
}
export -f job
export W R PY
for t in mn20 mn10 ast512; do for d in urban8k esc50; do for s in 0 1 2; do echo "$t $d $s"; done; done; done \
  | xargs -P "$P" -L 1 bash -c 'job $0 $1 $2'
echo "$(date '+%F %T') train_xds done" >> $R/logs/train_xds.done
