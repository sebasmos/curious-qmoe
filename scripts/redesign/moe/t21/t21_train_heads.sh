#!/bin/bash
# T21 FP32 heads (`esc`), protocol 3 recipe (scripts/benchmark.py, config esc50.yaml values), CPU, nice 10, 2 threads,
# one job at a time. Tags <d>_<design>_<tier>_s<seed>.   R=... D=d1 bash t21_train_heads.sh [design...]
: "${R:?}"; D=${D:-d1}
W=$HOME/Documents/qmoe-work/t21; PY=${PY:-python}
SEEDS=${SEEDS:-"0 1 2 3 4"}
for design in ${@:-pool official}; do for tier in ${TIERS:-mn10 ast}; do for s in $SEEDS; do
  tag=${D}_${design}_${tier}_s$s
  [ -f $R/outputs/$tag/esc/summary.json ] && continue
  case $design in
    pool) cv="experiment.cross_validation.n_splits=5" ;;
    official) cv="experiment.cross_validation.n_splits=2 +experiment.cross_validation.only_folds=[1]" ;;
  esac
  cd $W && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W nice -n 10 $PY scripts/benchmark.py \
    --config-path $W/config --config-name esc50 experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$s \
    experiment.datasets.esc.csv=$R/csv/$D/heads/${design}_${tier}.csv experiment.datasets.esc.label_col=class_id \
    "experiment.models_to_run=['esc']" experiment.cross_validation.fold_column=folder $cv \
    experiment.cross_validation.inner_split_groups=recording experiment.cross_validation.recording_token=0 \
    experiment.output_root=$R/outputs hydra.run.dir=$R/outputs/$tag experiment.metadata.tag=$tag \
    > $R/logs/train_$tag.log 2>&1
  echo "$tag exit $? $(date '+%F %T')" >> $R/logs/train_heads.done
done; done; done
echo "$(date '+%F %T') heads $* done" >> $R/logs/train_heads.done
