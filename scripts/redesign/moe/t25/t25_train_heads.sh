#!/bin/bash
# T25 FP32 heads (`esc`), protocol 3 recipe, exactly T21's t21_train_heads.sh call (pool design, 5 outer folds,
# inner split grouped by file, seeds 0..4), one CNN at a time, CPU, nice 10, 2 threads.   R=... bash t25_train_heads.sh
: "${R:?}"; D=${D:-d1}
W=$HOME/Documents/qmoe-work/t25; PY=${PY:-python}
SEEDS=${SEEDS:-"0 1 2 3 4"}
for tier in ${TIERS:-mn04 mn05 mn20 dymn04}; do for s in $SEEDS; do
  tag=${D}_pool_${tier}_s$s
  [ -f $R/outputs/$tag/esc/summary.json ] && continue
  cd $W && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W nice -n 10 $PY scripts/benchmark.py \
    --config-path $W/config --config-name esc50 experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$s \
    experiment.datasets.esc.csv=$R/csv/$D/heads/pool_${tier}.csv experiment.datasets.esc.label_col=class_id \
    "experiment.models_to_run=['esc']" experiment.cross_validation.fold_column=folder experiment.cross_validation.n_splits=5 \
    experiment.cross_validation.inner_split_groups=recording experiment.cross_validation.recording_token=0 \
    experiment.output_root=$R/outputs hydra.run.dir=$R/outputs/$tag experiment.metadata.tag=$tag \
    > $R/logs/train_$tag.log 2>&1
  echo "$tag exit $? $(date '+%F %T')" >> $R/logs/train_heads.done
done; done
echo "$(date '+%F %T') heads done" >> $R/logs/train_heads.done
