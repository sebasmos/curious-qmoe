#!/bin/bash
# Own-embedding heads for AST-Q4ch (w4, per channel), ESC-50 50 classes, seeds 1 to 4 (seed 0 exists in the stage 2 folder).
# Same recipe and command as stage2 logs/train_cheaphead.sh; output goes here, nothing in the stage 2 folder is touched.
PY=${PY_BIN:-$(dirname "$(command -v python)")}  # folder holding the qmoe environment's python
W=${W:-$(cd "$(dirname "$0")/../../.." && pwd)}
R=${R:?set R to the idea 5 stage 2 run folder (2026-10-20_2359_redesign-idea5-stage2_code-3c18c38)}
O=${O:-$(cd "$(dirname "$0")" && pwd)/heads_w4}
cd $W
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in "$@"; do
  tag=astesc50headw4_s$s
  nice -n 10 $PY/python scripts/benchmark.py --config-path $W/config --config-name esc50 experiment.datasets.esc.normalization_type=standard experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$s experiment.datasets.esc.csv=$R/csv/esc-50_ast_w4.csv experiment.datasets.esc.label_col=label experiment.cross_validation.inner_split_groups=recording "experiment.models_to_run=['esc']" experiment.metadata.tag=$tag experiment.output_root=$O hydra.run.dir=$O/$tag > $O/train_$tag.log 2>&1
done
