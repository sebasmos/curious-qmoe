#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c, post hoc (not in the plan): mn10 UrbanSound8K FP32 heads retrained on the exact-length clean FP32
# embeddings (seeds 0-2, protocol 3 recipe), scored on the exact-length files -> scores_unpr/urban. Answers whether the
# 12-point P5a loss is the padded heads failing to transfer or information lost at exact length.
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
PY=${PY:-python}
E=${E:-$HOME/Documents/DATASETS}
cd $W; export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p $R/csv $R/outputs_unp
csv=$R/csv/urbansound8k_mn10unp_fp32.csv
[ -f $csv ] || nice -n 10 $PY scripts/redesign/idea6/stage2/make_csv.py --emb $R/emb_cnn/urban8k_unpadded/mn10_fp32__clean.npy --meta $E/Urban8k/ast_768/urbansound8k.csv --out $csv
echo "$(date '+%F %T') start post-hoc unpadded heads" >> $R/logs/score.done
for s in 0 1 2; do
  tag=mn10unp_urban_s$s
  nice -n 10 $PY scripts/benchmark.py --config-path $W/config --config-name urban8k experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$s \
    "experiment.datasets.esc.csv=$csv" "experiment.models_to_run=['esc']" experiment.cross_validation.inner_split_groups=recording \
    "experiment.output_root=$R/outputs_unp" "hydra.run.dir=$R/outputs_unp/$tag" experiment.metadata.tag=$tag > $R/logs/train_${tag}_esc.log 2>&1 &
done
wait
nice -n 10 $PY scripts/redesign/idea6/stage2/score_eff.py --runs $R/outputs_unp/mn10unp_urban_s0/esc $R/outputs_unp/mn10unp_urban_s1/esc $R/outputs_unp/mn10unp_urban_s2/esc --emb-dir $R/emb_cnn/urban8k_unpadded --out $R/scores_unpr/urban --dim 960 --configs mn10_fp32 mn10_w8a8 > $R/logs/score_unpr_urban.log 2>&1
echo "$(date '+%F %T') post-hoc unpadded heads scored exit $?" >> $R/logs/score.done
