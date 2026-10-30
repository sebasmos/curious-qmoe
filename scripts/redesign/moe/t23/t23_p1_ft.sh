#!/bin/bash
# T23 fallback P1 re-check (ANALYSIS_PLAN.md C5): fine-tune fold 1 per task (GPU queue, one process per task and seed), then per-seed protocol 3 heads
# on the fine-tuned FP32 embeddings (fold 1 only, CPU), then the P1 table on the W8A8 fine-tuned embeddings.   bash t23_p1_ft.sh esc50 urban tau
R=$(cat ~/Documents/qmoe-work/t23-R.txt); W=$HOME/Documents/qmoe-work/t23; PY=${PY:-python}
G=$HOME/Documents/qmoe-work/fair_gpu_run.sh; export WAIT_MAX=999999; DS=$HOME/Documents/DATASETS
for ds in "$@"; do for s in 0 1 2; do
  free=$(df -g ~ | tail -1 | awk '{print $4}'); [ "$free" -lt 25 ] && { echo "disk $free GB, stop" >> $R/logs/p1_ft.done; exit 1; }
  [ -f $R/ft/$ds/f1_s$s/wg__clean.npy ] || { cd $HOME/Documents/qmoe-work/t23-scratch && $G claude-t23 nice -n 5 $PY -I $W/scripts/redesign/moe/t23/t23_finetune.py --task $ds --fold 1 --seeds $s --out $R/ft --device mps >> $R/logs/ft_${ds}.log 2>&1; echo "$(date '+%F %T') ft $ds s$s exit $?" >> $R/logs/p1_ft.done; }
done; done
for ds in "$@"; do for s in 0 1 2; do
  csv=$R/csv_ft/${ds}_f1_s$s.csv; mkdir -p $R/csv_ft
  case $ds in urban) meta=$DS/Urban8k/ast_768/urbansound8k.csv; cfg=urban8k; extra="";; esc50) meta=$DS/ESC-50/ast_768/esc-50.csv; cfg=esc50; extra="experiment.datasets.esc.label_col=label";;
    tau) meta=$DS/TAU2020Mobile/ast_768/tau2020mobile.csv; cfg=urban8k; extra="experiment.cross_validation.n_splits=5 experiment.cross_validation.recording_token=2";; esac
  [ -f $csv ] || $PY -c "
import numpy as np, pandas as pd, sys
m = pd.read_csv('$meta'); X = np.load('$R/ft/$ds/f1_s$s/wg_fp32__clean.npy'); n = len(X)
if '$ds' == 'tau' and not (m.device[:n] == 'a').all(): sys.exit('device-A block')
assert np.isfinite(X).all(); df = pd.concat([m.loc[:n-1, ['folder','name','label','category','class_id']].reset_index(drop=True), pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])], axis=1)
df.to_csv('$csv', index=False, float_format='%.8g')"
  tag=wg_${ds}_s$s; [ -f $R/outputs_p1ft/$tag/esc/summary.json ] && continue
  cd $W && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W nice -n 10 $PY scripts/benchmark.py --config-path $W/config --config-name $cfg experiment.datasets.esc.normalization_type=standard \
    experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$s "experiment.datasets.esc.csv=$csv" $extra "+experiment.cross_validation.only_folds=[1]" \
    "experiment.models_to_run=['esc']" experiment.cross_validation.inner_split_groups=recording "experiment.output_root=$R/outputs_p1ft" "hydra.run.dir=$R/outputs_p1ft/$tag" experiment.metadata.tag=$tag \
    > $R/logs/train_p1ft_$tag.log 2>&1; echo "$tag head exit $?" >> $R/logs/p1_ft.done
done; done
cd $W && OMP_NUM_THREADS=2 nice -n 10 $PY scripts/redesign/moe/t23/t23_p1.py --r $R --outs outputs_p1ft --ft $R/ft --tag fallback --seeds 0 1 2 > $R/logs/p1_fallback.log 2>&1; echo "p1 fallback exit $?" >> $R/logs/p1_ft.done
