#!/bin/bash
# T23 amendment 1: protocol 3 heads per (task, fold, seed), CPU nice 10, 2 threads, T15/C4 settings, only_folds=[f]. Runs in sweeps until every
# head exists: each sweep handles the (fold, seed) whose fine-tuned tail is finished (train_log.json). Per run: CSV from the FP32 clean
# embeddings, benchmark.py, t23_headcheck.py (summary.json check, scaler.pkl, check.npz), then the CSV and the FP32 embedding file are deleted
# (disk). Fold 1 reuses the P1 re-check heads (outputs_p1ft, same CSV, same settings).    bash t23_ft_heads.sh
R=$(cat ~/Documents/qmoe-work/t23-R.txt); W=$HOME/Documents/qmoe-work/t23; PY=${PY:-python}
DS=$HOME/Documents/DATASETS; H=$W/scripts/redesign/moe/t23/t23_headcheck.py; mkdir -p $R/outputs_ft $R/csv_ft
NF_esc50=5; NF_urban=10; NF_tau=5
while :; do
  left=0
  for ds in esc50 urban tau; do nf=NF_$ds; for f in $(seq 1 ${!nf}); do for s in 0 1 2; do
    tag=wg_${ds}_s${s}_f$f; run=$R/outputs_ft/$tag/esc; od=$R/ft/$ds/f${f}_s$s; csv=$R/csv_ft/${ds}_f${f}_s$s.csv
    [ -f $run/scaler.pkl ] && continue
    [ -f $od/train_log.json ] || { left=1; continue; }
    case $ds in urban) meta=$DS/Urban8k/ast_768/urbansound8k.csv; cfg=urban8k; extra="";; esc50) meta=$DS/ESC-50/ast_768/esc-50.csv; cfg=esc50; extra="experiment.datasets.esc.label_col=label";;
      tau) meta=$DS/TAU2020Mobile/ast_768/tau2020mobile.csv; cfg=urban8k; extra="experiment.cross_validation.n_splits=5 experiment.cross_validation.recording_token=2";; esac
    [ -f $csv ] || OMP_NUM_THREADS=2 nice -n 10 $PY -c "
import numpy as np, pandas as pd, sys
m = pd.read_csv('$meta'); X = np.load('$od/wg_fp32__clean.npy'); n = len(X)
if '$ds' == 'tau' and not (m.device[:n] == 'a').all(): sys.exit('device-A block')
assert np.isfinite(X).all(); df = pd.concat([m.loc[:n-1, ['folder','name','label','category','class_id']].reset_index(drop=True), pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])], axis=1)
df.to_csv('$csv.tmp', index=False, float_format='%.8g')" && mv $csv.tmp $csv
    if [ $f = 1 ] && [ -f $R/outputs_p1ft/wg_${ds}_s$s/esc/summary.json ] && [ ! -d $R/outputs_ft/$tag ]; then
      cp -R $R/outputs_p1ft/wg_${ds}_s$s $R/outputs_ft/$tag
      grep -q "csv_ft/${ds}_f1_s$s.csv" $run/config.yaml || { echo "$tag: copied config does not name $csv" >> $R/logs/ft_heads.done; exit 1; }
    fi
    [ -f $run/summary.json ] || { cd $W && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$W nice -n 10 $PY scripts/benchmark.py --config-path $W/config --config-name $cfg experiment.datasets.esc.normalization_type=standard \
      experiment.device=cpu experiment.track_emissions=false experiment.num_threads=2 experiment.seed=$s "experiment.datasets.esc.csv=$csv" $extra "+experiment.cross_validation.only_folds=[$f]" \
      "experiment.models_to_run=['esc']" experiment.cross_validation.inner_split_groups=recording "experiment.output_root=$R/outputs_ft" "hydra.run.dir=$R/outputs_ft/$tag" experiment.metadata.tag=$tag \
      > $R/logs/train_ft_$tag.log 2>&1; echo "$(date '+%F %T') $tag head exit $?" >> $R/logs/ft_heads.done; }
    cd $W && OMP_NUM_THREADS=2 PYTHONPATH=$W nice -n 10 $PY $H --run $run --fold $f >> $R/logs/headcheck.log 2>&1; rc=$?
    echo "$(date '+%F %T') $tag check exit $rc" >> $R/logs/ft_heads.done
    [ $rc -eq 0 ] || exit 1
    rm -f $csv; [ $f = 1 ] || rm -f $od/wg_fp32__clean.npy
  done; done; done
  [ $left = 0 ] && break
  sleep 120
done
echo "$(date '+%F %T') all heads done" >> $R/logs/ft_heads.done
