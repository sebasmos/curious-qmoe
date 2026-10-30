#!/bin/bash
# Audit: regenerate the ESC-50 log-mel cache (clean + the nine field conditions) that ort_int8.py reads; the stage 3c copy was deleted.
W=/Users/cajas.sebastian/Desktop/repositories/curious-qmoe-redesign-idea6-stage3c
PY=${PY:-python}
E=$HOME/Documents/DATASETS
cd $W; export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 $PY scripts/redesign/idea6/stage3c/ort_mels.py --csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out /Users/cajas.sebastian/Documents/qmoe-results/2026-10-23_1910_redesign-idea6-stage3c-audit_code-78992e2/cache_mel/esc50 --conditions clean bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0
echo "mels exit $?"
