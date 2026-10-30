#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c: float16 log-mel caches for the ORT int8 study, every condition, both datasets (deleted afterwards).
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
PY=${PY:-python}
E=${E:-$HOME/Documents/DATASETS}
cd $W; export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ALL="clean bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0 bg:20 pink:20 white:20"
ds=$1
echo "$(date '+%F %T') start mels $ds" >> $R/logs/extract.done
if [ $ds = esc50 ]; then
  nice -n 10 $PY scripts/redesign/idea6/stage3c/ort_mels.py --csv $E/ESC-50/ast_768/esc-50.csv --audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-csv $E/Urban8k/ast_768/urbansound8k.csv --out $R/cache_mel/esc50 --conditions $ALL > $R/logs/mels_esc50.log 2>&1
else
  nice -n 10 $PY scripts/redesign/idea6/stage3c/ort_mels.py --csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv --out $R/cache_mel/urban8k --pad-to-samples 64587 --conditions $ALL > $R/logs/mels_urban8k.log 2>&1
fi
echo "$(date '+%F %T') mels $ds exit $?" >> $R/logs/extract.done
