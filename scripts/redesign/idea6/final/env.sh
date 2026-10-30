# Shared paths for the idea 6 final round (source this file; R must be set to the results folder).
W=${W:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}  # repository root
PY=${PY:-python}
E=$HOME/Documents/DATASETS
Q=$HOME/Documents/qmoe-results
WT=$HOME/Documents/qmoe-weights/efficientat_release-v0.0.1
S2=$Q/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba
S3B=$Q/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db
S3C=$Q/2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1
UNP=$Q/2026-10-23_1918_redesign-idea6-stage3c-unpadded_code-cbf6be6
I5=$Q/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38
I7=$Q/2026-10-22_1105_redesign-idea7-stage1_code-e2af723
ESC_CSV=$E/ESC-50/ast_768/esc-50.csv
URB_CSV=$E/Urban8k/ast_768/urbansound8k.csv
ESC_AUDIO=$E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio
URB_AUDIO=$E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio
NEW9="reverb:0.3 reverb:0.6 reverb:1.2 clip:0.5 clip:0.25 clip:0.1 lowpass:4000 lowpass:2000 lowpass:1000"
export PYTHONPATH=$W OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
