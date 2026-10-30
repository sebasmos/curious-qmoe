#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
# Idea 6 Stage 3c orchestration, UrbanSound8K side: once urban_a ends, resume urban_unp (P5) from its checkpoint;
# once every padded process has ended, build the mel cache, the per-fold int8 models and embeddings, and score way (a).
R=${R:?set R to the stage 3c results folder}
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
S3B=${S3B:?set S3B, see REPRODUCE_PAPER.md}
PY=${PY:-python}
PYO=${PYO:-python}
E=${E:-$HOME/Documents/DATASETS}
WT=${WT:?set WT to the EfficientAT release v0.0.1 weights folder}
cd $W; export PYTHONPATH=$W PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
URB=(--csv $E/Urban8k/ast_768/urbansound8k.csv --audio-dir $E/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio --bg-audio-dir $E/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --bg-csv $E/ESC-50/ast_768/esc-50.csv)
ALL="clean bg:10 bg:5 bg:0 pink:10 pink:5 pink:0 white:10 white:5 white:0 bg:20 pink:20 white:20"
until grep -q "urban_a exit" $R/logs/extract.done; do sleep 30; done
echo "$(date '+%F %T') start urban_unp2 (resume from checkpoint)" >> $R/logs/extract.done
nice -n 10 $PY scripts/redesign/idea6/stage3c/extract_cnn.py "${URB[@]}" --out-dir $R/emb_cnn/urban8k_unpadded --tail-cpu --models mn10 --conditions $ALL --weights-dir $WT --configs fp32 w8a8 --device mps --batch-size 64 --chunk 256 > $R/logs/extract_urban_unp2.log 2>&1
echo "$(date '+%F %T') urban_unp2 exit $?" >> $R/logs/extract.done
until grep -q "urban_b exit" $R/logs/extract.done; do sleep 30; done
bash $HERE/mels.sh urban8k
for m in mn10; do
  echo "$(date '+%F %T') start ort build/embed $m urban8k" >> $R/logs/extract.done
  nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py build --model $m --ds urban8k --r $R --s3b $S3B > $R/logs/ort_build_${m}_urban8k.log 2>&1
  nice -n 10 $PYO scripts/redesign/idea6/stage3c/ort_int8.py embed --model $m --ds urban8k --r $R --s3b $S3B > $R/logs/ort_embed_${m}_urban8k.log 2>&1
  echo "$(date '+%F %T') ort build/embed $m urban8k exit $?" >> $R/logs/extract.done
  nice -n 10 $PY scripts/redesign/idea6/stage3c/score_int8.py --mode a --model $m --task urban --s3b $S3B --r $R > $R/logs/score_int8a_${m}_urban.log 2>&1
  echo "$(date '+%F %T') score int8a $m urban exit $?" >> $R/logs/extract.done
done
