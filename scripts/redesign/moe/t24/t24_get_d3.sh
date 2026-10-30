#!/bin/bash
# T24 D3 (FSDnoisy18k, Zenodo 2529934): download, md5 check against Zenodo, unpack the metadata (meta step) or prepare
# the audio the models read (audio step: t24_prep_audio.py, 16 kHz float32, first 10.24 s), then delete each archive.
# Archive content is read as data only. Stops if free space would fall below 25 GB.
#   R=... bash t24_get_d3.sh meta|audio
: "${R:?}"; STEP=${1:?}
T=$HOME/Documents/qmoe-work/t24/scripts/redesign/moe; PY=${PY:-python}
B=$HOME/Documents/DATASETS/FSDnoisy18k; Z=$B/zips; L=$R/logs/download_d3.log; mkdir -p $Z $R/logs
U=https://zenodo.org/api/records/2529934/files
free() { df -g $HOME | tail -1 | awk '{print $4}'; }
need() { [ "$(free)" -ge "$1" ] || { echo "$(date '+%F %T') free $(free) GB < $1 GB needed, stop (disk)" >> $L; exit 7; }; }
get() { [ -f $Z/$1.ok ] && return; need $(( 26 + $2 / 1000000000 ))
  bash $T/t21/pdownload.sh $U/$1/content $Z/$1 $2 $3 6 >> $L 2>&1 && touch $Z/$1.ok || { echo "$(date '+%F %T') $1 failed" >> $L; exit 1; }; }
echo "$(date '+%F %T') $STEP start, free $(free) GB" >> $L
case $STEP in
  meta)
    get FSDnoisy18k.meta.zip 79343 96e27a4a63b7a2870522ddcedb5d8296
    get FSDnoisy18k.doc.zip 90519 093a1ca185ec341ca4eac14215e7f96b
    mkdir -p $B/meta_x && (cd $B/meta_x && unzip -oq $Z/FSDnoisy18k.meta.zip && unzip -oq $Z/FSDnoisy18k.doc.zip) ;;
  audio)
    for part in test train; do
      [ -f $Z/$part.prepared ] && continue
      if [ $part = test ]; then get FSDnoisy18k.audio_test.zip 304186599 1ac73d70b4ef3f81900d98c261a832de
      else get FSDnoisy18k.audio_train.zip 9238586835 34dc1d34ca44622af5bf439ceb6f0d55; fi
      OMP_NUM_THREADS=2 nice -n 10 $PY -I $T/t24/t24_prep_audio.py $Z/FSDnoisy18k.audio_$part.zip $R/csv/d3/all.csv $part \
        $B/audio16k_t24 $B/tmp >> $L 2>&1 || { echo "$(date '+%F %T') prep $part failed" >> $L; exit 3; }
      touch $Z/$part.prepared; rm -f $Z/FSDnoisy18k.audio_$part.zip
      echo "$(date '+%F %T') $part prepared, archive deleted, free $(free) GB" >> $L
    done
    rmdir $B/tmp 2>/dev/null
    echo "$(date '+%F %T') d3 audio ready: $(ls $B/audio16k_t24 | wc -l) files" >> $L ;;
esac
