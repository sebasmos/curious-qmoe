#!/bin/bash
# T24 D4 (TUT Acoustic Scenes 2017, Zenodo 400515 development and 1040168 evaluation): download, md5 check against
# Zenodo, unpack the metadata (meta step) or prepare the audio the models read one archive at a time (audio step:
# t24_prep_audio.py, 16 kHz float32 mono mean, first 10.24 s), deleting each archive after it is prepared. Archive
# content is read as data only. Stops if free space would fall below 25 GB.   R=... bash t24_get_d4.sh meta|audio
: "${R:?}"; STEP=${1:?}
T=$HOME/Documents/qmoe-work/t24/scripts/redesign/moe; PY=${PY:-python}
B=$HOME/Documents/DATASETS/TUT2017; Z=$B/zips; L=$R/logs/download_d4.log; mkdir -p $Z $R/logs
free() { df -g $HOME | tail -1 | awk '{print $4}'; }
need() { [ "$(free)" -ge "$1" ] || { echo "$(date '+%F %T') free $(free) GB < $1 GB needed, stop (disk)" >> $L; exit 7; }; }
get() { [ -f $Z/$2.ok ] && return; need $(( 26 + $3 / 1000000000 + 1 ))
  bash $T/t21/pdownload.sh https://zenodo.org/api/records/$1/files/$2/content $Z/$2 $3 $4 6 >> $L 2>&1 && { touch $Z/$2.ok; echo "\"$2\": \"$4\"," >> $B/md5_verified.lines; } || { echo "$(date '+%F %T') $2 failed" >> $L; exit 1; }; }
DEV=TUT-acoustic-scenes-2017-development; EV=TUT-acoustic-scenes-2017-evaluation
echo "$(date '+%F %T') $STEP start, free $(free) GB" >> $L
case $STEP in
  meta)
    get 400515 $DEV.meta.zip 104321 9007fd4772d816590c5db5f5e9568f5d
    get 400515 $DEV.doc.zip 54796 2065495aaf3f1103e795c9899e2af1df
    get 400515 $DEV.error.zip 1432 802c700b021769e52a2c1e3b9c117a1b
    get 1040168 $EV.meta.zip 25517 200eee9493e8044403e1326e3d05cfde
    get 1040168 $EV.doc.zip 54545 53709a07416ea3b617c02fcf67dbeb9c
    mkdir -p $B/meta_x && for f in $DEV.meta $DEV.doc $DEV.error $EV.meta $EV.doc; do (cd $B/meta_x && unzip -oq $Z/$f.zip); done ;;
  audio)
    while read rec name size md5 part; do
      [ -f $Z/$name.prepared ] && continue
      get $rec $name $size $md5
      OMP_NUM_THREADS=2 nice -n 10 $PY -I $T/t24/t24_prep_audio.py $Z/$name $R/csv/d4/all.csv $part $B/audio16k_t24 $B/tmp >> $L 2>&1 \
        || { echo "$(date '+%F %T') prep $name failed" >> $L; exit 3; }
      touch $Z/$name.prepared; rm -f $Z/$name
      echo "$(date '+%F %T') $name prepared, archive deleted, free $(free) GB" >> $L
    done <<LIST
1040168 $EV.audio.1.zip 1071856687 3d6dda4445871e9544e0fefe7d14c7d9 eval
1040168 $EV.audio.2.zip 1073362972 4085ef5fa286f2169074993a4e405953 eval
1040168 $EV.audio.3.zip 1071521152 cac432579e7cf2dff0aec7aaed248956 eval
1040168 $EV.audio.4.zip 382756463 664bf09c3d24bd26c6b587f1d709de36 eval
400515 $DEV.audio.1.zip 1071445248 251325a9afaaad0326ad1c57f57d514a dev
400515 $DEV.audio.2.zip 1073453613 c26861e05147dc319b4250eb103d9d99 dev
400515 $DEV.audio.3.zip 1073077819 a4815775f8a5e629179726ee4cd4f55a dev
400515 $DEV.audio.4.zip 1072822038 1732b03afe8c53ef8bba80ba14766e57 dev
400515 $DEV.audio.5.zip 1072644652 611be754a0c951185c6ae4b7643c19a0 dev
400515 $DEV.audio.6.zip 1072667888 165a201db800d3ea76fce5a9c2bd97d7 dev
400515 $DEV.audio.7.zip 1073417661 c7d79db84264401c0f8680dcc36013ad dev
400515 $DEV.audio.8.zip 1072381222 35043f25123439392338c790494c7a19 dev
400515 $DEV.audio.9.zip 1072087738 0805dcf5d8e6871dc9610182b2efb93a dev
400515 $DEV.audio.10.zip 1046262120 5df83a191295a04e290b125c634e13e7 dev
LIST
    rmdir $B/tmp 2>/dev/null
    (echo "{"; sed '$ s/,$//' $B/md5_verified.lines; echo "}") > $B/md5_verified.json
    echo "$(date '+%F %T') d4 audio ready: $(ls $B/audio16k_t24 | wc -l) files" >> $L ;;
esac
