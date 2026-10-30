#!/bin/bash
# T21 D2 download (FSD50K audio, Zenodo 4060432), md5 check, merge of the split zips, extraction of the qualifying
# clips only, deletion of the zips. Stops if free space would fall below 25 GB.   R=... bash t21_get_d2.sh
: "${R:?}"
T=$HOME/Documents/qmoe-work/t21/scripts/redesign/moe/t21; PY=${PY:-python}
Z=$HOME/Documents/DATASETS/FSD50K/zips; A=$HOME/Documents/DATASETS/FSD50K/audio_raw; L=$R/logs/download_d2.log
U=https://zenodo.org/api/records/4060432/files
free() { df -g $HOME | tail -1 | awk '{print $4}'; }
need() { [ "$(free)" -ge "$1" ] || { echo "$(date '+%F %T') free $(free) GB < $1 GB needed, stop (disk)" >> $L; exit 7; }; }
get() { [ -f $Z/$1.ok ] && return; need $(( 27 + $2 / 1000000000 * 2 ))
  bash $T/pdownload.sh $U/$1/content $Z/$1 $2 $3 6 >> $L 2>&1 && touch $Z/$1.ok || { echo "$(date '+%F %T') $1 failed" >> $L; exit 1; }; }
echo "$(date '+%F %T') start, free $(free) GB" >> $L
get FSD50K.eval_audio.z01 3221225472 3090670eaeecc013ca1ff84fe4442aeb
get FSD50K.eval_audio.zip 3037675767 6fa47636c3a3ad5c7dfeba99f2637982
get FSD50K.dev_audio.z01 3221225472 faa7cf4cc076fc34a44a479a5ed862a3
get FSD50K.dev_audio.z02 3221225472 8f9b66153e68571164fb1315d00bc7bc
get FSD50K.dev_audio.z03 3221225472 1196ef47d267a993d30fa98af54b7159
get FSD50K.dev_audio.z04 3221225472 d088ac4e11ba53daf9f7574c11cccac9
get FSD50K.dev_audio.z05 3221225472 81356521aa159accd3c35de22da28c7f
get FSD50K.dev_audio.zip 2306663327 c480d119b8f7a7e32fdb58f3ea4d6c5a
echo "$(date '+%F %T') all md5 OK, free $(free) GB" >> $L
for s in eval dev; do
  m=$Z/${s}_merged.zip
  if [ ! -f $Z/$s.extracted ]; then
    sz=$(cat $Z/FSD50K.${s}_audio.z* $Z/FSD50K.${s}_audio.zip 2>/dev/null | wc -c); need $(( 26 + sz / 1000000000 ))
    [ -f $m.ok ] || { (cd $Z && zip -q -s 0 FSD50K.${s}_audio.zip --out $m) >> $L 2>&1 && touch $m.ok || { echo "$(date '+%F %T') merge $s failed" >> $L; exit 2; }; }
    rm -f $Z/FSD50K.${s}_audio.z0* $Z/FSD50K.${s}_audio.zip
    csvf=$R/csv/d2/pool.csv; [ $s = eval ] && csvf=$R/csv/d2/extra.csv
    nice -n 10 $PY -I $T/t21_unzip_subset.py $m $csvf $A/$s >> $L 2>&1 || { echo "$(date '+%F %T') extract $s failed" >> $L; exit 3; }
    touch $Z/$s.extracted; rm -f $m $m.ok
  fi
  echo "$(date '+%F %T') $s extracted, free $(free) GB" >> $L
done
echo "$(date '+%F %T') d2 audio ready: $(ls $A/dev | wc -l) dev, $(ls $A/eval | wc -l) eval" >> $L
