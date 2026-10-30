#!/usr/bin/env bash
# Idea 6 option B: TAU Urban Acoustic Scenes 2020 Mobile, Development dataset (Zenodo record 3819968, DOI
# 10.5281/zenodo.3819968, licence "other-nc": free for non-commercial research use). 16 audio zips, meta and doc:
# 27,364,817,167 bytes. Each file is downloaded with resume, its MD5 is checked against the Zenodo record's, the zip
# is extracted into audio_raw/tau2020mobile_zenodo_3819968/ and then deleted, so the disk holds the extracted wavs
# (about 27 GB) and never both. The MD5 list below is the Zenodo API's (zenodo_3819968.json, kept next to the audio).
#   bash scripts/redesign/idea6/optionB/download_tau2020.sh [DEST] [P]
set -uo pipefail
DEST=${1:-$HOME/Documents/DATASETS/TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968}
P=${2:-3}
BASE=https://zenodo.org/api/records/3819968/files
PFX=TAU-urban-acoustic-scenes-2020-mobile-development
mkdir -p "$DEST/zips" "$DEST/logs"
curl -sL "https://zenodo.org/api/records/3819968" -o "$DEST/zenodo_3819968.json"
cat > "$DEST/md5_zenodo.txt" <<'MD5'
b1e85b8a908d3d6a6ab73268f385d5c8  TAU-urban-acoustic-scenes-2020-mobile-development.audio.1.zip
4310a13cc2943d6ce3f70eba7ba4c784  TAU-urban-acoustic-scenes-2020-mobile-development.audio.2.zip
ed38956c4246abb56190c1e9b602b7b8  TAU-urban-acoustic-scenes-2020-mobile-development.audio.3.zip
97ab8560056b6816808dedc044dcc023  TAU-urban-acoustic-scenes-2020-mobile-development.audio.4.zip
b50f5e0bfed33cd8e52cb3e7f815c6cb  TAU-urban-acoustic-scenes-2020-mobile-development.audio.5.zip
fbf856a3a86fff7520549c899dc94372  TAU-urban-acoustic-scenes-2020-mobile-development.audio.6.zip
0dbffe7b6e45564da649378723284062  TAU-urban-acoustic-scenes-2020-mobile-development.audio.7.zip
bb6f77832bf0bd9f786f965beb251b2e  TAU-urban-acoustic-scenes-2020-mobile-development.audio.8.zip
a65596a5372eab10c78e08a0de797c9e  TAU-urban-acoustic-scenes-2020-mobile-development.audio.9.zip
2ad595819ffa1d56d2de4c7ed43205a6  TAU-urban-acoustic-scenes-2020-mobile-development.audio.10.zip
0ad29f7040a4e6a22cfd639b3a6738e5  TAU-urban-acoustic-scenes-2020-mobile-development.audio.11.zip
e5f4400c6b9697295fab4cf507155a2f  TAU-urban-acoustic-scenes-2020-mobile-development.audio.12.zip
8855ab9f9896422746ab4c5d89d8da2f  TAU-urban-acoustic-scenes-2020-mobile-development.audio.13.zip
092ad744452cd3e7de78f988a3d13020  TAU-urban-acoustic-scenes-2020-mobile-development.audio.14.zip
4b5eb85f6592aebf846088d9df76b420  TAU-urban-acoustic-scenes-2020-mobile-development.audio.15.zip
2e0a89723e58a3836be019e6996ae460  TAU-urban-acoustic-scenes-2020-mobile-development.audio.16.zip
6eae9db553ce48e4ea246e34e50a3cf5  TAU-urban-acoustic-scenes-2020-mobile-development.meta.zip
175f40dc3fec144347abad4d2962b7ae  TAU-urban-acoustic-scenes-2020-mobile-development.doc.zip
MD5
one() {  # file name
  local f=$1 z="$DEST/zips/$1" want got
  [ -f "$DEST/logs/$f.ok" ] && return 0
  want=$(grep " $f\$" "$DEST/md5_zenodo.txt" | cut -d' ' -f1)
  for try in 1 2 3 4 5; do
    curl -sL -C - --retry 5 --retry-delay 10 -o "$z" "$BASE/$f/content" && break
    sleep 30
  done
  got=$(md5 -q "$z")
  if [ "$got" != "$want" ]; then echo "$(date +%T) $f MD5 MISMATCH $got != $want" >> "$DEST/logs/download.log"; return 1; fi
  unzip -q -o "$z" -d "$DEST" && rm -f "$z" && echo "$got  $f  $(date +%T)" > "$DEST/logs/$f.ok"
  echo "$(date +%T) $f ok $(stat -f %z "$DEST/logs/$f.ok" >/dev/null && echo extracted)" >> "$DEST/logs/download.log"
}
export -f one; export DEST BASE
{ echo $PFX.meta.zip; echo $PFX.doc.zip; for i in $(seq 1 16); do echo $PFX.audio.$i.zip; done; } | xargs -P "$P" -I{} bash -c 'one {}'
n=$(ls "$DEST"/logs/*.ok 2>/dev/null | wc -l | tr -d ' ')
echo "$(date +%T) $n of 18 files verified and extracted" >> "$DEST/logs/download.log"
ls "$DEST"/$PFX/audio/*.wav 2>/dev/null | wc -l >> "$DEST/logs/download.log"
