#!/bin/bash
# Parallel ranged download of one Zenodo file, then concatenation and md5 check.
#   pdownload.sh URL OUT SIZE MD5 [PARTS]
set -u
URL=$1; OUT=$2; SIZE=$3; MD5=$4; P=${5:-8}
D=$OUT.parts; mkdir -p "$D"
CH=$(( (SIZE + P - 1) / P ))
pids=()
for i in $(seq 0 $((P - 1))); do
  a=$(( i * CH )); b=$(( a + CH - 1 )); [ $b -ge $SIZE ] && b=$(( SIZE - 1 ))
  want=$(( b - a + 1 ))
  ( for try in $(seq 1 2000); do
      have=$(stat -f %z "$D/p$i" 2>/dev/null || echo 0)
      [ "$have" -ge "$want" ] && break
      [ $try -gt 1 ] && sleep 10; curl -sfL --connect-timeout 30 --speed-limit 2000 --speed-time 45 --max-time 3600 -r $(( a + have ))-$b "$URL" >> "$D/p$i"
    done ) &
  pids+=($!)
done
wait "${pids[@]}"
cat $(for i in $(seq 0 $((P - 1))); do echo "$D/p$i"; done) > "$OUT"
got=$(md5 -q "$OUT")
if [ "$got" = "$MD5" ] && [ "$(stat -f %z "$OUT")" = "$SIZE" ]; then echo "$(date +%T) $OUT md5 OK $got"; rm -rf "$D"; else echo "$(date +%T) $OUT md5 MISMATCH $got (want $MD5)"; exit 1; fi
