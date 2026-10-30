#!/bin/bash
: "${R:?}"
source "$(dirname "$0")/env.sh"
cd $W
nice -n 10 $PY scripts/redesign/idea6/final/audio_stats_new.py --csv $ESC_CSV --audio-dir $ESC_AUDIO --i7 $I7 --dataset esc50 --out $R/audio_stats/esc50 > $R/logs/audio_stats_esc50.log 2>&1
echo "$(date '+%F %T') audio_stats esc50 exit $?" >> $R/logs/audio_stats.done
nice -n 10 $PY scripts/redesign/idea6/final/audio_stats_new.py --csv $URB_CSV --audio-dir $URB_AUDIO --i7 $I7 --dataset urban --out $R/audio_stats/urban > $R/logs/audio_stats_urban.log 2>&1
echo "$(date '+%F %T') audio_stats urban exit $?" >> $R/logs/audio_stats.done
