#!/bin/bash
# Part B extraction, two streams (at most two MPS processes at once), nice 10, 2 threads each.
#   R=... bash extract_final.sh 1|2
# stream 1: ESC-50 mn10 + mn20 (W8A8), ESC-50 AST-Q4 @512, UrbanSound8K mn20 (padded)
# stream 2: UrbanSound8K mn10 (exact length, --tail-cpu), UrbanSound8K AST-Q4 @400
: "${R:?}"
source "$(dirname "$0")/env.sh"
cd $W
X=scripts/redesign/idea6/final
run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $R/logs/extract.done
  nice -n 10 "$@" > $R/logs/extract_$name.log 2>&1
  echo "$(date '+%F %T') $name exit $?" >> $R/logs/extract.done
}
case $1 in
  1)
    run esc_cnn $PY $X/extract_cnn.py --csv $ESC_CSV --audio-dir $ESC_AUDIO --out-dir $R/emb_cnn/esc50 --weights-dir $WT \
      --models mn10 mn20 --configs w8a8 --conditions $NEW9 --device mps --batch-size 64 --chunk 256
    run esc_ast $PY $X/extract_ast_short.py --csv $ESC_CSV --audio-dir $ESC_AUDIO --out-dir $R/emb_ast/esc50_512 --configs w4g128 \
      --conditions $NEW9 --max-length 512 --device mps --batch-size 16 --chunk 256
    run urban_cnn_pad $PY $X/extract_cnn.py --csv $URB_CSV --audio-dir $URB_AUDIO --out-dir $R/emb_cnn/urban8k --weights-dir $WT \
      --models mn20 --configs w8a8 --conditions $NEW9 --device mps --batch-size 64 --chunk 256 --pad-to-samples 64587 ;;
  2)
    run urban_cnn_unp $PY $X/extract_cnn.py --csv $URB_CSV --audio-dir $URB_AUDIO --out-dir $R/emb_cnn/urban8k_unpadded --weights-dir $WT \
      --models mn10 --configs w8a8 --conditions $NEW9 --device mps --batch-size 64 --chunk 256 --tail-cpu
    run urban_ast $PY $X/extract_ast_short.py --csv $URB_CSV --audio-dir $URB_AUDIO --out-dir $R/emb_ast/urban8k_400 --configs w4g128 \
      --conditions $NEW9 --max-length 400 --device mps --batch-size 16 --chunk 256 ;;
esac
echo "$(date '+%F %T') stream $1 done" >> $R/logs/extract.done
