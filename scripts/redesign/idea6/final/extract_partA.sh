#!/bin/bash
# Part A extra embeddings (clean only), run after stream 1 of extract_final.sh so that two MPS processes run at most:
#   mn20 FP32 and W8A8 at exact length on UrbanSound8K (tail-cpu), and the AST at 512 frames (FP32 and w4g128) on the
#   shared-class clips of both datasets (the meta CSVs written by realshift_final.py prepare).
#   R=... bash extract_partA.sh
: "${R:?}"
source "$(dirname "$0")/env.sh"
cd $W
X=scripts/redesign/idea6/final
until grep -q "stream 1 done" $R/logs/extract.done 2>/dev/null; do sleep 30; done
run() {
  local name=$1; shift
  echo "$(date '+%F %T') start $name" >> $R/logs/extract.done
  nice -n 10 "$@" > $R/logs/extract_$name.log 2>&1
  echo "$(date '+%F %T') $name exit $?" >> $R/logs/extract.done
}
run a_ast_esc $PY $X/extract_ast_short.py --csv $R/csv/shared_esc50_meta.csv --audio-dir $ESC_AUDIO --out-dir $R/emb_ast/shared512/esc50 \
  --configs fp32 w4g128 --conditions clean --max-length 512 --device mps --batch-size 16 --chunk 256
run a_ast_urban $PY $X/extract_ast_short.py --csv $R/csv/shared_urban8k_meta.csv --audio-dir $URB_AUDIO --out-dir $R/emb_ast/shared512/urban8k \
  --configs fp32 w4g128 --conditions clean --max-length 512 --device mps --batch-size 16 --chunk 256
run a_mn20_urban $PY $X/extract_cnn.py --csv $URB_CSV --audio-dir $URB_AUDIO --out-dir $R/emb_cnn/urban8k_exact_e6 --weights-dir $WT \
  --models mn20 --configs fp32 w8a8 --conditions clean --device mps --batch-size 64 --chunk 256 --tail-cpu
echo "$(date '+%F %T') part A extraction done" >> $R/logs/extract.done
