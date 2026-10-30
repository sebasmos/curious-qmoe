#!/bin/bash
# Check 1 of ANALYSIS_PLAN.md: each extractor copy reproduces the stored clean embeddings of its system on the first
# 64 clips. R=... bash check_extract.sh
: "${R:?}"
source "$(dirname "$0")/env.sh"
cd $W
C=$R/logs/check_extract; mkdir -p $C
nice -n 10 $PY scripts/redesign/idea6/final/extract_cnn.py --csv $ESC_CSV --audio-dir $ESC_AUDIO --out-dir $C/esc_cnn --weights-dir $WT \
  --models mn10 mn20 --configs w8a8 --conditions clean --device mps --batch-size 64 --chunk 256 --limit 64 > $C/esc_cnn.log 2>&1
nice -n 10 $PY scripts/redesign/idea6/final/extract_cnn.py --csv $URB_CSV --audio-dir $URB_AUDIO --out-dir $C/urb_unp --weights-dir $WT \
  --models mn10 --configs w8a8 --conditions clean --device mps --batch-size 64 --chunk 256 --tail-cpu --limit 64 > $C/urb_unp.log 2>&1
nice -n 10 $PY scripts/redesign/idea6/final/extract_cnn.py --csv $URB_CSV --audio-dir $URB_AUDIO --out-dir $C/urb_pad --weights-dir $WT \
  --models mn20 --configs w8a8 --conditions clean --device mps --batch-size 64 --chunk 256 --pad-to-samples 64587 --limit 64 > $C/urb_pad.log 2>&1
nice -n 10 $PY scripts/redesign/idea6/final/extract_ast_short.py --csv $ESC_CSV --audio-dir $ESC_AUDIO --out-dir $C/esc_ast --configs w4g128 \
  --conditions clean --max-length 512 --device mps --batch-size 16 --chunk 256 --limit 64 > $C/esc_ast.log 2>&1
nice -n 10 $PY scripts/redesign/idea6/final/extract_ast_short.py --csv $URB_CSV --audio-dir $URB_AUDIO --out-dir $C/urb_ast --configs w4g128 \
  --conditions clean --max-length 400 --device mps --batch-size 16 --chunk 256 --limit 64 > $C/urb_ast.log 2>&1
$PY - <<PYEOF
import numpy as np, json
Q="$Q"; R="$R"
pairs = {
 "esc mn10_w8a8": (f"{R}/logs/check_extract/esc_cnn/mn10_w8a8__clean.npy", "$S3B/emb_cnn/esc50/mn10_w8a8__clean.npy"),
 "esc mn20_w8a8": (f"{R}/logs/check_extract/esc_cnn/mn20_w8a8__clean.npy", "$S3C/emb_cnn/esc50/mn20_w8a8__clean.npy"),
 "urban mn10_w8a8 exact": (f"{R}/logs/check_extract/urb_unp/mn10_w8a8__clean.npy", "$S3C/emb_cnn/urban8k_unpadded/mn10_w8a8__clean.npy"),
 "urban mn20_w8a8 padded": (f"{R}/logs/check_extract/urb_pad/mn20_w8a8__clean.npy", "$S3C/emb_cnn/urban8k/mn20_w8a8__clean.npy"),
 "esc ast w4g128@512": (f"{R}/logs/check_extract/esc_ast/w4g128__clean.npy", "$S2/emb_ast_short/esc50/w4g128__clean.npy"),
 "urban ast w4g128@400": (f"{R}/logs/check_extract/urb_ast/w4g128__clean.npy", "$S2/emb_ast_short/urban8k/w4g128__clean.npy"),
}
out={}
for k,(a,b) in pairs.items():
    x=np.load(a); y=np.load(b)[:64]
    d=np.abs(x[:64]-y).max(); rel=d/np.abs(y).max()
    out[k]={"max_abs_diff":float(d),"max_abs_value":float(np.abs(y).max()),"relative":float(rel),"cos_min":float(((x[:64]*y).sum(1)/np.linalg.norm(x[:64],axis=1)/np.linalg.norm(y,axis=1)).min())}
    print(k,out[k])
json.dump(out,open(f"{R}/logs/check_extract.json","w"),indent=1)
PYEOF
