#!/bin/bash
# Downloads the public datasets and checks every archive against its published MD5, into the layout the drivers expect
# (E defaults to ~/Documents/DATASETS). Archives are kept after extraction unless KEEP=0.
#   bash scripts/download_datasets.sh esc50|urban8k|fsdk2018|fsd50k-meta|weights|all
# FSD50K audio (30 GB, only the clips of the single-label subset are extracted) and TAU 2020 Mobile (27 GB) have their own
# scripts, which also check MD5s: scripts/redesign/moe/t21/t21_get_d2.sh and scripts/redesign/idea6/optionB/download_tau2020.sh.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
E=${E:-$HOME/Documents/DATASETS}
QW=${QW:-$HOME/Documents/qmoe-weights}
KEEP=${KEEP:-1}
md5of() { if command -v md5 >/dev/null; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }
get() {  # url, output file, md5
  local url=$1 out=$2 want=$3
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(md5of "$out")" = "$want" ]; then echo "ok (cached) $out"; return; fi
  curl -fL --retry 5 --retry-delay 10 -C - -o "$out" "$url"
  local got; got=$(md5of "$out")
  [ "$got" = "$want" ] || { echo "MD5 MISMATCH $out: $got, expected $want" >&2; exit 1; }
  echo "ok $out $got"
}
Z=https://zenodo.org/api/records

esc50() {  # ESC-50, GitHub archive of the repository at commit 33c8ce9e (the zip comment carries the commit)
  local d=$E/ESC-50/audio_raw/esc50_master_zip
  mkdir -p "$d"
  local sha=33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6   # the commit the runs used, pinned rather than the moving master branch
  curl -fL --retry 5 -o "$d/ESC-50-master.zip" https://github.com/karolpiczak/ESC-50/archive/$sha.zip
  local got; got=$(md5of "$d/ESC-50-master.zip")
  # GitHub may rebuild archives byte-differently; the check that matters is the metadata CSV and the clip count
  # 629e8e9e... is the master-branch archive the runs used; 071b4401... the pinned-commit archive as GitHub served it on 29 Oct 2026
  [ "$got" = 629e8e9ebc1592bcceb06da6bec40275 ] || [ "$got" = 071b44018315e034b2c6e8064543d19c ] || echo "note: zip MD5 $got differs from the archives seen before; checking content"
  unzip -q -o "$d/ESC-50-master.zip" -d "$d"
  if [ -d "$d/ESC-50-$sha" ]; then mkdir -p "$d/ESC-50-master" && cp -R "$d/ESC-50-$sha/." "$d/ESC-50-master/" && rm -rf "$d/ESC-50-$sha"; fi
  [ "$(md5of "$d/ESC-50-master/meta/esc50.csv")" = 54a0d0055a10bb7df84ad340a148722e ] || { echo "esc50.csv MD5 mismatch" >&2; exit 1; }
  [ "$(ls "$d/ESC-50-master/audio" | grep -c '\.wav$')" = 2000 ] || { echo "expected 2000 ESC-50 clips" >&2; exit 1; }
  mkdir -p "$E/ESC-50/ast_768"; cp "$ROOT/data/meta/esc-50.csv" "$E/ESC-50/ast_768/esc-50.csv"
  [ "$KEEP" = 1 ] || rm -f "$d/ESC-50-master.zip"
}
urban8k() {  # UrbanSound8K, Zenodo 1203745
  local d=$E/Urban8k/audio_raw/urbansound8k_zenodo_1203745
  get "$Z/1203745/files/UrbanSound8K.tar.gz/content" "$d/UrbanSound8K.tar.gz" 9aa69802bbf37fb986f71ec1483a196e
  tar -xzf "$d/UrbanSound8K.tar.gz" -C "$d"
  [ "$(md5of "$d/UrbanSound8K/metadata/UrbanSound8K.csv")" = 5015c37ce7d54c0bb8cd40a0b5ee2074 ] || { echo "UrbanSound8K.csv MD5 mismatch" >&2; exit 1; }
  mkdir -p "$E/Urban8k/ast_768"; cp "$ROOT/data/meta/urbansound8k.csv" "$E/Urban8k/ast_768/urbansound8k.csv"
  [ "$KEEP" = 1 ] || rm -f "$d/UrbanSound8K.tar.gz"
}
fsdk2018() {  # FSDKaggle2018, Zenodo 2552860 (T21 D1)
  local d=$E/FSDKaggle2018
  get "$Z/2552860/files/FSDKaggle2018.meta.zip/content" "$d/zips/FSDKaggle2018.meta.zip" f16828ac8be0e5285b9175a12f90d784
  get "$Z/2552860/files/FSDKaggle2018.doc.zip/content" "$d/zips/FSDKaggle2018.doc.zip" 4f2d1ac88f33a62f9db3108b269ee1b7
  get "$Z/2552860/files/FSDKaggle2018.audio_test.zip/content" "$d/zips/FSDKaggle2018.audio_test.zip" f85c8665c3fc39311ea1b748e194a81d
  get "$Z/2552860/files/FSDKaggle2018.audio_train.zip/content" "$d/zips/FSDKaggle2018.audio_train.zip" a8b47ff4b52022c178f88fa5c6f080d0
  mkdir -p "$d/meta_x" "$d/audio_raw"
  unzip -q -o "$d/zips/FSDKaggle2018.meta.zip" -d "$d/meta_x"; unzip -q -o "$d/zips/FSDKaggle2018.doc.zip" -d "$d/meta_x"
  unzip -q -o "$d/zips/FSDKaggle2018.audio_test.zip" -d "$d/audio_raw"; unzip -q -o "$d/zips/FSDKaggle2018.audio_train.zip" -d "$d/audio_raw"
  [ "$KEEP" = 1 ] || rm -f "$d"/zips/FSDKaggle2018.audio_*.zip
}
fsd50k_meta() {  # FSD50K ground truth and clip metadata, Zenodo 4060432 (T21 D2); the audio comes with t21_get_d2.sh
  local d=$E/FSD50K
  get "$Z/4060432/files/FSD50K.ground_truth.zip/content" "$d/zips/FSD50K.ground_truth.zip" ca27382c195e37d2269c4c866dd73485
  get "$Z/4060432/files/FSD50K.metadata.zip/content" "$d/zips/FSD50K.metadata.zip" b9ea0c829a411c1d42adb9da539ed237
  mkdir -p "$d/meta_x"
  unzip -q -o "$d/zips/FSD50K.ground_truth.zip" -d "$d/meta_x"; unzip -q -o "$d/zips/FSD50K.metadata.zip" -d "$d/meta_x"
}
weights() {  # EfficientAT release v0.0.1 (MIT) and PANNs (Zenodo 3987831, CC BY 4.0, T23 only)
  local u=https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1 d=$QW/efficientat_release-v0.0.1
  get $u/mn04_as_mAP_432.pt "$d/mn04_as_mAP_432.pt" d288913b10fd32c968b3d756236060a6
  get $u/mn05_as_mAP_443.pt "$d/mn05_as_mAP_443.pt" 712258b9d276be95421f039061074f87
  get $u/mn10_as_mAP_471.pt "$d/mn10_as_mAP_471.pt" cd7ce78dd3faf9b7972252201df29643
  get $u/mn20_as_mAP_478.pt "$d/mn20_as_mAP_478.pt" 7c6cfe0db5e5b256a4477559500129f1
  get $u/dymn04_as.pt "$d/dymn04_as.pt" 3b9611cfdb89126f3e9f7453dd908dbd
  get "$Z/3987831/files/Wavegram_Cnn14_mAP=0.389.pth/content" "$QW/panns-wavegram/Wavegram_Cnn14_mAP=0.389.pth" 1e3506ab640371e0b5a417b15fd66d21
  get "$Z/3987831/files/Cnn14_mAP=0.431.pth/content" "$QW/panns-cnn14/Cnn14_mAP=0.431.pth" 541141fa2ee191a88f24a3219fff024e
  # AudioSet label names, read by scripts/redesign/moe/t15_check_cnn14.py; EfficientAT at the commit the vendored model code follows
  get https://github.com/fschmid56/EfficientAT/raw/a425fdce92572e602a1d5634799bd9f1f2efa806/metadata/class_labels_indices.csv \
    "$QW/efficientat_src-a425fdc/metadata/class_labels_indices.csv" 68c0b5c199d6cc789852b144cc3ab500
  # The AST (MIT/ast-finetuned-audioset-10-10-0.4593, revision f826b80d28226b62986cc218e5cec390b1096902) is fetched by
  # transformers on first use; run once online, then set HF_HUB_OFFLINE=1 as the original runs did.
}
case ${1:-} in
  esc50) esc50 ;; urban8k) urban8k ;; fsdk2018) fsdk2018 ;; fsd50k-meta) fsd50k_meta ;; weights) weights ;;
  all) esc50; urban8k; fsdk2018; fsd50k_meta; weights ;;
  *) sed -n 2,6p "$0"; exit 2 ;;
esac
