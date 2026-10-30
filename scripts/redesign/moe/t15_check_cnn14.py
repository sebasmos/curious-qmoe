"""T15 checks C2 and C3 of the CNN14 view (ANALYSIS_PLAN_T15.md section 3d): checkpoint size and MD5; our mel matrix and DFT window against the buffers stored in the
checkpoint; our front end against the vendored torchlibrosa classes (librosa replaced by a three-function stub, because librosa's dependencies are not installed) on random
clips; FP32 AudioSet clipwise outputs on a few clips with a clear AudioSet counterpart.   python -I t15_check_cnn14.py --ckpt F --torchlibrosa DIR --out JSON"""
import argparse, hashlib, json, sys, types
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=Path, required=True); ap.add_argument("--torchlibrosa", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--root", type=Path, required=True); a = ap.parse_args()
sys.path.insert(0, str(a.root))
import numpy as np, torch
from scipy.signal import get_window
import pandas as pd
from QWave import panns_cnn14 as P
res = {}
h = hashlib.md5()
with open(a.ckpt, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b)
res["md5"], res["size"] = h.hexdigest(), a.ckpt.stat().st_size
res["md5_expected"], res["size_expected"] = "541141fa2ee191a88f24a3219fff024e", 327428481
res["md5_ok"] = res["md5"] == res["md5_expected"] and res["size"] == res["size_expected"]
res["buffers"] = P.check_buffers(a.ckpt)
# torchlibrosa with a stub librosa (get_window, pad_center, filters.mel returning the checkpoint's own mel matrix)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=True); sd = ck["model"] if "model" in ck else ck
melW = sd["logmel_extractor.melW"].numpy()
lib = types.ModuleType("librosa"); lib.filters = types.SimpleNamespace(get_window=lambda w, n, fftbins=True: get_window(w, n, fftbins=fftbins), mel=lambda **kw: melW.T)
lib.util = types.SimpleNamespace(pad_center=lambda d, size: np.pad(d, ((size - len(d)) // 2, size - len(d) - (size - len(d)) // 2)), exceptions=types.SimpleNamespace(ParameterError=ValueError))
sys.modules["librosa"] = lib
sys.path.insert(0, str(a.torchlibrosa.parent))
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
spec = Spectrogram(n_fft=1024, hop_length=320, win_length=1024, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
logm = LogmelFilterBank(sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
spec.eval(); logm.eval()
g = torch.Generator().manual_seed(0)
x = torch.randn(8, 32000 * 4, generator=g) * torch.rand(8, 1, generator=g)
with torch.no_grad():
    ref = logm(spec(x))
ours = P.log_mel(x)
res["frontend_max_abs_diff_db"] = float((ref - ours).abs().max()); res["frontend_ref_range_db"] = [float(ref.min()), float(ref.max())]
# C3: AudioSet sanity on real clips
base = P.load_cnn14(a.ckpt)
names = pd.read_csv(Path.home() / "Documents/qmoe-weights/efficientat_src-a425fdc/metadata/class_labels_indices.csv")["display_name"].tolist()
sys.path.insert(0, str(a.root / "scripts/redesign/idea6/stage2"))
from audio_shift import index_wavs, load_clip
import torchaudio.functional as AF
D = Path.home() / "Documents/DATASETS"
clips = [("dog bark", D / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio/1-100032-A-0.wav"), ("siren", None), ("engine/idle", None)]
esc = pd.read_csv(D / "ESC-50/ast_768/esc-50.csv", usecols=["name", "label"])
idx = index_wavs(D / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio")
out = []
for cat, lab in [("dog", 0), ("siren", 42), ("engine", 44), ("rain", 10), ("clock_tick", 38)]:
    r = esc[esc.label == lab]
    if r.empty:
        continue
    stem = Path(r.iloc[0]["name"]).stem
    w = torch.from_numpy(load_clip(idx[stem]))[None]
    with torch.no_grad():
        p = base.audioset_logits(P.log_mel(AF.resample(w, 16000, 32000)))[0]
    top = torch.topk(p, 5)
    out.append({"esc50_category": cat, "clip": stem, "top5": [(names[i], round(float(v), 3)) for v, i in zip(top.values, top.indices)]})
res["audioset_top5"] = out
a.out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
