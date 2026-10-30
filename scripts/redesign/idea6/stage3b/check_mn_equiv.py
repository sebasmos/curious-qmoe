"""One-off check: QWave/efficientat_mn.py against EfficientAT's own code (commit a425fdc, read in full first),
same released weights, on random input and on real ESC-50 clips. Usage: python -I check_mn_equiv.py SRC WT REPO OUT"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
src, wt, repo, out = sys.argv[1:5]
sys.path.insert(0, repo)
sys.path.insert(1, src)
os.chdir(src)  # helpers/utils.py reads metadata/class_labels_indices.csv relative to the working directory
import numpy as np, torch
torch.set_num_threads(2)
from models.mn.model import get_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import labels
from QWave import efficientat_mn as M
sys.path.insert(0, os.path.join(repo, "scripts/redesign/idea6/stage2"))
from audio_shift import load_clip
import contextlib, io
res = {}
esc = "/Users/cajas.sebastian/Documents/DATASETS/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"
clips = {"dog": "1-100032-A-0.wav", "rooster": "1-26806-A-1.wav", "rain": "1-17367-A-10.wav", "siren": "1-31482-A-42.wav",
         "church_bells": "1-13571-A-46.wav", "crying_baby": "1-187207-A-20.wav"}
for name, w in (("mn04", 0.4), ("mn10", 1.0)):
    with contextlib.redirect_stdout(io.StringIO()):
        ref = get_model(width_mult=w, head_type="mlp").eval()
    sd = torch.load(os.path.join(wt, {"mn04": "mn04_as_mAP_432.pt", "mn10": "mn10_as_mAP_471.pt"}[name]), map_location="cpu", weights_only=True)
    ref.load_state_dict(sd, strict=True)
    mine = M.load_mn(name, os.path.join(wt, {"mn04": "mn04_as_mAP_432.pt", "mn10": "mn10_as_mAP_471.pt"}[name]))
    torch.manual_seed(0)
    x = torch.randn(3, 1, 128, 501)
    with torch.no_grad():
        lr, fr = ref(x)
        lm, fm = mine.audioset_logits(x), mine(x)
    r = {"random_logits_maxabs": float((lr - lm).abs().max()), "random_feat_maxabs": float((fr - fm).abs().max()),
         "emb_dim": int(fm.shape[1])}
    with contextlib.redirect_stdout(io.StringIO()):
        mel = AugmentMelSTFT().eval()
    tops = {}
    for k, f in clips.items():
        wav16 = torch.from_numpy(load_clip(__import__("pathlib").Path(esc) / f))[None]
        w32 = M.resample_to_32k(wav16)
        with torch.no_grad():
            a = mel(w32)[:, None]
            b = M.log_mel(w32)
            la, fa = ref(a)
            lb = mine.audioset_logits(b)
        r.setdefault("mel_maxabs", 0.0)
        r["mel_maxabs"] = max(r["mel_maxabs"], float((a - b).abs().max()))
        r.setdefault("clip_logits_maxabs", 0.0)
        r["clip_logits_maxabs"] = max(r["clip_logits_maxabs"], float((la - lb).abs().max()))
        p = torch.sigmoid(lb[0])
        tops[k] = [(labels[i], round(float(p[i]), 3)) for i in p.argsort(descending=True)[:3].tolist()]
    r["top3_audioset"] = tops
    r["frames_5s"] = int(b.shape[-1])
    res[name] = r
    print(name, json.dumps(r, indent=1))
open(out, "w").write(json.dumps(res, indent=1))
