"""One-off check: QWave/efficientat_mn.py (mn05, mn20) and QWave/efficientat_dymn.py (dymn04) against EfficientAT's
own code (commit a425fdc), same released weights, on random input and on real ESC-50 clips.
Usage: python -I check_equiv.py SRC WT REPO OUT"""
import contextlib, io, json, os, sys, warnings
warnings.filterwarnings("ignore")
src, wt, repo, out = sys.argv[1:5]
sys.path.insert(0, repo)
sys.path.insert(1, src)
os.chdir(src)  # helpers/utils.py reads metadata/class_labels_indices.csv relative to the working directory
import numpy as np, torch
torch.set_num_threads(2)
from models.mn.model import get_model as get_mn
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.utils import labels
from QWave import efficientat_mn as M
from QWave import efficientat_dymn as D
sys.path.insert(0, os.path.join(repo, "scripts/redesign/idea6/stage2"))
from audio_shift import load_clip
res = {}
esc = "/Users/cajas.sebastian/Documents/DATASETS/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"
clips = {"dog": "1-100032-A-0.wav", "rooster": "1-26806-A-1.wav", "rain": "1-17367-A-10.wav", "siren": "1-31482-A-42.wav",
         "church_bells": "1-13571-A-46.wav", "crying_baby": "1-187207-A-20.wav"}
with contextlib.redirect_stdout(io.StringIO()):
    mel = AugmentMelSTFT().eval()
for name, w, files in (("mn05", 0.5, "mn05_as_mAP_443.pt"), ("mn20", 2.0, "mn20_as_mAP_478.pt"), ("dymn04", 0.4, "dymn04_as.pt")):
    path = os.path.join(wt, files)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    with contextlib.redirect_stdout(io.StringIO()):
        if name.startswith("dymn"):
            ref = get_dymn(width_mult=w, pretrained_name=None).eval()
            # get_model sets T_max = pretrain_final_temp (1.0) only when pretrained_name is given; do it by hand
            for mod in ref.modules():
                if mod.__class__.__name__ == "DynamicConv":
                    mod.temperature = 1.0
            mine = D.load_dymn(name, path)
        else:
            ref = get_mn(width_mult=w, head_type="mlp").eval()
            mine = M.load_mn(name, path)
    ref.load_state_dict(sd, strict=True)
    torch.manual_seed(0)
    x = torch.randn(3, 1, 128, 501)
    with torch.no_grad():
        lr, fr = ref(x)
        lm, fm = mine.audioset_logits(x), mine(x)
    r = {"weights": files, "random_logits_maxabs": float((lr - lm).abs().max()), "random_feat_maxabs": float((fr - fm).abs().max()),
         "emb_dim": int(fm.shape[1]), "n_state": len(sd), "n_params": int(sum(v.numel() for v in sd.values()))}
    tops = {}
    for k, f in clips.items():
        wav16 = torch.from_numpy(load_clip(__import__("pathlib").Path(esc) / f))[None]
        w32 = M.resample_to_32k(wav16)
        with torch.no_grad():
            a = mel(w32)[:, None]
            b = M.log_mel(w32)
            la, fa = ref(a)
            lb = mine.audioset_logits(b)
            fb = mine(b)
        r["mel_maxabs"] = max(r.get("mel_maxabs", 0.0), float((a - b).abs().max()))
        r["clip_logits_maxabs"] = max(r.get("clip_logits_maxabs", 0.0), float((la - lb).abs().max()))
        r["clip_feat_maxabs"] = max(r.get("clip_feat_maxabs", 0.0), float((fa - fb).abs().max()))
        p = torch.sigmoid(lb[0])
        tops[k] = [(labels[i], round(float(p[i]), 3)) for i in p.argsort(descending=True)[:3].tolist()]
    r["top3_audioset"] = tops
    # cost against the published GMACs at 10 s (1001 frames): mn05 0.16, mn20 2.06, dymn04 0.12
    fr10 = M.frames_of(320000)
    c = D.feature_macs(mine, fr10) if name.startswith("dymn") else M.feature_macs(mine, fr10)
    r["gmacs_10s"] = c["macs"] / 1e9
    if "aggregate_macs" in c:
        r["gmacs_10s_aggregate"] = c["aggregate_macs"] / 1e9
    res[name] = r
    print(name, json.dumps(r, indent=1), flush=True)
open(out, "w").write(json.dumps(res, indent=1))
