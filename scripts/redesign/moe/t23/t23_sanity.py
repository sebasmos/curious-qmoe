"""T23 functional check of the reimplementation (ANALYSIS_PLAN.md C1): strict load, dropped keys, parameter count, AudioSet top-3 tags
on six ESC-50 clips chosen by file-name target (dog 0, rooster 1, rain 10, crying baby 20, siren 42, church bells 46), FP32 and W8A8.
No task score. Run with python -I.    python -I t23_sanity.py CKPT LABELS_CSV OUT_TXT"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2"))
import csv, hashlib
import numpy as np, torch
from QWave.panns_wavegram import load_wavegram, dropped_keys, QuantWavegram
from QWave import efficientat_mn as M
from audio_shift import load_clip
ck, lab, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
torch.set_num_threads(2)
names = [r["display_name"] for r in csv.DictReader(open(lab))]
m = load_wavegram(ck); q = QuantWavegram(m, 8, 8, "cpu")
lines = [f"checkpoint {ck.name} sha256 {hashlib.sha256(ck.read_bytes()).hexdigest()}", f"dropped keys: {dropped_keys(ck)}",
         f"parameters: {sum(p.numel() for p in m.parameters())}"]
aud = ROOT.parent  # unused
A = Path.home() / "Documents/DATASETS/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"
for t in (0, 1, 10, 20, 42, 46):
    f = sorted(A.glob(f"*-{t}.wav"))[0]
    x = torch.from_numpy(load_clip(f)).float()[None]
    w = M.resample_to_32k(x)
    with torch.no_grad():
        p = torch.sigmoid(m.fc_audioset(m(w)))[0].numpy(); pq = torch.sigmoid(q.net.fc_audioset(q(w)))[0].numpy()
    top = lambda v: ", ".join(f"{names[i]} {v[i]:.2f}" for i in np.argsort(-v)[:3])
    lines.append(f"{f.name}: FP32 [{top(p)}] | W8A8 [{top(pq)}]")
out.write_text("\n".join(lines) + "\n"); print("\n".join(lines))
