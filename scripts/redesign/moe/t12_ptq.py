"""T12 quantization route: merge the trained LoRA into the AST, quantize every encoder Linear to w4g128 with QWave.backbone_quant (8-bit activations),
and score fold-1 test clips with the trained head. Descriptive only (ANALYSIS_PLAN_T12.md section 6).
    python t12_ptq.py --runs DIR --arm twin|spec --seed 0 --steps N --out FILE.npz"""
import argparse, os, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=Path, required=True); ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, required=True); ap.add_argument("--k", type=int, default=2); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--config", default="w4g128")
a = ap.parse_args()
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t12_common import *
from common import load_table
from t1c_common import PILOT
from audio_shift import FIELD_SEEDS, NoiseMaker, field_assignment, index_wavs, load_clip, mix
from QWave.backbone_quant import QuantAST, parse_config
torch.set_num_threads(2); dev = torch.device("mps"); B = NL - a.k
E = Path.home() / "Documents/DATASETS"; AUD = E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio"; BGA = E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"
d = load_table(); stems = d.stem.tolist(); te = np.load(PILOT / f"t1/raw/s{a.seed}_f1.npz")["te"]; index = index_wavs(AUD)
bg_names = [Path(s).stem for s in pd.read_csv(E / "ESC-50/ast_768/esc-50.csv", usecols=["name"])["name"]]; bg_index = index_wavs(BGA); nm = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)
asg = field_assignment(len(d), FIELD_SEEDS["test"])
def cond(cn, r):
    if cn == "clean": return "clean", None
    if cn == "bg0": return "bg", 0.0
    return ("clean", None) if asg[r] == "clean" else (asg[r].split("__")[0], float(asg[r].split("__")[1][:-2]))
ext, model = load_ast(400, dev); tail = Tail(model, B, "lora")
tail.load_state_dict({k: v.to(dev) for k, v in torch.load(a.runs / f"{a.arm}_k{a.k}_s{a.seed}_n{a.steps}_tail.pt").items()}); tail.eval()
tail.write_back(model, B); q = QuantAST(model); q.set_config(*parse_config(a.config)); out = {"te": te}
for cn in ("clean", "bg0", "field"):
    Z = []
    for s in range(0, len(te), 16):
        ws = []
        for r in te[s:s + 16]:
            x = load_clip(index[stems[r]]); kind, snr = cond(cn, int(r))
            if kind != "clean": x, _ = mix(x, nm(stems[r], kind, len(x)), snr)
            ws.append(x)
        f = ext(ws, sampling_rate=16000, return_tensors="pt")["input_values"].contiguous().to(dev)
        with torch.no_grad(): Z.append(tail.head(q(f)).float().cpu().numpy())
    out[f"test_{cn}"] = np.concatenate(Z); print(cn, "done", flush=True)
np.savez(a.out, **out)
