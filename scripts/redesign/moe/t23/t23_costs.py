"""T23 P2: priced bit-ops per clip of the waveform expert (Wavegram-CNN14 W8A8) against the stored AST-Q4 and MN10-Q8 costs, per task.
Needs no weights and no audio: MACs come from hooks on a zero input of our own model code (QWave/panns_wavegram.py).
Convention (T15's t15_costs.py, QWave/costs.py): front end (16 -> 32 kHz resampler taps per output sample, plus pre_conv0, kept FP32)
x 32 x 32; W8A8 MACs x 8 x 8; one rescale per output element of a quantized layer at 32 x 32; plus the protocol 3 `esc` head bit-ops.
Primary price: the padded shape that runs (ESC-50 80000, UrbanSound8K 64587, TAU 160000 samples at 16 kHz). Secondary: each clip's
own length (T15's lengths.npy), mean over clips.
    python t23_costs.py --out FILE.json
"""
import argparse, contextlib, io, json, math, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from QWave import efficientat_mn as M
from QWave.costs import cost
from QWave.panns_wavegram import WavegramCnn14, macs_at
from scripts.benchmark import build_model
Q = Path.home() / "Documents/qmoe-results"
S3C = Q / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1"; OPTB = Q / "2026-10-23_2133_redesign-idea6-optionB_code-f320295"
T15 = Q / "2026-10-25_1415_moe-t15-t18_code-2ef25b0"; T21 = Q / "2026-10-26_1408_moe-t21_code-731570e"
PAD = {"esc50": 80000, "urban": 64587, "tau": 160000}; K = {"esc50": 50, "urban": 10, "tau": 10}; CFG = {"esc50": "esc50", "urban": "urban8k", "tau": "urban8k"}
PUBLISHED = {"Wavegram-CNN": 44.234e9, "Res1dNet31": 32.688e9, "CNN14": 42.220e9}   # PANNs Table XII, multi-adds per 10 s clip at 32 kHz
ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
torch.set_num_threads(2)
m = WavegramCnn14().eval()
_, _, taps = M.resample_taps()
cache = {}
def clip_cost(n16, k, ds):
    n32 = max(M.n32_of(int(n16)), 20480)  # the five 2x pools need 64 wavegram frames; shorter clips (UrbanSound8K) are priced at 0.64 s, as T15 priced CNN14 at 32 frames
    if n32 not in cache: cache[n32] = macs_at(m, n32)
    d = cache[n32]
    front = taps * n32 + d["front"]
    return dict(front_macs=front, w8a8_macs=d["macs"], out_elems=d["out_elems"],
                bitops=front * 1024 + d["macs"] * 64 + d["out_elems"] * 1024 + head[ds][1], macs=front + d["macs"] + head[ds][0])
head = {}
for ds in PAD:
    cfg = OmegaConf.load(ROOT / "config" / f"{CFG[ds]}.yaml")
    with contextlib.redirect_stdout(io.StringIO()): head[ds] = tuple(map(float, cost(build_model("esc", 2048, K[ds], cfg))))
st = {}
for ds in ("esc50", "urban"):
    c = json.load(open(S3C / f"analysis/{ds}/costs.json")); L = {"esc50": 512, "urban": 400}[ds]
    st[ds] = dict(ast_q4=c["ast"][f"w4g128@{L}+esc"], ast_q4_key=f"stage3c analysis/{ds}/costs.json ast w4g128@{L}+esc",
                  mn10_q8=c["cheap"]["cnn:mn10unp:w8a8:esc" if ds == "urban" else "cnn:mn10:w8a8:esc"])
c = json.load(open(OPTB / "analysis/tau/costs.json"))
st["tau"] = dict(ast_q4=c["ast"]["ast:w4g128+esc"], ast_q4_key="optionB analysis/tau/costs.json ast:w4g128+esc", mn10_q8=c["cheap"]["cnn:mn10:w8a8:esc"])
out = {"convention": __doc__.split("\n")[2:6], "published_multiadds_10s_32k": PUBLISHED}
d10 = macs_at(m, 320000)
out["check_10s"] = dict(our_macs_total=d10["front"] + d10["macs"], published_multiadds=PUBLISHED["Wavegram-CNN"],
                        ratio_published_over_ours=PUBLISHED["Wavegram-CNN"] / (d10["front"] + d10["macs"]),
                        params=int(sum(p.numel() for p in m.parameters())), params_published=80991759)
import pandas as pd
fk = pd.read_csv(T21 / "tables/d1/costs_per_clip_summary.csv", index_col=0)
out["fsdk2018_ast_q4_mean_max"] = [float(fk.loc["mean", "ast_w4g128_bitops"]), float(fk.loc["max", "ast_w4g128_bitops"])]
for ds in PAD:
    p = clip_cost(PAD[ds], K[ds], ds)
    lens = np.load(T15 / f"emb/{ds}/lengths.npy")
    u, cnt = np.unique(lens, return_counts=True)
    own = sum(c_ * clip_cost(n, K[ds], ds)["bitops"] for n, c_ in zip(u, cnt)) / cnt.sum()
    # conservative reading: the published multi-adds taken as MACs, scaled linearly to the padded length, all at 8 x 8
    cons = PUBLISHED["Wavegram-CNN"] * (PAD[ds] / 160000) * 64
    out[ds] = dict(pad_16k=PAD[ds], wave_q8_bitops_padded=p["bitops"], wave_q8_bitops_own_length_mean=float(own), wave_q8_detail_padded=p,
                   wave_q8_bitops_conservative_published=cons, **st[ds],
                   ratio_wave_over_ast_padded=p["bitops"] / st[ds]["ast_q4"], ratio_wave_over_ast_conservative=cons / st[ds]["ast_q4"],
                   ratio_wave_over_mn10=p["bitops"] / st[ds]["mn10_q8"], p2_pass=bool(p["bitops"] < st[ds]["ast_q4"] and cons < st[ds]["ast_q4"]))
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
for ds in PAD:
    o = out[ds]; print(f"{ds}: wave {o['wave_q8_bitops_padded']:.4g} (own length {o['wave_q8_bitops_own_length_mean']:.4g}, conservative {o['wave_q8_bitops_conservative_published']:.4g}) "
                       f"AST-Q4 {o['ast_q4']:.4g} MN10-Q8 {o['mn10_q8']:.4g} ratio {o['ratio_wave_over_ast_padded']:.3f} cons {o['ratio_wave_over_ast_conservative']:.3f} P2 {o['p2_pass']}")
print(out["check_10s"], out["fsdk2018_ast_q4_mean_max"])
