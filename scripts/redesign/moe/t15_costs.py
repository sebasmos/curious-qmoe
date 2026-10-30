"""Priced cost per clip of each view expert (ANALYSIS_PLAN_T15.md section 3), in the repo's convention: frontend MACs x 1024 + backbone MACs x 64 + backbone output
elements x 1024 (rescale price) + `esc` head bit-ops, means over the true clip lengths (lengths.npy of the extraction). Also checks the formula against the stored
mn10 cost (cnn:mn10:w8a8:esc of stage 3c, ESC-50).   python t15_costs.py --r R --datasets urban esc50 tau"""
import argparse, contextlib, io, json, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "scripts/redesign/idea6"))
from QWave import efficientat_mn as M, panns_cnn14 as P
from QWave.costs import cost
from scripts.benchmark import build_model
from omegaconf import OmegaConf
import t15_views as V
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--datasets", nargs="+", default=["urban", "esc50", "tau"]); a = ap.parse_args()
torch.set_num_threads(2)
W = Path.home() / "Documents/qmoe-weights"
base = M.load_mn("mn10", W / "efficientat_release-v0.0.1/mn10_as_mAP_471.pt"); cnn14 = P.load_cnn14(W / "panns-cnn14/Cnn14_mAP=0.431.pth")
NCLS = {"urban": 10, "esc50": 50, "tau": 10}; CFG = {"urban": "urban8k", "esc50": "esc50", "tau": "urban8k"}
def head_bitops(dim, ds):
    cfg = OmegaConf.load(ROOT / "config" / f"{CFG[ds]}.yaml")
    with contextlib.redirect_stdout(io.StringIO()): macs, bitops = cost(build_model("esc", dim, NCLS[ds], cfg))
    return float(macs), float(bitops)
def hw_macs(model, rows, frames):
    parts = [0.0, 0.0]
    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size; n_out = out.shape[1] * out.shape[2] * out.shape[3]; parts[0] += n_out * (mod.in_channels // mod.groups) * kh * kw
        else: n_out = mod.out_features; parts[0] += mod.in_features * mod.out_features
        parts[1] += n_out
    hs = [m.register_forward_hook(hook) for m in model.features.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad(): model.features(torch.zeros(1, 1, rows, frames))
    for h in hs: h.remove()
    return parts[0], parts[1]
_mac = {}
out = json.load(open(a.r / 'inputs/view_costs.json')) if (a.r / 'inputs/view_costs.json').exists() else {}
for ds in a.datasets:
    lens = np.load(a.r / f"emb/{ds}/lengths.npy")
    if (lens == 0).any(): raise SystemExit(f"{ds}: lengths.npy has zero entries (clean pass incomplete)")
    uniq, cnt = np.unique(lens, return_counts=True); out[ds] = {}
    for v in V.ALL_VIEWS:
        fe = bb = oe = 0.0
        for n, c in zip(uniq, cnt):
            fr = V.frames_of(v, int(n)); fe += c * V.frontend_macs(v, int(n))
            if v == "v6":
                key = ("v6", max(fr, 32))  # five 2x2 poolings need at least 32 frames; shorter clips (UrbanSound8K has some under 0.3 s) are priced at 32
                if key not in _mac: d = P.macs_at(cnn14, key[1]); _mac[key] = (d["macs"], d["out_elems"])
            else:
                key = (V.n_rows(v), fr)
                if key not in _mac: _mac[key] = hw_macs(base, *key)
            bb += c * _mac[key][0]; oe += c * _mac[key][1]
        fe, bb, oe = fe / len(lens), bb / len(lens), oe / len(lens)
        hm, hb = head_bitops(V.DIM[v], ds)
        out[ds][v] = dict(frontend_macs_mean=fe, backbone_macs_mean=bb, backbone_out_elems_mean=oe, head_macs=hm, head_bitops=hb, frontend_bitops=fe * 1024,
                          backbone_bitops_rescale1024=bb * 64 + oe * 1024, total_bitops=fe * 1024 + bb * 64 + oe * 1024 + hb, mean_frames=float(np.mean([V.frames_of(v, int(n)) for n in lens])), n_rows=V.n_rows(v))
        print(ds, v, f"{out[ds][v]['total_bitops']:.4e}", flush=True)
# check of the formula against the stored mn10 cost on ESC-50 (all clips 80000 samples) when v0 exists
st = json.load(open(Path.home() / "Documents/qmoe-results/2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1/analysis/esc50/costs.json"))["cheap"]["cnn:mn10:w8a8:esc"]
if "esc50" in out: out["check_mn10_esc50"] = dict(ours=out["esc50"]["v0"]["total_bitops"], stored=st, rel_diff=out["esc50"]["v0"]["total_bitops"] / st - 1)
(a.r / "inputs").mkdir(exist_ok=True); (a.r / "inputs/view_costs.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out.get("check_mn10_esc50")))
