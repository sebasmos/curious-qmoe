"""T22 step P2 (ANALYSIS_PLAN.md C3): priced cost per clip of the coarse expert from the architecture alone, front end included,
against the stored MN10-Q8 cost of each task (stage 3c / option B costs.json, the numbers T13 and T21 used). No data is read except
the clip lengths, which are fixed by the padding rule (every clip runs at the task's padded length).
    python t22_costs.py --out DIR"""
import argparse, contextlib, io, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from omegaconf import OmegaConf
from QWave import efficientat_mn as M
from QWave.costs import cost
import t22_view as V

ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
torch.set_num_threads(2)
Q = Path.home() / "Documents/qmoe-results"
S3C = Q / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1"; OPTB = Q / "2026-10-23_2133_redesign-idea6-optionB_code-f320295"
MN10_STORED = {"esc50": (S3C / "analysis/esc50/costs.json", "cnn:mn10:w8a8:esc"), "urban": (S3C / "analysis/urban/costs.json", "cnn:mn10unp:w8a8:esc"),
               "tau": (OPTB / "analysis/tau/costs.json", "cnn:mn10:w8a8:esc")}
NCLS = {"esc50": 50, "urban": 10, "tau": 10}
out = {"rule": "P2: coarse expert priced bit-ops <= 0.5 x MN10-Q8 priced bit-ops, front end included, on every task", "tasks": {}}
for arch in ("mn04", "mn10"):
    base = M.build_mn(arch)
    for task, n16 in V.PAD16.items():
        fr = V.frames_of(n16)
        x = V.logmel(torch.zeros(1, n16))
        assert x.shape[-2:] == (V.N_MELS, fr), (x.shape, fr)
        bb_macs, bb_out = V.feature_macs(base, V.N_MELS, fr)
        fe = V.frontend_macs(n16)
        head = cost(torch.nn.Linear(base.emb_dim, NCLS[task]))
        tot = fe * 1024 + bb_macs * 64 + bb_out * 1024 + head[1]
        f, key = MN10_STORED[task]; ref = json.load(open(f))["cheap"][key]
        # native MN10 recomputed with the same formula on the same padded length, as a check of the counting
        nat_fr = M.frames_of(M.n32_of(n16)); nat = M.feature_macs(base if arch == "mn10" else M.build_mn("mn10"), nat_fr)
        cfg = OmegaConf.load(ROOT / "config/esc50.yaml")
        from scripts.benchmark import build_model
        with contextlib.redirect_stdout(io.StringIO()): esc_head = cost(build_model("esc", 960, NCLS[task], cfg))
        nat_tot = M.frontend_macs(n16) * 1024 + nat["macs"] * 64 + nat["out_elems"] * 1024 + esc_head[1]
        out["tasks"].setdefault(task, {})[arch] = dict(
            padded_samples_16k=n16, frames=fr, mel_rows=V.N_MELS, frontend_macs=fe, backbone_macs=bb_macs, backbone_out_elems=bb_out,
            head_macs=float(head[0]), head_bitops=float(head[1]), frontend_bitops=fe * 1024, backbone_bitops_rescale1024=bb_macs * 64 + bb_out * 1024,
            total_bitops=tot, mn10q8_stored_bitops=ref, mn10q8_stored_key=key, ratio_to_mn10q8=tot / ref, p2_passes=bool(tot <= 0.5 * ref),
            unpriced_macs_total=fe + bb_macs + float(head[0]), mn10_native_recomputed_at_padded_length=nat_tot,
            ratio_to_native_recomputed=tot / nat_tot)
        print(task, arch, f"coarse {tot:.4e}  MN10-Q8 stored {ref:.4e}  ratio {tot / ref:.3f}  (native recomputed at pad {nat_tot:.4e}, ratio {tot / nat_tot:.3f})", flush=True)
a.out.mkdir(parents=True, exist_ok=True); (a.out / "p2_costs.json").write_text(json.dumps(out, indent=1))
