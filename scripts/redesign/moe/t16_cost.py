"""T16 costs (priced bit-ops per clip, repo convention) of the early-exit path and of every single model, per bed. Reads stored cost tables and sidecars; computes the FP32 AST at the task's frame count
with QWave.backbone_cost on the shortened model. python t16_cost.py --out FILE.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from t16_common import *
from QWave.backbone_quant import backbone_cost, parse_config
from scripts.redesign.idea6.costs_idea6 import head_costs
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args()
res = {}
HEAD = head_costs(768, ["esc"])
SIDE = {"urban": I5 / "emb/urban8k/fp32__clean.json", "esc50": I5 / "emb/esc50/fp32__clean.json", "tau": OPTB / "emb_ast/tau/fp32__clean.json"}
for bed in BEDS:
    L = FRAMES[bed]; _, model = load_ast(L, "cpu"); bc = backbone_cost(model, *parse_config("fp32"), p_bits=16); ct = cost_table(bed)
    c3 = json.load(open((OPTB if bed == "tau" else S3C) / f"analysis/{bed}/costs.json"))["ast"]
    S_tot = c3["ast:fp32+esc"] if bed == "tau" else c3[f"fp32@{L}+esc"]          # priced FP32 AST at the task's frame count: fbank + backbone + head, as stored
    extras = S_tot - bc["bitops"]                                                   # fbank front end and the 768-d head
    head = HEAD["esc50" if bed == "esc50" else "urban"]["esc"]["bitops"]            # TAU head has 10 classes like UrbanSound8K
    parts = bc["parts"]; enc = parts["linear"]["bitops"] + parts["attention"]["bitops"]; patch = parts["patch_embed"]["bitops"]
    assert 0 < head < extras < 2e10, (head, extras)
    K = BEDS[bed]["ncls"]; router = (1536 + K + 1) * 128 * 32 * 32 + 128 * 32 * 32; router_out = (K + 1) * 128 * 32 * 32 + 128 * 32 * 32     # largest hidden size in the grid, FP32 MACs
    S4 = ct["E2b"] if bed == "tau" else ct["E2c"]                                       # stored price of the 4-bit AST of the same graph (E2c at 400/512 frames, E2b at 1024)
    res[bed] = dict(frames=L, E1=ct["E1"], fbank=extras - head, head=head, patch=patch, router=float(router), router_out=float(router_out), singles={**ct, "S": S_tot},
                    fp32=dict(enc_per_block=enc / 12, S_tot=S_tot, enc_total=enc), w4g128=dict(enc_per_block=(S4 - (extras - head) - patch - head) / 12, S_tot=S4, strong="E2b" if bed == "tau" else "E2c"))
    print(bed, {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in res[bed].items() if k != "singles"}, {k: f"{v:.4g}" for k, v in res[bed]["singles"].items()})
a.out.write_text(json.dumps(res, indent=1))
