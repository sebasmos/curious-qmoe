"""Check 4: per-clip bit-ops of every configuration from first principles, compared with the
extraction sidecars (backbone_cost_per_clip) and the head costs in Stage 2's static.csv; and the
break-even price B (bit-ops per extra group-rescale MAC) at which each passing 5% cascade stops
passing (bisection, Bonferroni 30 as Stage 2).

    python check4_costs.py --stage2 STAGE2_DIR --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute import backbone_bitops, head_bitops, parent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage2", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
res = {"per_config_T": {}, "sidecar_mismatch": []}
for ds in ["esc50", "urban8k"]:
    for j in sorted((a.stage2 / "emb" / ds).glob("*__clean.json")):
        side = json.loads(j.read_text())
        c = side["config"]
        if "_p8" in c:
            continue
        mine = backbone_bitops(c)
        res["per_config_T"][c] = round(mine / 1e12, 4)
        if abs(mine - side["backbone_cost_per_clip"]["bitops"]) > 1:
            res["sidecar_mismatch"].append((ds, c, mine, side["backbone_cost_per_clip"]["bitops"]))
S = pd.read_csv(a.stage2 / "analysis" / "esc50" / "static.csv")
S["head_bitops"] = S.bitops - [backbone_bitops(c) for c in S.config]
res["esc50_head_bitops_from_static_csv"] = S.groupby("head").head_bitops.agg(["min", "max"]).to_dict()
res["esc50_head_bitops_mine"] = {"esc": head_bitops("esc", 50), "4": head_bitops("4", 50)}
res["group_overhead_extra_macs_per_clip"] = {g: (backbone_bitops(f"w3g{g}", 1.0) - backbone_bitops(f"w3g{g}"))
                                             for g in (128, 64, 32)}


def verdict_at(S, C, cond, ch, B, n_cls):
    S = S[S.cond == cond].copy()
    S["cost"] = [backbone_bitops(c, B) + head_bitops(h, n_cls) for c, h in zip(S.config, S["head"])]
    g = C[(C.cond == cond) & (C.cheap == ch) & (C.point == "b05")]
    cost = np.mean([backbone_bitops(ch, B) + head_bitops("esc", n_cls)
                    + r * (backbone_bitops(parent(ch), B) + head_bitops("esc", n_cls)) for r in g.rate])
    sp = S.groupby(["config", "head"]).agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
    comp = sp[sp.cost <= cost * (1 + 1e-9)].sort_values(["f1", "cost"], ascending=[False, True]).iloc[0]
    gs = S[(S.config == comp.config) & (S["head"] == comp["head"])].set_index(["seed", "fold"]).f1
    gc = g.set_index(["seed", "fold"]).f1
    k = sorted(set(gs.index) & set(gc.index))
    x, z = gc.loc[k].to_numpy(), gs.loc[k].to_numpy()
    ok = (x - z).mean() > 0 and 30 * stats.ttest_rel(x, z).pvalue < 0.05 and 30 * stats.wilcoxon(x, z).pvalue < 0.05
    return ok, f"{comp.config}+{comp['head']}"


rows = []
for ds, n_cls in [("esc50", 50), ("esc5", 5), ("urban", 10)]:
    S = pd.read_csv(a.out / "bitops" / ds / "static_pairs.csv")
    C = pd.read_csv(a.out / "bitops" / ds / "cascade_pairs.csv")
    for cond in sorted(C.cond.unique()):
        for ch in ["n3g64", "w3g64"]:
            ok0, comp0 = verdict_at(S, C, cond, ch, 0.0, n_cls)
            if not ok0:
                continue
            lo, hi = 0.0, 1024.0
            if verdict_at(S, C, cond, ch, hi, n_cls)[0]:
                rows.append(dict(dataset=ds, cond=cond, cheap=ch, break_even_B=">1024"))
                continue
            for _ in range(12):
                mid = (lo + hi) / 2
                if verdict_at(S, C, cond, ch, mid, n_cls)[0]:
                    lo = mid
                else:
                    hi = mid
            rows.append(dict(dataset=ds, cond=cond, cheap=ch, break_even_B=round(hi, 1),
                             comparator_after=verdict_at(S, C, cond, ch, hi, n_cls)[1]))
res["break_even"] = rows
(a.out / "check4_costs.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
