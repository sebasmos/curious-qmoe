"""Check 4 of ANALYSIS_PLAN.md: the clean rows of part B reproduce the earlier stages' numbers.

Every static system (cheap tier x head kind, AST-Q4 x head) on the clean condition: per (seed, fold) F1 and the
cost against Stage 3c's (ESC-50, UrbanSound8K mn20) and the follow-up's (UrbanSound8K exact-length mn10)
analysis/<task>/static.csv. For ESC-50 also the picked cascade: F1, cost, random-gate F1 and noise-gate F1 against the
P3 clean row of Stage 3c's selected.csv. Exit status 1 if any difference exceeds 1e-9 (cost: relative).

    python check_repro.py --r R --s3c S3C --unp UNP --s3b S3B --out R/logs/check_repro.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ["esc", "8", "4", "bitnet"]


def cmp_static(mine, ref, name_map):
    worst_f1 = worst_cost = 0.0
    n = 0
    for m, r in name_map.items():
        a = mine[(mine.system == m) & (mine.condition == "clean")].sort_values(["seed", "fold"])
        b = ref[(ref.system == r) & (ref.condition == "clean")].sort_values(["seed", "fold"])
        if len(a) != len(b) or len(a) == 0:
            raise SystemExit(f"{m} / {r}: {len(a)} against {len(b)} rows")
        if not (a.seed.to_numpy() == b.seed.to_numpy()).all() or not (a.fold.to_numpy() == b.fold.to_numpy()).all():
            raise SystemExit(f"{m}: (seed, fold) keys differ")
        worst_f1 = max(worst_f1, float(np.abs(a.f1.to_numpy() - b.f1.to_numpy()).max()))
        worst_cost = max(worst_cost, float((np.abs(a.cost.to_numpy() - b.cost.to_numpy()) / b.cost.to_numpy()).max()))
        n += 1
    return {"systems": n, "max_abs_f1_diff": worst_f1, "max_rel_cost_diff": worst_cost}


def main():
    ap = argparse.ArgumentParser()
    for k in ("r", "s3c", "unp", "s3b", "out"):
        ap.add_argument(f"--{k}", required=True, type=Path)
    a = ap.parse_args()
    res = {}
    me = pd.read_csv(a.r / "analysis/b_esc50/static.csv")
    ref = pd.read_csv(a.s3c / "analysis/esc50/static.csv")
    # mn10 on ESC-50 is Stage 3b's; Stage 3c's table holds it too (the hull set 'stage3b')
    nm = {f"mn20:{k}": f"cnn:mn20:w8a8:{k}" for k in KINDS} | {f"mn10:{k}": f"cnn:mn10:w8a8:{k}" for k in KINDS} \
        | {f"ast:{h}": f"ast:w4g128@512+{h}" for h in ("esc", "4")}
    res["esc50_statics"] = cmp_static(me, ref, nm)
    mu = pd.read_csv(a.r / "analysis/b_urban/static.csv")
    ru = pd.read_csv(a.unp / "analysis/urban/static.csv")
    nmu = {f"mn10:{k}": f"cnn:mn10unpr:w8a8:{k}" for k in KINDS} | {f"mn20:{k}": f"cnn:mn20:w8a8:{k}" for k in KINDS} \
        | {f"ast:{h}": f"ast:w4g128@400+{h}" for h in ("esc", "4")}
    res["urban_statics"] = cmp_static(mu, ru, nmu)
    sel = pd.read_csv(a.s3c / "analysis/esc50/selected.csv")
    sel = sel[(sel.family == "P3") & (sel.condition == "clean") & (sel.hull_set == "new")].iloc[0]
    cas = pd.read_csv(a.r / "analysis/b_esc50/cascade_vs_comparators.csv")
    cas = cas[cas.condition == "clean"].iloc[0]
    res["esc50_cascade_clean"] = {"f1": abs(cas.f1 - sel.f1), "cost_rel": abs(cas.cost - sel.cost) / sel.cost,
                                  "rate": abs(cas.rate - sel.rate), "random_f1": abs(cas.random_f1 - sel.random_f1),
                                  "noise_gate_f1": abs(cas.noise_f1 - sel.audio_f1)}
    res = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()}
    worst = max(vv for v in res.values() for kk, vv in v.items() if kk != "systems")
    res["worst"] = worst
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if worst <= 1e-9 else 1)


if __name__ == "__main__":
    main()
