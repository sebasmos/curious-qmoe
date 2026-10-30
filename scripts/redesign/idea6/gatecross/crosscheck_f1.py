"""Check this analysis' macro F1 against the F1 stored by the earlier analyses (static.csv of Stage 3c; seeds averaged
within folds on both sides). A mismatch above 1e-9 on any matched (system, condition, fold) would mean the score files were
read or assembled differently from the earlier work.

    python crosscheck_f1.py --out OUT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_by_precision import S3C, OB, UNP  # noqa: E402

TIER = {"FP32": "fp32", "Q8": "w8", "Q6": "w6", "Q5": "w5", "Q4": "w4g128", "Q4pc": "w4", "Q3": "w3g64"}
TIER_REAL = {"FP32": "fp32", "Q8": "w8g64", "Q4": "w4g128", "Q4pc": "w4", "Q3": "w3g64"}


def old_name(r):
    h = r["head"]
    if r.task == "tau":
        if r.fam == "cnn":
            return None if r.backbone == "INT8" else f"cnn:{r.model}:{ {'FP32': 'fp32', 'Q8sim': 'w8a8'}[r.backbone] }:{h}"
        return f"ast:{ {'FP32': 'fp32', 'Q8': 'w8g64', 'Q4': 'w4g128', 'Q4n': 'n4'}[r.backbone] }+{h}"
    if r.fam == "cnn":
        if r.backbone == "INT8":
            return f"cnn:{r.model}int8b:int8:{h}"
        bb = {"FP32": "fp32", "Q8sim": "w8a8"}[r.backbone]
        mdl = "mn10unpr" if r.model == "mn10exact" else r.model
        return f"cnn:{mdl}:{bb}:{h}"
    if r.model == "ast1024":
        return f"ast:{TIER[r.backbone]}+{h}"
    if r.backbone in TIER_REAL:
        return f"ast:{TIER_REAL[r.backbone]}@{r.length}+{h}"
    return None


def old_cond(c, task):
    if task == "tau":
        return c if c in ("clean", "shift") else f"dev:{c}"
    return "clean" if c in ("clean", "field") else f"{c.rstrip('0123456789')}__{c[len(c.rstrip('0123456789')):]}db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    sy = pd.read_csv(a.out / "systems.csv")
    f = pd.read_csv(a.out / "fold_metrics.csv.gz")
    rows = []
    jobs = [("esc50", S3C / "analysis/esc50/static.csv", False), ("esc5", S3C / "analysis/esc5/static.csv", False),
            ("urban", S3C / "analysis/urban/static.csv", False), ("urban", UNP / "analysis/urban/static.csv", True),
            ("tau", OB / "analysis/tau/static.csv", False)]
    for task, path, exact in jobs:
        old = pd.read_csv(path)
        dup = old.groupby(["system", "condition", "seed", "fold"]).f1.agg(lambda x: x.max() - x.min()).max()  # same system in several sets
        old = old.groupby(["system", "condition", "seed", "fold"], as_index=False).f1.mean()
        old = old.groupby(["system", "condition", "fold"], as_index=False).f1.mean().rename(columns={"f1": "f1_old"})
        spread = dup
        s = sy[(sy.task == task) & ((sy.model == "mn10exact") == exact)].copy()
        s["old"] = s.apply(old_name, axis=1)
        s = s.dropna(subset=["old"])
        new = f[f.sid.isin(s.sid)].merge(s[["sid", "old"]], on="sid")
        new["condition"] = new.cond.map(lambda c: "field" if c == "field" else old_cond(c, task))
        m = new.merge(old, left_on=["old", "condition", "fold"], right_on=["system", "condition", "fold"], how="inner")
        d = (m.f1 - m.f1_old).abs()
        rows.append(dict(task=task + (" exact-length" if exact else ""), source=path.parts[-4], matched_cells=len(m), systems=m.sid.nunique(), max_abs_diff=d.max(),
                         n_above_1e9=int((d > 1e-9).sum()), old_duplicates_spread=float(spread)))
        bad = m[d > 1e-9]
        if len(bad):
            print(bad.groupby("sid").size().head(20))
    out = pd.DataFrame(rows)
    out.to_csv(a.out / "crosscheck_f1.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
