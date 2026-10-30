"""T26 Part B (ANALYSIS_PLAN.md C6), DESCRIPTIVE: S1 and S2 on the development data (ESC-50 50 classes, UrbanSound8K,
TAU) from stored score files, against the widened hull of the statics that exist there with matching folds.

ESC-50 and UrbanSound8K: MN04-Q8 and MN10-Q8 from stage 3b, MN05-Q8, MN20-Q8, DyMN04-Q8 from stage 3c (all on the
padded 4.04 s UrbanSound8K input, so the five CNNs read the same audio), AST-Q4 = AST w4g128 at the clips' real
length (512 frames ESC-50, 400 UrbanSound8K) from stage 2; FP32 `esc` heads; beds clean and the field mix.
TAU: MN10-Q8, MN20-Q8 and AST w4g128 (10 s, 1024 frames) from option B; no MN04, MN05 or DyMN04 exist there, so the
TAU hull has MN10-Q8, MN20-Q8, AST-Q4, always-average and MN10+MN20 only; beds device A (clean) and the other
devices (shift); the stored option B heads are trained on device A.
Costs: the stored per-clip priced bit-ops (stage 3c and option B costs.json), a pair the sum of both.

    python t26_part_b.py --r R26 [--tasks esc50 urban tau]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t26_core as K6  # noqa: E402

T, A = K6.T, K6.A
Q = Path.home() / "Documents/qmoe-results"
S2 = Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
S3B = Q / "2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db"
S3C = Q / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1"
OPTB = Q / "2026-10-23_2133_redesign-idea6-optionB_code-f320295"
TASKS = {"esc50": dict(seeds=[0, 1, 2, 3, 4], K=50, short=512), "urban": dict(seeds=[0, 1, 2], K=10, short=400),
         "tau": dict(seeds=[0, 1, 2], K=10, short=None)}


def sdir(task, tier, s):
    if tier in ("mn04", "mn10"):
        return S3B / f"scores_cnn/{task}/{tier}_{task}_s{s}/esc", f"{tier}_w8a8"
    if tier in ("mn05", "mn20", "dymn04"):
        return S3C / f"scores_cnn/{task}/{tier}_{task}_s{s}/esc", f"{tier}_w8a8"
    return S2 / f"scores_ast_short/{task}/astshort_{task}_s{s}/esc", "w4g128"


def bed_esc_urban(task):
    cfg = TASKS[task]
    seeds, K = cfg["seeds"], cfg["K"]
    tiers = ["mn04", "mn05", "mn10", "mn20", "dymn04", "ast"]
    cj = json.loads((S3C / f"analysis/{task}/costs.json").read_text())
    cost1 = {m: cj["cheap"][f"cnn:{m}:w8a8:esc"] for m in tiers if m != "ast"}
    cost1["ast"] = cj["ast"][f"w4g128@{cfg['short']}+esc"]
    SD = {(m, s): A.ScoreDir(sdir(task, m, s)[0]) for m in tiers for s in seeds}
    cfgn = {m: sdir(task, m, 0)[1] for m in tiers}
    get = lambda m, s, c: SD[(m, s)].get(cfgn[m], c)  # noqa: E731
    r0 = get("mn10", seeds[0], "clean")
    y, fold = r0["y"], r0["fold"]
    for m in tiers:
        for s in seeds:
            r = get(m, s, "clean")
            if not (np.array_equal(r["y"], y) and np.array_equal(r["fold"], fold) and r["crc"] == r0["crc"]):
                raise SystemExit(f"{task} {m} s{s}: rows differ from MN10-Q8's")
    folds = sorted(np.unique(fold).tolist())
    keys = [(s, f) for s in seeds for f in folds]
    n = len(y)
    ratio = float(np.mean([(fold == f).sum() / (n - (fold == f).sum()) for f in folds]))
    cc = lambda idx: {m: np.full(len(idx), v) for m, v in cost1.items()}  # noqa: E731

    def iv(s, f):
        idx, P = None, {}
        for m in tiers:
            i2, lg = get(m, s, "clean")["iv"][f]
            if idx is None:
                idx = i2
            elif not np.array_equal(i2, idx):
                raise SystemExit(f"{task} {m} s{s} f{f}: inner-validation rows differ")
            P[m] = A.softmax(lg)
        if np.intersect1d(idx, np.flatnonzero(fold == f)).size:
            raise SystemExit("inner validation overlaps test")
        return y[idx], P, cc(idx)

    def test(c, s, f):
        te = np.flatnonzero(fold == f)
        return y[te], {m: A.softmax(get(m, s, c)["z"][te]) for m in tiers}, cc(te)

    return K6.Bed(task, K, keys, iv, test, ["clean", "field"], ratio), cost1


def bed_tau():
    cfg = TASKS["tau"]
    seeds, K = cfg["seeds"], cfg["K"]
    files = {"mn10": "mn10_tau_s{}/esc/mn10_w8a8__all.npz", "mn20": "mn20_tau_s{}/esc/mn20_w8a8__all.npz",
             "ast": "ast_tau_s{}/esc/w4g128__all.npz"}
    Z = {(m, s): np.load(OPTB / "scores" / f.format(s)) for m, f in files.items() for s in seeds}
    z0 = Z[("mn10", seeds[0])]
    y, fold, dev = z0["y"].astype(np.int64), z0["fold"].astype(int), z0["device"].astype(str)
    for (m, s), z in Z.items():
        if not (np.array_equal(z["y"], y) and np.array_equal(z["fold"], fold) and int(z["names_crc32"]) == int(z0["names_crc32"])):
            raise SystemExit(f"tau {m} s{s}: rows differ")
    cj = json.loads((OPTB / "analysis/tau/costs.json").read_text())
    cost1 = {"mn10": cj["cheap"]["cnn:mn10:w8a8:esc"], "mn20": cj["cheap"]["cnn:mn20:w8a8:esc"], "ast": cj["ast"]["ast:w4g128+esc"]}
    folds = sorted(int(k[7:]) for k in z0.files if k.startswith("iv_idx_"))
    keys = [(s, f) for s in seeds for f in folds]
    a = dev == "a"
    ratio = float(np.mean([(a & (fold == f)).sum() / (a.sum() - (a & (fold == f)).sum()) for f in folds]))
    cc = lambda idx: {m: np.full(len(idx), v) for m, v in cost1.items()}  # noqa: E731

    def iv(s, f):
        idx = Z[("mn10", s)][f"iv_idx_{f}"]
        P = {}
        for m in files:
            if not np.array_equal(Z[(m, s)][f"iv_idx_{f}"], idx):
                raise SystemExit(f"tau {m} s{s} f{f}: inner-validation rows differ")
            P[m] = A.softmax(Z[(m, s)][f"iv_logits_{f}"].astype(np.float64))
        if np.intersect1d(idx, np.flatnonzero(fold == f)).size:
            raise SystemExit("inner validation overlaps test")
        return y[idx], P, cc(idx)

    def test(c, s, f):
        te = np.flatnonzero((fold == f) & (a if c == "clean" else ~a))
        return y[te], {m: A.softmax(Z[(m, s)]["test_logits"][te].astype(np.float64)) for m in files}, cc(te)

    return K6.Bed("tau", K, keys, iv, test, ["clean", "shift"], ratio), cost1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = a.r / ("_smoke" if a.smoke else "tables") / "b"
    out.mkdir(parents=True, exist_ok=True)
    allc, V = [], {"part": "B", "label": "DESCRIPTIVE (development data, seen in earlier rounds)", "tasks": {}}
    for task in a.tasks:
        bed, cost1 = bed_tau() if task == "tau" else bed_esc_urban(task)
        if a.smoke:
            K6.smoke_permute(bed)
        SEL, pick, th, sp_iv = K6.select_budgets(bed)
        TT, preds = K6.run_test(bed, th)
        CMP, ST = K6.compare_all(bed, TT, preds, systems=("s1", "s2"))
        SEL.to_csv(out / f"{task}_budget_selection.csv", index=False)
        TT.to_csv(out / f"{task}_by_seed_fold.csv", index=False)
        ST.to_csv(out / f"{task}_statics_by_condition.csv", index=False)
        allc.append(CMP)
        V["tasks"][task] = {"budgets": pick, "nb_ratio": bed.ratio, "seeds": sorted({k[0] for k in bed.keys}),
                            "n_folds": len({k[1] for k in bed.keys}), "costs_per_clip": cost1,
                            "statics_in_hull": sorted(ST.system.unique().tolist(), key=K6.WIDE.index)}
        print(task, "done", pick, flush=True)
    CMP = pd.concat(allc)
    CMP.to_csv(out / "compare_by_condition.csv", index=False)
    (out / "summary.json").write_text(json.dumps(V, indent=1, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(CMP[["dataset", "cond", "system", "f1", "cost", "rate", "hull", "hull_f1", "d_hull", "nb_lo_hull", "nb_hi_hull", "folds_pos_hull",
               "f1_mn10+mn20", "d_vs_mn10+mn20"]].round(4).to_string())


if __name__ == "__main__":
    main()
