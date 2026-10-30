"""Stress tests of the 5% budget cascades (n3g64 -> w8g64 and w3g64 -> w8g64), every dataset
and condition. Reads recompute.py outputs (OUT/bitops/<dataset>/) and the Stage 2 scores.

  multiplicity  Bonferroni and Holm over larger families (check 5)
  fold level    seeds share each fold's test clips: average the 5 (3) seeds within a fold,
                then t-test and Wilcoxon over the 5 (10) fold differences
  overhead      the same comparison when every group-wise configuration pays one extra
                rescale MAC per token, output row and group beyond the first, at B bit-ops
                (B = 0 is Stage 2's rule; 1024 is an fp32 MAC, Stage 2's own price for FP32)
  bits read     memory-bound view: weight bits read per clip, as Stage 2 counts them and with
                the nested cheap pass reading the 4th bit-plane its rounding needs
  hull          the best random time-sharing of two static systems at the cascade's mean
                bit-ops (expected macro F1 over 200 draws per (seed, fold))
  random gate   the same cascade escalating a random subset of the same size
  clean_b05     the 5% threshold set on clean inner validation and applied to every condition

    python stress.py --stage2 STAGE2_DIR --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute import (LIN_W, Run, backbone_bitops, f1_fast, head_bitops, margin, parent,  # noqa: E402
                       thr_rate)

ap = argparse.ArgumentParser()
ap.add_argument("--stage2", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--draws", type=int, default=200)
a = ap.parse_args()
DS = ["esc50", "esc5", "urban"]
CHEAP = ["n3g64", "w3g64"]
OTHER_BITS = 40_083_456          # every FP32 parameter outside the 72 Linear layers, in bits
rng = np.random.default_rng(12345)


def bits_read_static(cfg):
    if cfg == "fp32":
        return 32 * LIN_W + OTHER_BITS + 0.0
    k = int(cfg[1])
    g = int(cfg.split("g")[1]) if "g" in cfg else None
    scales = 32 * (LIN_W // g if g else 82_944)
    return float(k * LIN_W + scales + OTHER_BITS)


def paired(x, z, keys):
    d = x - z
    folds = np.array([k[1] for k in keys])
    fm = np.array([d[folds == f].mean() for f in np.unique(folds)])
    return dict(diff_pts=100 * d.mean(), p_t=stats.ttest_rel(x, z).pvalue, p_w=stats.wilcoxon(x, z).pvalue,
                n_pairs=len(d), n_folds=len(fm), fold_diffs_pts=(100 * fm).round(3).tolist(),
                p_t_fold=stats.ttest_1samp(fm, 0).pvalue, p_w_fold=stats.wilcoxon(fm).pvalue)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


out = a.out
res_rows = []

# ---------------------------------------------------------------- multiplicity
allp = []
for ds in DS:
    P = pd.read_csv(out / "bitops" / ds / "points.csv")
    P["dataset"] = ds
    allp.append(P)
AP = pd.concat(allp, ignore_index=True)
# Stage 2's every-family comparisons (primary verified equal to the audit's to 3e-15)
fam_rows = []
for ds in DS:
    fc = pd.read_csv(a.stage2 / "analysis" / ds / "front_comparisons.csv")
    fc = fc[fc.p_ttest.notna()]
    fam_rows.append(fc[["dataset", "family", "method", "operating_point", "p_ttest", "p_wilcoxon", "mean_diff"]])
for d in sorted((a.stage2 / "analysis").glob("*_cheaphead")):
    for c in (d / "front").glob("*/comparisons.csv"):
        fc = pd.read_csv(c)
        fc = fc[fc.p_ttest.notna()] if "p_ttest" in fc else fc.iloc[0:0]
        fc["family"] = "cheaphead"
        fam_rows.append(fc[["dataset", "family", "method", "operating_point", "p_ttest", "p_wilcoxon", "mean_diff"]])
FAM = pd.concat(fam_rows, ignore_index=True)
fam_counts = FAM.groupby("family").size().to_dict()
n_all = len(FAM)
n_primary_all = len(AP)
AP["holm_t_primary_all"] = holm(AP.p_t)
AP["holm_w_primary_all"] = holm(AP.p_w)
# Holm over every computed comparison: rank the primary p values inside the full list
pt_full = np.concatenate([AP.p_t.to_numpy(), FAM[FAM.family != "primary"].p_ttest.to_numpy()])
pw_full = np.concatenate([AP.p_w.to_numpy(), FAM[FAM.family != "primary"].p_wilcoxon.to_numpy()])
AP["holm_t_everything"] = holm(pt_full)[:len(AP)]
AP["holm_w_everything"] = holm(pw_full)[:len(AP)]
m_ds = AP.groupby("dataset").size().to_dict()


def verdict(diff, pt, pw):
    return "PASS" if diff > 0 and pt < 0.05 and pw < 0.05 else "FAIL"


mult = []
for _, r in AP[(AP.point == "b05") & AP.cheap.isin(CHEAP)].iterrows():
    mult.append(dict(
        dataset=r.dataset, cond=r.cond, cheap=r.cheap, f1=r.f1, bitops_T=r.bitops / 1e12, comparator=r.comparator,
        comp_f1=r.comp_f1, diff_pts=r.diff_pts,
        bonf30=verdict(r.diff_pts, min(1, 30 * r.p_t), min(1, 30 * r.p_w)),
        bonf_dataset=verdict(r.diff_pts, min(1, m_ds[r.dataset] * r.p_t), min(1, m_ds[r.dataset] * r.p_w)),
        bonf_all_primary=verdict(r.diff_pts, min(1, n_primary_all * r.p_t), min(1, n_primary_all * r.p_w)),
        holm_all_primary=verdict(r.diff_pts, r.holm_t_primary_all, r.holm_w_primary_all),
        bonf_everything=verdict(r.diff_pts, min(1, n_all * r.p_t), min(1, n_all * r.p_w)),
        holm_everything=verdict(r.diff_pts, r.holm_t_everything, r.holm_w_everything),
        fold_t_bonf30=verdict(r.diff_pts, min(1, 30 * r.p_t_fold), 0.0),
        fold_t_and_w_bonf30=verdict(r.diff_pts, min(1, 30 * r.p_t_fold), min(1, 30 * r.p_w_fold)),
        p_t=r.p_t, p_w=r.p_w, p_t_fold=r.p_t_fold, p_w_fold=r.p_w_fold))
M = pd.DataFrame(mult)
M.to_csv(out / "stress_multiplicity.csv", index=False)

# ------------------------------------------------- overhead and bits read (cost views)
cost_rows = []
for ds in DS:
    S = pd.read_csv(out / "bitops" / ds / "static_pairs.csv")
    C = pd.read_csv(out / "bitops" / ds / "cascade_pairs.csv")
    n_cls = {"esc50": 50, "esc5": 5, "urban": 10}[ds]
    for view in ["bitops_B0", "bitops_B256", "bitops_B512", "bitops_B1024", "bitsread_stage2", "bitsread_rounding"]:
        if view.startswith("bitops"):
            B = float(view.split("B")[1])
            S["cost"] = [backbone_bitops(c, B) + head_bitops(h, n_cls) for c, h in zip(S.config, S["head"])]
            C["cost"] = [backbone_bitops(c, B) + head_bitops("esc", n_cls) + r * (backbone_bitops(parent(c), B)
                         + head_bitops("esc", n_cls)) for c, r in zip(C.cheap, C.rate)]
        else:
            S["cost"] = [bits_read_static(c) for c in S.config]
            extra = 1 if view == "bitsread_rounding" else 0
            cc = []
            for c, r in zip(C.cheap, C.rate):
                k = int(c[1])
                if c.startswith("n"):          # nested: one stored parent, read planes on demand
                    cc.append(bits_read_static(c) + extra * LIN_W + r * (8 - k - extra) * LIN_W)
                else:                           # two stored models
                    cc.append(bits_read_static(c) + r * bits_read_static(parent(c)))
            C["cost"] = cc
        for cond in sorted(C.cond.unique()):
            s = S[S.cond == cond]
            sp = s.groupby(["config", "head"]).agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
            for ch in CHEAP:
                g = C[(C.cond == cond) & (C.cheap == ch) & (C.point == "b05")]
                if g.empty:
                    continue
                c = g.cost.mean()
                ok = sp[sp.cost <= c * (1 + 1e-9)].sort_values(["f1", "cost"], ascending=[False, True])
                comp = ok.iloc[0]
                gs = s[(s.config == comp.config) & (s["head"] == comp["head"])].set_index(["seed", "fold"]).f1
                gc = g.set_index(["seed", "fold"]).f1
                keys = sorted(set(gs.index) & set(gc.index))
                t = paired(gc.loc[keys].to_numpy(), gs.loc[keys].to_numpy(), keys)
                cost_rows.append(dict(dataset=ds, view=view, cond=cond, cheap=ch, f1=g.f1.mean(), cost=c,
                                      comparator=f"{comp.config}+{comp['head']}", comp_f1=comp.f1,
                                      comp_cost=comp.cost, diff_pts=t["diff_pts"],
                                      verdict_bonf30=verdict(t["diff_pts"], min(1, 30 * t["p_t"]),
                                                             min(1, 30 * t["p_w"]))))
CV = pd.DataFrame(cost_rows)
CV.to_csv(out / "stress_cost_views.csv", index=False)

# ----------------------------------------- hull, random gate, clean-calibrated b05
s2 = a.stage2 / "scores"
hull_rows, gate_rows, cb_rows = [], [], []
for ds in DS:
    S = pd.read_csv(out / "bitops" / ds / "static_pairs.csv")
    C = pd.read_csv(out / "bitops" / ds / "cascade_pairs.csv")
    n_cls = {"esc50": 50, "esc5": 5, "urban": 10}[ds]
    tags = sorted(p.name for p in (s2 / ds).iterdir() if p.is_dir() and "headw4" not in p.name)
    runs = {(int(t.split("_s")[1]), h): Run(s2 / ds / t / h) for t in tags for h in ("esc", "4")}
    for cond in sorted(C.cond.unique()):
        s = S[S.cond == cond]
        sp = s.groupby(["config", "head"]).agg(f1=("f1", "mean"), cost=("bitops", "mean")).reset_index()
        for ch in CHEAP:
            g = C[(C.cond == cond) & (C.cheap == ch) & (C.point == "b05")]
            if g.empty:
                continue
            c = g.bitops.mean()
            # best interpolation of two static systems at cost c (or one system at <= c)
            best = None
            for _, lo in sp[sp.cost <= c].iterrows():
                cand = [(lo, lo, 0.0)] + [(lo, hi, (c - lo.cost) / (hi.cost - lo.cost))
                                           for _, hi in sp[sp.cost > c].iterrows()]
                for x, y, lam in cand:
                    v = (1 - lam) * x.f1 + lam * y.f1
                    if best is None or v > best[0]:
                        best = (v, x, y, lam)
            v, lo, hi, lam = best
            mix, casc, rgate, keys = [], [], [], []
            for (seed, fold), gg in g.groupby(["seed", "fold"]):
                rl = runs[(seed, lo["head"])].get(lo.config, cond)
                rh = runs[(seed, hi["head"])].get(hi.config, cond)
                te = np.flatnonzero(rl["tf"] == fold)
                y = rl["y"][te]
                pl, ph = rl["p"][te].argmax(1), rh["p"][te].argmax(1)
                draws = rng.random((a.draws, len(te))) < lam
                mix.append(np.mean([f1_fast(y, np.where(d, ph, pl), n_cls) for d in draws]))
                casc.append(float(gg.f1.iloc[0]))
                # random gate: same number escalated as the margin gate in this (seed, fold)
                rc = runs[(seed, "esc")].get(ch, cond)
                rp = runs[(seed, "esc")].get(parent(ch), cond)
                pc, pp = rc["p"][te].argmax(1), rp["p"][te].argmax(1)
                k = int(round(float(gg.rate.iloc[0]) * len(te)))
                rg = []
                for _ in range(a.draws):
                    e = np.zeros(len(te), bool)
                    e[rng.choice(len(te), k, replace=False)] = True
                    rg.append(f1_fast(y, np.where(e, pp, pc), n_cls))
                rgate.append(np.mean(rg))
                keys.append((seed, fold))
            casc, mix, rgate = map(np.array, (casc, mix, rgate))
            t = paired(casc, mix, keys)
            hull_rows.append(dict(dataset=ds, cond=cond, cheap=ch, f1=casc.mean(), bitops_T=c / 1e12,
                                  hull_lo=f"{lo.config}+{lo['head']}", hull_hi=f"{hi.config}+{hi['head']}",
                                  lam=lam, hull_f1_interp=v, hull_f1_sim=mix.mean(), **t,
                                  verdict_bonf30=verdict(t["diff_pts"], min(1, 30 * t["p_t"]), min(1, 30 * t["p_w"]))))
            t2 = paired(casc, rgate, keys)
            gate_rows.append(dict(dataset=ds, cond=cond, cheap=ch, f1=casc.mean(), random_gate_f1=rgate.mean(),
                                  gate_gain_pts=t2["diff_pts"], p_t=t2["p_t"], p_w=t2["p_w"]))
        # clean-calibrated 5% threshold (the deployable rule) for each cheap config
        for ch in CHEAP:
            par = parent(ch)
            rows = []
            for seed in sorted({k[0] for k in runs}):
                run = runs[(seed, "esc")]
                if not (run.exists(ch, cond) and run.exists(par, cond)):
                    continue
                rc, rp, rcl = run.get(ch, cond), run.get(par, cond), run.get(ch, "clean")
                for fold in sorted(rc["iv"]):
                    t_ = thr_rate(margin(rcl["iv"][fold][1]), 0.05)
                    te = np.flatnonzero(rc["tf"] == fold)
                    e = margin(rc["p"][te]) < t_
                    pred = np.where(e, rp["p"][te].argmax(1), rc["p"][te].argmax(1))
                    from sklearn.metrics import f1_score
                    rows.append(dict(seed=seed, fold=fold, f1=f1_score(rc["y"][te], pred, average="macro",
                                                                        zero_division=0), rate=e.mean(),
                                     bitops=backbone_bitops(ch) + head_bitops("esc", n_cls)
                                     + e.mean() * (backbone_bitops(par) + head_bitops("esc", n_cls))))
            g = pd.DataFrame(rows)
            c = g.bitops.mean()
            ok = sp[sp.cost <= c * (1 + 1e-9)].sort_values(["f1", "cost"], ascending=[False, True])
            comp = ok.iloc[0]
            gs = s[(s.config == comp.config) & (s["head"] == comp["head"])].set_index(["seed", "fold"]).f1
            gc = g.set_index(["seed", "fold"]).f1
            keys = sorted(set(gs.index) & set(gc.index))
            t = paired(gc.loc[keys].to_numpy(), gs.loc[keys].to_numpy(), keys)
            cb_rows.append(dict(dataset=ds, cond=cond, cheap=ch, f1=g.f1.mean(), rate=g.rate.mean(),
                                bitops_T=c / 1e12, comparator=f"{comp.config}+{comp['head']}", comp_f1=comp.f1,
                                diff_pts=t["diff_pts"], p_t=t["p_t"], p_w=t["p_w"],
                                verdict_bonf30=verdict(t["diff_pts"], min(1, 30 * t["p_t"]), min(1, 30 * t["p_w"]))))
        print(ds, cond, "done", flush=True)
pd.DataFrame(hull_rows).to_csv(out / "stress_hull.csv", index=False)
pd.DataFrame(gate_rows).to_csv(out / "stress_random_gate.csv", index=False)
pd.DataFrame(cb_rows).to_csv(out / "stress_clean_b05.csv", index=False)
json.dump({"family_counts_stage2_comparisons": fam_counts, "n_everything": n_all, "n_primary_all": n_primary_all,
           "n_primary_per_dataset": m_ds}, open(out / "stress_families.json", "w"), indent=1)
print(json.dumps({"family_counts": fam_counts, "n_everything": n_all, "n_primary_all": n_primary_all, "m_ds": m_ds}))
