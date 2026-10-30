"""T25 analysis (ANALYSIS_PLAN.md C3 to C6): the fixed T21 cascade against the equal-cost hull widened to every stage 3c
CNN at simulated 8 bits and every pairwise average of two of the five CNNs.

The cascade is rebuilt from T21's stored score files with T21's scalers and budget (read from T21's tables, never
re-chosen) and must reproduce T21's tables/<d>/primary_by_seed_fold.csv to 1e-9 for every (condition, seed, fold)
before any new number is computed. The hull is T21's code path (stage 2's hull_pair over the mean test F1 and mean
cost of each static on the condition, the per-(seed, fold) F1 of the hull as the exact expected macro F1 of random
time-sharing between its two endpoints, Nadeau-Bengio with T21's ratio, T21's clip-level bootstrap) with the systems
list widened.

    python t25_analyze.py --r R25 --base R21 --d d1             # one dataset
    python t25_analyze.py --r R25 --combine d1 [d3]             # Holm over the datasets run
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t21"))
import t21_analyze as T  # noqa: E402  T21's analysis module (stage 2 helpers A, holm, exp_f1, nb_ci, f1_rows, Scores)

A, holm = T.A, T.holm
from QWave import efficientat_dymn as DY  # noqa: E402
from QWave import efficientat_mn as M  # noqa: E402

CNN5 = ["mn04", "mn05", "mn10", "mn20", "dymn04"]
NEW = ["mn04", "mn05", "mn20", "dymn04"]
PAIRS = [f"{a}+{b}" for a, b in itertools.combinations(CNN5, 2)]
NARROW = ["mn10", "ast", "avg"]
WIDE = NARROW + NEW + PAIRS  # 17 systems
FAMILY = 21
WEIGHTS = Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1"
WFILE = {"mn04": "mn04_as_mAP_432.pt", "mn05": "mn05_as_mAP_443.pt", "mn20": "mn20_as_mAP_478.pt", "dymn04": "dymn04_as.pt"}


def lib_of(m):
    return DY if m.startswith("dymn") else M


def cnn_costs(R, base, D, K):
    """Per-clip priced bit-ops and MACs of each new CNN at W8A8 on T21's clip lengths: MN front end at 32 x 32,
    backbone W8A8 with rescales at 32 x 32 (DyMN: kernel aggregation at 32 x 32 as stage 3c), the FP32 head."""
    L = np.load(base / "emb" / D / "mn_pool" / "lengths__clean.npy")
    L25 = np.load(R / "emb" / D / "cnn_pool" / "lengths__clean.npy")
    if not np.array_equal(L, L25):
        raise SystemExit("T25 extraction read different clip lengths than T21")
    cfg = OmegaConf.load(T.ROOT / "config/esc50.yaml")
    fm = np.array([M.frontend_macs(int(x)) for x in L])
    fr = np.array([M.frames_of(M.n32_of(int(x))) for x in L])
    out = {"frontend": (fm * 1024.0, fm)}
    for m in NEW:
        lib = lib_of(m)
        b = lib.load_dymn(m, WEIGHTS / WFILE[m]) if m.startswith("dymn") else M.load_mn(m, WEIGHTS / WFILE[m])
        with contextlib.redirect_stdout(io.StringIO()):
            hm, hb = T.net_cost(T.build_model("esc", b.emb_dim, K, cfg))
        cache = {f: lib.feature_macs(b, int(f)) for f in np.unique(fr)}
        bits = np.array([lib.backbone_bitops({"aggregate_macs": 0.0, **cache[f]}, "w8a8", 1024.0) for f in fr])
        macs = np.array([cache[f]["macs"] + cache[f].get("aggregate_macs", 0.0) for f in fr])
        out[m] = (fm * 1024.0 + bits + hb, fm + macs + hm)
    return out


def load_t21_design(base, D):
    sc = pd.read_csv(base / "tables" / D / "primary_scaler_choice.csv").set_index("tier").chosen
    V = json.loads((base / "tables" / D / "verdict.json").read_text())
    return sc["cheap"], sc["ast"], V["design"]["budget"], V


def analyze(R, base, D, seeds, out):
    K = int(pd.read_csv(base / "csv" / D / "pool.csv").class_id.nunique())
    ks, ka, bstar, V21 = load_t21_design(base, D)
    C = T.Scores(base, D, "pool", "mn10_w8a8", seeds)
    S = T.Scores(base, D, "pool", "ast_w4g128", seeds)
    N = {m: T.Scores(R, D, "pool", f"{m}_w8a8", seeds) for m in NEW}
    y = C.y
    for nm, Z in [("ast", S), *N.items()]:
        if not (np.array_equal(Z.y, y) and np.array_equal(Z.fold, C.fold)):
            raise SystemExit(f"{nm}: rows or folds differ from T21's MN10")
    keys = [(s, f) for s in seeds for f in C.folds]
    # ── costs ──
    c21, _ = T.per_clip_costs(base, D, K)
    cc, ca = c21["priced"]
    mc, ma = c21["macs"]
    cn = cnn_costs(R, base, D, K)
    cost = {"mn10": cc, "ast": ca, **{m: cn[m][0] for m in NEW}}
    macs = {"mn10": mc, "ast": ma, **{m: cn[m][1] for m in NEW}}
    fe_b, fe_m = cn["frontend"]
    # ── scaler per new tier on clean inner validation (T21's rule: refit only if strictly higher) ──
    scaler, sc_rows = {"mn10": ks, "ast": ka}, []
    for m, Z in N.items():
        v = {}
        for k in ("fp32fit", "refit"):
            vals = []
            for s, f in keys:
                iv, p = Z.iv(s, f, "clean", k)
                iv0, _ = C.iv(s, f, "clean", ks)
                if not np.array_equal(iv, iv0):
                    raise SystemExit(f"{m} s{s} f{f}: inner-validation rows differ from T21's")
                vals.append(A.f1_fast(y[iv], p.argmax(1), K))
            v[k] = float(np.mean(vals)) * 100
        scaler[m] = "refit" if v["refit"] > v["fp32fit"] else "fp32fit"
        sc_rows.append({"tier": m, "iv_f1_fp32fit": v["fp32fit"], "iv_f1_refit": v["refit"], "chosen": scaler[m]})
    pd.DataFrame(sc_rows).to_csv(out / "t25_scaler_choice.csv", index=False)
    TIER = {"mn10": C, "ast": S, **N}
    # ── T21 thresholds (budget fixed by T21) ──
    rate = T.BUDGETS[bstar]
    thr = {}
    for s, f in keys:
        _, pc = C.iv(s, f, "clean", ks)
        thr[(s, f)] = A.thr_rate(A.margin(pc), rate)
    # ── best pair on inner validation of each condition family (pre-registered secondary) ──
    best_pair, bp_rows = {}, []
    for fam in ("clean", "field"):
        iv_f1 = {}
        for pname in PAIRS:
            a_, b_ = pname.split("+")
            vals = []
            for s, f in keys:
                iv, pa = TIER[a_].iv(s, f, fam, scaler[a_])
                _, pb = TIER[b_].iv(s, f, fam, scaler[b_])
                ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
                vals.append(A.f1_fast(y[iv][ok], ((pa + pb) / 2)[ok].argmax(1), K))
            iv_f1[pname] = float(np.mean(vals)) * 100
            bp_rows.append({"family": fam, "pair": pname, "iv_f1": iv_f1[pname]})
        best_pair[fam] = max(iv_f1, key=iv_f1.get)
    # ── test: every system per (condition, seed, fold) ──
    ref = pd.read_csv(base / "tables" / D / "primary_by_seed_fold.csv")
    ref = ref[ref.system.isin(["mn", "ast", "avg", "cascade"])].set_index(["cond", "seed", "fold", "system"])
    rows, preds = [], {}
    worst = 0.0
    for c in T.CONDS:
        for s, f in keys:
            te = np.flatnonzero(C.fold == f)
            P = {nm: TIER[nm].test(s, c, scaler[nm])[te] for nm in ["mn10", "ast", *NEW]}
            for nm, p in P.items():
                if not np.isfinite(p).all():
                    raise SystemExit(f"{nm} {c} s{s} f{f}: missing test logits")
            yt = y[te]
            esc = A.margin(P["mn10"]) < thr[(s, f)]
            pr = {nm: p.argmax(1) for nm, p in P.items()}
            pr["avg"] = ((P["mn10"] + P["ast"]) / 2).argmax(1)
            pr["cascade"] = np.where(esc, pr["avg"], pr["mn10"])
            cst = {nm: cost[nm][te].mean() for nm in ["mn10", "ast", *NEW]}
            mcs = {nm: macs[nm][te].mean() for nm in ["mn10", "ast", *NEW]}
            cst["avg"], mcs["avg"] = cst["mn10"] + cst["ast"], mcs["mn10"] + mcs["ast"]
            cst["cascade"] = float(np.mean(cc[te] + esc * ca[te]))
            mcs["cascade"] = float(np.mean(mc[te] + esc * ma[te]))
            shared = {}
            for pname in PAIRS:
                a_, b_ = pname.split("+")
                pr[pname] = ((P[a_] + P[b_]) / 2).argmax(1)
                cst[pname], mcs[pname] = cst[a_] + cst[b_], mcs[a_] + mcs[b_]
                shared[pname] = cst[pname] - fe_b[te].mean()
            for nm in pr:
                rows.append({"cond": c, "seed": s, "fold": f, "system": nm, "f1": A.f1_fast(yt, pr[nm], K) * 100,
                             "cost": cst[nm], "macs": mcs[nm], "cost_shared_frontend": shared.get(nm, cst[nm]),
                             "rate": float(esc.mean()) if nm == "cascade" else np.nan})
            for nm21, nm in (("mn", "mn10"), ("ast", "ast"), ("avg", "avg"), ("cascade", "cascade")):
                r21 = ref.loc[(c, s, f, nm21)]
                d_ = max(abs(r21.f1 - rows[-len(pr) + list(pr).index(nm)]["f1"]), abs(r21.cost - cst[nm]) / r21.cost)
                if nm == "cascade":
                    d_ = max(d_, abs(r21.rate - float(esc.mean())))
                worst = max(worst, d_)
                if d_ > 1e-9:
                    raise SystemExit(f"T21 reproduction failed at {c} s{s} f{f} {nm}: {d_}")
            preds[(c, s, f)] = (yt, pr)
    print(f"T21 cascade, MN10, AST and always-average reproduced on every (cond, seed, fold); worst diff {worst:.2e}", flush=True)
    TT = pd.DataFrame(rows)
    TT.to_csv(out / "t25_by_seed_fold.csv", index=False)
    # NB ratio exactly as T21
    fold_n = {f: int((C.fold == f).sum()) for f in C.folds}
    ratio = float(np.mean([fold_n[f] / (len(y) - fold_n[f] - (len(y) - sum(fold_n.values()))) for f in C.folds]))
    res, stat_rows = [], []
    for c in T.CONDS:
        g = TT[TT.cond == c]
        sp_all = g.groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean"), macs=("macs", "mean"),
                                         cost_shared=("cost_shared_frontend", "mean")).reset_index()
        for r_ in sp_all.itertuples():
            stat_rows.append({"cond": c, "system": r_.system, "f1": r_.f1, "cost": r_.cost, "macs": r_.macs,
                              "cost_shared_frontend": r_.cost_shared})
        xc = g[g.system == "cascade"].set_index(["seed", "fold"]).sort_index()
        Cc, Cm = float(xc.cost.mean()), float(xc.macs.mean())
        vals = lambda nm: g[g.system == nm].set_index(["seed", "fold"]).sort_index().f1  # noqa: E731
        r = {"cond": c, "budget": bstar, "cascade_f1": xc.f1.mean(), "cascade_cost": Cc, "cascade_macs": Cm, "rate": xc.rate.mean()}

        def hull_of(systems, costcol, at):
            sp = sp_all[sp_all.system.isin(systems)][["system", "f1", costcol]].rename(columns={costcol: "cost"}).copy()
            sp["f1"] /= 100
            v, lo, hi, lam = A.hull_pair(sp, at)
            h = {}
            for s, f in keys:
                yt, pr = preds[(c, s, f)]
                h[(s, f)] = (A.f1_fast(yt, pr[lo], K) if lam == 0 else T.exp_f1(yt, pr[lo], pr[hi], lam, K)) * 100
            return np.array([h[k_] for k_ in xc.index]), lo, hi, lam

        def compare(tag, z):
            d = xc.f1.to_numpy() - z
            fm = pd.Series(d, index=xc.index).groupby(level="fold").mean().to_numpy()
            t = A.tests(xc.f1.to_numpy(), z, list(xc.index), ratio)
            lo_, hi_ = T.nb_ci(fm, ratio)
            r.update({f"d_{tag}": float(fm.mean()), f"p_nb_{tag}": t["p_nb"], f"p_signflip_{tag}": t["p_signflip"],
                      f"nb_lo_{tag}": lo_, f"nb_hi_{tag}": hi_, f"folds_pos_{tag}": int((fm > 0).sum()),
                      f"fold_d_{tag}": json.dumps(np.round(fm, 3).tolist())})

        hv_n, lo_n, hi_n, lam_n = hull_of(NARROW, "cost", Cc)
        compare("narrow", hv_n)
        r.update({"hull_narrow": f"{lo_n}|{hi_n}|{lam_n:.4f}", "hull_narrow_f1": hv_n.mean()})
        hv, lo, hi, lam = hull_of(WIDE, "cost", Cc)
        compare("wide", hv)
        r.update({"hull_wide": f"{lo}|{hi}|{lam:.4f}", "hull_wide_f1": hv.mean(), "hull_wide_lo": lo, "hull_wide_hi": hi, "hull_wide_lam": lam})
        # descriptive sensitivities (not tests of the plan's rules): pairs with one shared front end; unpriced MACs
        tmp = sp_all.copy()
        sp_all["cost_sh"] = np.where(sp_all.system.isin(PAIRS), sp_all.cost_shared, sp_all.cost)
        hs, los, his, lams = hull_of(WIDE, "cost_sh", Cc)
        compare("wide_sharedfe", hs)
        r["hull_wide_sharedfe"] = f"{los}|{his}|{lams:.4f}"
        hm, lom, him, lamm = hull_of(WIDE, "macs", Cm)
        compare("wide_macs", hm)
        r["hull_wide_macs"] = f"{lom}|{him}|{lamm:.4f}"
        sp_all = tmp
        # best two-CNN average chosen on inner validation of the condition family, at its own cost
        fam = "field" if c == "field" else "clean"
        bp = best_pair[fam]
        compare("bestpair", vals(bp).to_numpy())
        r.update({"best_pair_iv": bp, "best_pair_f1": vals(bp).mean(),
                  "best_pair_cost": float(g[g.system == bp].cost.mean())})
        pt = sp_all[sp_all.system.isin(PAIRS)].sort_values("f1", ascending=False).iloc[0]
        r.update({"best_pair_test_descriptive": pt.system, "best_pair_test_f1": pt.f1, "best_pair_test_cost": pt.cost})
        if c in ("field", "clean"):  # T21's clip-level bootstrap of cascade minus the widened hull
            bs = np.random.default_rng(T.BOOT_SEED)
            acc = np.zeros(T.NB_BOOT)
            for f in C.folds:
                n_f = fold_n[f]
                I = bs.integers(0, n_f, (T.NB_BOOT, n_f))
                dsum = np.zeros(T.NB_BOOT)
                for s in seeds:
                    yt, pr = preds[(c, s, f)]
                    Y = yt[I]
                    fc = T.f1_rows(Y, pr["cascade"][I], K)
                    fh = (1 - lam) * T.f1_rows(Y, pr[lo][I], K) + (lam * T.f1_rows(Y, pr[hi][I], K) if lam > 0 else 0)
                    dsum += fc - fh
                acc += dsum / len(seeds)
            acc = acc / len(C.folds) * 100
            r["boot_lo_wide"], r["boot_hi_wide"] = float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))
        res.append(r)
    RES = pd.DataFrame(res)
    RES.to_csv(out / "t25_by_condition.csv", index=False)
    pd.DataFrame(stat_rows).to_csv(out / "t25_statics_by_condition.csv", index=False)
    pd.DataFrame(bp_rows).to_csv(out / "t25_pair_iv_selection.csv", index=False)
    # narrow-hull check against T21's own verdict
    fr, cl = RES[RES.cond == "field"].iloc[0], RES[RES.cond == "clean"].iloc[0]
    for bed, rr, key in (("field", fr, "H1"), ("clean", cl, "H2")):
        if abs(rr.d_narrow - V21[key]["d"]) > 1e-9:
            raise SystemExit(f"narrow hull on {bed} gives {rr.d_narrow}, T21 reported {V21[key]['d']}")
    print("T21's narrow-hull gains reproduced (field and clean)", flush=True)
    p1 = float(fr.p_nb_wide)
    V = {"dataset": D, "base": str(base), "seeds": seeds, "nb_ratio": ratio, "scalers": scaler, "budget": bstar,
         "H1_wide": {"rule": "field mix: cascade minus the widened equal-cost hull >= 1.0 macro F1, Nadeau-Bengio two-sided, Holm over datasets < 0.05",
                     "d": float(fr.d_wide), "nb_ci95": [fr.nb_lo_wide, fr.nb_hi_wide], "boot_ci95": [fr.boot_lo_wide, fr.boot_hi_wide],
                     "p_nb": p1, "p_family21": min(1.0, FAMILY * p1), "p_signflip": float(fr.p_signflip_wide),
                     "folds_positive": int(fr.folds_pos_wide), "n_folds": len(C.folds), "fold_d": fr.fold_d_wide,
                     "hull": fr.hull_wide, "hull_f1": float(fr.hull_wide_f1), "cascade_f1": float(fr.cascade_f1),
                     "cascade_cost": float(fr.cascade_cost)},
         "H2_wide": {"rule": "same on clean audio", "d": float(cl.d_wide), "nb_ci95": [cl.nb_lo_wide, cl.nb_hi_wide],
                     "boot_ci95": [cl.boot_lo_wide, cl.boot_hi_wide], "p_nb": float(cl.p_nb_wide),
                     "p_family21": min(1.0, FAMILY * float(cl.p_nb_wide)), "folds_positive": int(cl.folds_pos_wide),
                     "hull": cl.hull_wide, "passes": bool(cl.d_wide >= 1.0 and cl.p_nb_wide < 0.05)},
         "best_pair": {bed: {"pair_chosen_on_iv": rr.best_pair_iv, "pair_f1": float(rr.best_pair_f1), "pair_cost": float(rr.best_pair_cost),
                             "cascade_f1": float(rr.cascade_f1), "cascade_cost": float(rr.cascade_cost), "d": float(rr.d_bestpair),
                             "nb_ci95": [rr.nb_lo_bestpair, rr.nb_hi_bestpair], "p_nb": float(rr.p_nb_bestpair),
                             "folds_positive": int(rr.folds_pos_bestpair), "best_pair_on_test_descriptive": rr.best_pair_test_descriptive}
                       for bed, rr in (("field", fr), ("clean", cl))},
         "narrow_reproduced": {"field": float(fr.d_narrow), "clean": float(cl.d_narrow)}}
    (out / "verdict.json").write_text(json.dumps(V, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(RES[["cond", "cascade_f1", "cascade_cost", "hull_narrow_f1", "d_narrow", "hull_wide", "hull_wide_f1", "d_wide", "p_nb_wide",
               "folds_pos_wide", "best_pair_iv", "best_pair_f1", "d_bestpair"]].round(4).to_string())
    return V


def combine(R, ds):
    V = {d: json.loads((R / "tables" / d / "verdict.json").read_text()) for d in ds}
    ph = holm(np.array([V[d]["H1_wide"]["p_nb"] for d in ds]))
    out = {"datasets_in_holm": ds, "family": FAMILY, "H1_wide": {}}
    for d, p in zip(ds, ph):
        h = V[d]["H1_wide"]
        out["H1_wide"][d] = {"d": h["d"], "nb_ci95": h["nb_ci95"], "p_nb": h["p_nb"], "p_holm": float(p),
                             "p_family21": h["p_family21"], "folds_positive": h["folds_positive"],
                             "passes": bool(h["d"] >= 1.0 and p < 0.05)}
    out["H2_wide"] = {d: V[d]["H2_wide"] for d in ds}
    (R / "tables" / "verdict_combined.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=Path.home() / "Documents/qmoe-results/2026-10-26_1408_moe-t21_code-731570e")
    ap.add_argument("--d", default="d1")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--combine", nargs="*", default=None)
    a = ap.parse_args()
    if a.combine is not None:
        return combine(a.r, a.combine)
    out = a.r / "tables" / a.d
    out.mkdir(parents=True, exist_ok=True)
    analyze(a.r, a.base, a.d, a.seeds, out)


if __name__ == "__main__":
    main()
