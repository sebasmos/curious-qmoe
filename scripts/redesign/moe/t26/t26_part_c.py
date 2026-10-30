"""T26 Part C (ANALYSIS_PLAN.md C7), PRE-REGISTERED: FSDnoisy18k (d3) and TUT Acoustic Scenes 2017 (d4) with T24's
folds, fold models, test set, mixer, MN10-Q8 and AST-Q4 scores, and T26's MN20-Q8 (and, where the precondition
passes, MN04, MN05, DyMN04) scored by T24's code path.

    python t26_part_c.py --r R26 --d d3 --seeds 0 1 2 --precondition     # MN20-Q8 minus MN10-Q8 on the field mix
    python t26_part_c.py --r R26 --d d3 --seeds 0 1 2                    # H1-C, S2, comparators (needs all tiers)
    python t26_part_c.py --r R26 --combine d3 d4                         # Holm over the datasets that pass
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "t24"))
import t26_core as K6  # noqa: E402
import t24_analyze as T24  # noqa: E402

T, A = K6.T, K6.A
from QWave import efficientat_dymn as DY  # noqa: E402
from QWave import efficientat_mn as M  # noqa: E402

R24 = Path.home() / "Documents/qmoe-results/2026-10-28_1827_moe-t24_code-9a51cf5"
NEW = ["mn04", "mn05", "mn20", "dymn04"]
WEIGHTS = Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1"
WFILE = {"mn04": "mn04_as_mAP_432.pt", "mn05": "mn05_as_mAP_443.pt", "mn20": "mn20_as_mAP_478.pt", "dymn04": "dymn04_as.pt"}
PRE_BAR = 1.0


def cnn_costs(R, D, K, models):
    """T25's per-clip pricing (front end at 32 x 32, W8A8 backbone with rescales at 32 x 32, DyMN kernel aggregation
    as stage 3c, the FP32 head) on the clip lengths of all.csv, which must equal T24's MN10 lengths."""
    L = np.load(R24 / "emb" / D / "mn_all" / "lengths__clean.npy")
    if not np.array_equal(L, np.load(R / "emb" / D / "cnn_all" / "lengths__clean.npy")):
        raise SystemExit("T26 extraction read different clip lengths than T24")
    cfg = OmegaConf.load(T.ROOT / "config/esc50.yaml")
    fm = np.array([M.frontend_macs(int(x)) for x in L])
    fr = np.array([M.frames_of(M.n32_of(int(x))) for x in L])
    out = {"frontend": fm * 1024.0, "_macs": {}}
    for m in models:
        lib = DY if m.startswith("dymn") else M
        b = DY.load_dymn(m, WEIGHTS / WFILE[m]) if m.startswith("dymn") else M.load_mn(m, WEIGHTS / WFILE[m])
        with contextlib.redirect_stdout(io.StringIO()):
            hm, hb = T.net_cost(T.build_model("esc", b.emb_dim, K, cfg))
        cache = {f: lib.feature_macs(b, int(f)) for f in np.unique(fr)}
        out[m] = fm * 1024.0 + np.array([lib.backbone_bitops({"aggregate_macs": 0.0, **cache[f]}, "w8a8", 1024.0) for f in fr]) + hb
        out["_macs"][m] = fm + np.array([cache[f]["macs"] + cache[f].get("aggregate_macs", 0.0) for f in fr]) + hm
    return out


def build_bed(R, D, seeds, models):
    K = int(pd.read_csv(R24 / "csv" / D / "all.csv").class_id.nunique())
    sc24 = pd.read_csv(R24 / "tables" / D / "primary_scaler_choice.csv").set_index("tier").chosen
    V24 = json.loads((R24 / "tables" / D / "verdict.json").read_text())
    Z = {"mn10": T24.Scores(R24, D, "mn10_w8a8", seeds), "ast": T24.Scores(R24, D, "ast_w4g128", seeds),
         **{m: T24.Scores(R, D, f"{m}_w8a8", seeds) for m in models}}
    C = Z["mn10"]
    for m, z in Z.items():
        if not (np.array_equal(z.y, C.y) and np.array_equal(z.test_idx, C.test_idx) and z.folds == C.folds):
            raise SystemExit(f"{m}: rows, test set or folds differ from T24's MN10")
    y, te = C.y, C.test_idx
    keys = [(s, f) for s in seeds for f in C.folds]
    scaler, sc_rows = {"mn10": sc24["cheap"], "ast": sc24["ast"]}, []
    for m in models:  # T21's rule on clean inner validation
        v = {}
        for k in ("fp32fit", "refit"):
            vals = []
            for s, f in keys:
                iv, p = Z[m].iv(s, f, "clean", k)
                if not np.array_equal(iv, C.iv(s, f, "clean", k)[0]):
                    raise SystemExit(f"{m} s{s} f{f}: inner-validation rows differ from T24's")
                vals.append(A.f1_fast(y[iv], p.argmax(1), K))
            v[k] = float(np.mean(vals)) * 100
        scaler[m] = K6.scaler_rule(v)
        sc_rows.append({"tier": m, "iv_f1_fp32fit": v["fp32fit"], "iv_f1_refit": v["refit"], "chosen": scaler[m]})
    c24, _ = T24.per_clip_costs(R24, D, K)
    cn = cnn_costs(R, D, K, models)
    cost = {"mn10": c24["priced"][0], "ast": c24["priced"][1], **{m: cn[m] for m in models}, "frontend": cn["frontend"]}
    macs = {"mn10": c24["macs"][0], "ast": c24["macs"][1], **cn["_macs"]}
    rate = T24.BUDGETS[V24["design"]["budget"]]
    cthr = {(s, f): A.thr_rate(A.margin(C.iv(s, f, "clean", scaler["mn10"])[1]), rate) for s, f in keys}
    n_iv = {f: len(C.iv(seeds[0], f, "clean", "fp32fit")[0]) for f in C.folds}
    ratio = float(np.mean([len(te) / (C.n_fit[f] + n_iv[f]) for f in C.folds]))
    if abs(ratio - V24["nb_ratio"]) > 1e-12:
        raise SystemExit("Nadeau-Bengio ratio differs from T24's")

    def iv(s, f):
        idx, P = None, {}
        for m, z in Z.items():
            i2, p = z.iv(s, f, "clean", scaler[m])
            if idx is None:
                idx = i2
            elif not np.array_equal(i2, idx):
                raise SystemExit(f"{m} s{s} f{f}: inner-validation rows differ")
            P[m] = p
        if np.intersect1d(idx, te).size:
            raise SystemExit("inner validation overlaps test")
        return y[idx], P, {m: v[idx] for m, v in cost.items()}

    def test(c, s, f):
        return y[te], {m: z.test(s, f, c, scaler[m]) for m, z in Z.items()}, {**{m: v[te] for m, v in cost.items()},
                                                                              "_macs": {m: v[te] for m, v in macs.items()}}

    bed = K6.Bed(D, K, keys, iv, test, T24.CONDS, ratio, boot="shared", cascade_thr=cthr, boot_seed=T24.BOOT_SEED)
    return bed, scaler, sc_rows, V24


def check_t24(TT, D):
    ref = pd.read_csv(R24 / "tables" / D / "primary_by_seed_fold.csv")
    ref = ref[ref.system.isin(["mn", "ast", "avg", "cascade"])].replace({"system": {"mn": "mn10"}}).set_index(["cond", "seed", "fold", "system"])
    mine = TT[TT.system.isin(["mn10", "ast", "avg", "cascade"])].set_index(["cond", "seed", "fold", "system"])
    j = mine[["f1", "cost"]].join(ref[["f1", "cost"]], rsuffix="_t24", how="inner")
    if len(j) != len(ref) or len(j) != len(mine):
        raise SystemExit("rows differ from T24's primary_by_seed_fold.csv")
    df1 = float((j.f1 - j.f1_t24).abs().max())
    dc = float(((j.cost - j.cost_t24).abs() / j.cost_t24).max())
    if max(df1, dc) > 1e-9:
        raise SystemExit(f"T24 reproduction failed: F1 {df1}, cost {dc}")
    return {"rows": len(j), "worst_f1": df1, "worst_cost_rel": dc}


def precondition(R, D, seeds, out):
    bed, scaler, sc_rows, V24 = build_bed(R, D, seeds, ["mn20"])
    TT, _ = K6.run_test(bed, {})
    chk = check_t24(TT, D)
    g = TT[(TT.cond == "field") & TT.system.isin(["mn10", "mn20"])].pivot_table(index=["seed", "fold"], columns="system", values="f1")
    fm = (g["mn20"] - g["mn10"]).groupby(level="fold").mean()
    gc = TT[(TT.cond == "clean") & TT.system.isin(["mn10", "mn20"])].pivot_table(index=["seed", "fold"], columns="system", values="f1")
    fmc = (gc["mn20"] - gc["mn10"]).groupby(level="fold").mean()
    P = {"dataset": D, "rule": "MN20-Q8 minus MN10-Q8 on the field mix >= 1.0 macro F1 points (mean over the fold models, seeds averaged within folds, as T24 applied its precondition)",
         "value": float(fm.mean()), "fold_values": np.round(fm.to_numpy(), 3).tolist(), "folds_at_or_above_bar": int((fm >= PRE_BAR).sum()),
         "passes": bool(fm.mean() >= PRE_BAR), "mn20_field_f1": float(g["mn20"].mean()), "mn10_field_f1": float(g["mn10"].mean()),
         "clean_descriptive": {"value": float(fmc.mean()), "fold_values": np.round(fmc.to_numpy(), 3).tolist()},
         "scaler_mn20": scaler["mn20"], "t24_reproduction": chk, "seeds": seeds}
    pd.DataFrame(sc_rows).to_csv(out / "precondition_scaler_choice.csv", index=False)
    (out / "precondition.json").write_text(json.dumps(P, indent=1))
    print(json.dumps(P, indent=1), flush=True)
    return P


def full(R, D, seeds, out, smoke=False, partial=False):
    """partial=True (a dataset that failed the precondition, ANALYSIS_PLAN.md C7): MN20-Q8 is the only new tier, the
    hull holds MN10-Q8, MN20-Q8, AST-Q4, always-average and MN10+MN20; DESCRIPTIVE, written under descriptive_partial/."""
    pre = json.loads((out / "precondition.json").read_text())
    if partial:
        out = out / "descriptive_partial"
        out.mkdir(parents=True, exist_ok=True)
    bed, scaler, sc_rows, V24 = build_bed(R, D, seeds, ["mn20"] if partial else NEW)
    TT0, preds0 = K6.run_test(bed, {})
    chk = check_t24(TT0, D)
    if smoke:
        K6.smoke_permute(bed)
        out = out / "_smoke"
        out.mkdir(parents=True, exist_ok=True)
    SEL, pick, th, sp_iv = K6.select_budgets(bed)
    TT, preds = K6.run_test(bed, th)
    CMP, ST = K6.compare_all(bed, TT, preds)
    pd.DataFrame(sc_rows).to_csv(out / "scaler_choice.csv", index=False)
    SEL.to_csv(out / "budget_selection.csv", index=False)
    sp_iv.to_csv(out / "iv_statics.csv", index=False)
    TT.to_csv(out / "by_seed_fold.csv", index=False)
    CMP.to_csv(out / "compare_by_condition.csv", index=False)
    ST.to_csv(out / "statics_by_condition.csv", index=False)
    row = lambda sysn, cond: CMP[(CMP.cond == cond) & (CMP.system == sysn)].iloc[0]  # noqa: E731
    keep = lambda r: {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()  # noqa: E731
                      if k.endswith(("_hull", "_vs_mn10+mn20", "_vs_cascade", "_vs_s1")) or k in ("f1", "cost", "rate", "hull", "hull_f1")}
    f1r = row("s1", "field")
    V = {"part": "C", "label": "DESCRIPTIVE, partial hull (precondition failed; not a test)" if partial else "PRE-REGISTERED", "dataset": D, "seeds": seeds, "nb_ratio": bed.ratio, "scalers": scaler, "budgets": pick,
         "t24_cascade_budget": V24["design"]["budget"], "t24_reproduction": chk, "precondition": pre,
         "H1_C": {"rule": "field mix: S1 minus the widened equal-cost hull >= 1.0 macro F1, Nadeau-Bengio two-sided, Holm over the datasets that pass the precondition < 0.05",
                  "d": float(f1r.d_hull), "nb_ci95": [float(f1r.nb_lo_hull), float(f1r.nb_hi_hull)],
                  "boot_ci95": [float(f1r.boot_lo_hull), float(f1r.boot_hi_hull)], "p_nb": float(f1r.p_nb_hull),
                  "p_family22": min(1.0, K6.FAMILY * float(f1r.p_nb_hull)), "p_signflip": float(f1r.p_signflip_hull),
                  "folds_positive": int(f1r.folds_pos_hull),
                  "n_folds": int(f1r.n_folds_hull), "fold_d": f1r.fold_d_hull, "hull": f1r.hull, "hull_f1": float(f1r.hull_f1),
                  "s1_f1": float(f1r.f1), "s1_cost": float(f1r.cost), "s1_rate": float(f1r.rate)},
         "results": {f"{s}|{c}": keep(row(s, c)) for s in ("s1", "s2", "cascade") for c in ("field", "clean")},
         "mn10+mn20": {c: {"f1": float(TT[(TT.cond == c) & (TT.system == "mn10+mn20")].f1.mean()),
                           "cost": float(TT[(TT.cond == c) & (TT.system == "mn10+mn20")].cost.mean())} for c in ("field", "clean")}}
    (out / "verdict.json").write_text(json.dumps(V, indent=1, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(CMP[CMP.cond.isin(["clean", "field"])][["cond", "system", "f1", "cost", "rate", "hull", "hull_f1", "d_hull", "nb_lo_hull", "nb_hi_hull",
                                                   "p_nb_hull", "folds_pos_hull"]].round(4).to_string())
    return V


def combine(R, ds):
    pre = {d: json.loads((R / "tables" / "c" / d / "precondition.json").read_text()) for d in ds}
    tested = [d for d in ds if pre[d]["passes"]]
    out = {"datasets_tried": ds, "precondition": {d: {"value": pre[d]["value"], "fold_values": pre[d]["fold_values"], "passes": pre[d]["passes"]} for d in ds},
           "datasets_in_holm": tested, "family": K6.FAMILY, "H1_C": {}}
    if tested:
        V = {d: json.loads((R / "tables" / "c" / d / "verdict.json").read_text()) for d in tested}
        ph = K6.holm(np.array([V[d]["H1_C"]["p_nb"] for d in tested]))
        for d, p in zip(tested, ph):
            h = V[d]["H1_C"]
            out["H1_C"][d] = {**{k: h[k] for k in ("d", "nb_ci95", "boot_ci95", "p_nb", "p_family22", "folds_positive", "n_folds", "hull")},
                              "p_holm": float(p), "passes": bool(h["d"] >= 1.0 and p < 0.05)}
    out["verdict"] = ("PASS on " + ", ".join(d for d in tested if out["H1_C"][d]["passes"])) if any(out["H1_C"][d]["passes"] for d in tested) else \
        ("FAIL" if tested else "STOPPED (no dataset passed the precondition; H1-C not tested)")
    (R / "tables" / "c" / "verdict_combined.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--d")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--precondition", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--combine", nargs="*", default=None)
    a = ap.parse_args()
    if a.combine is not None:
        return combine(a.r, a.combine)
    out = a.r / "tables" / "c" / a.d
    out.mkdir(parents=True, exist_ok=True)
    if a.precondition:
        return precondition(a.r, a.d, a.seeds, out)
    full(a.r, a.d, a.seeds, out, a.smoke, a.partial)


if __name__ == "__main__":
    main()
