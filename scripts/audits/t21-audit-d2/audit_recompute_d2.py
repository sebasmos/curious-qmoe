"""Recompute the D2 primary design from the stored logits: scaler and budget choice on clean inner validation,
test F1s, escalation, the equal-cost hull with always-average, random gate, best single, NB tests, bootstrap, H3.
Usage: python -I audit_recompute_d2.py <T21 results> <audit out>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from audit_common import (BUDGETS, CONDS, FOLDS, K, NOISE, SEEDS, Store, exact_mix_f1_present, macro_f1,  # noqa: E402
                          margin, mc_mix_f1, nb_test, signflip, softmax, thr_for_rate, upper_hull_pair)

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
pool = pd.read_csv(R / "csv/d2/pool.csv")
y_all = pool.class_id.to_numpy()
fold_all = pool.folder.to_numpy()
C, S = Store(R, "mn10_w8a8"), Store(R, "ast_w4g128")
cc = np.load(OUT / "cost_mn_per_clip.npy")
ca = np.load(OUT / "cost_ast_per_clip.npy")
log = open(OUT / "audit_recompute.log", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


# ---------- structure ----------
for s in SEEDS:
    for T in (C, S):
        assert np.array_equal(T.get(s, "y"), y_all), "y differs from pool.csv"
        assert np.array_equal(T.get(s, "fold"), fold_all), "fold differs from pool.csv"
    for f in FOLDS:
        ivc, iva = C.get(s, f"iv_idx_{f}"), S.get(s, f"iv_idx_{f}")
        assert np.array_equal(ivc, iva), "iv rows differ between tiers"
        assert not np.isin(ivc, np.flatnonzero(fold_all == f)).any(), "iv overlaps test"
P("structure: y and fold equal pool.csv for both tiers and all seeds; iv rows equal across tiers; iv disjoint from test")
fold_n = {f: int((fold_all == f).sum()) for f in FOLDS}
ratio = float(np.mean([fold_n[f] / (len(y_all) - fold_n[f]) for f in FOLDS]))
P("fold sizes", fold_n, "NB ratio (mean n_test/n_train)", round(ratio, 6))
P("classes absent from test fold:", {f: int(K - len(np.unique(y_all[fold_all == f]))) for f in FOLDS})

# my macro F1 against sklearn on a few cases
s0p = softmax(C.get(0, "test__field__refit")).argmax(1)
for f in FOLDS:
    te = fold_all == f
    a, b = macro_f1(y_all[te], s0p[te]), f1_score(y_all[te], s0p[te], average="macro", zero_division=0)
    assert abs(a - b) < 1e-12, (a, b)
P("my macro F1 equals sklearn f1_score(average='macro') on the 5 field test folds (seed 0, MN10)")

# ---------- scaler choice ----------
sc = {}
rows = []
for name, T in (("cheap", C), ("ast", S)):
    m = {}
    for k in ("fp32fit", "refit"):
        m[k] = np.mean([macro_f1(y_all[T.get(s, f"iv_idx_{f}")], T.get(s, f"iv__{f}__clean__{k}").argmax(1)) for s in SEEDS for f in FOLDS]) * 100
    sc[name] = "refit" if m["refit"] > m["fp32fit"] else "fp32fit"
    rows.append({"tier": name, "iv_fp32fit": m["fp32fit"], "iv_refit": m["refit"], "chosen": sc[name]})
P("scaler choice:", rows)
kc, ka = sc["cheap"], sc["ast"]

# ---------- budget choice on clean inner validation ----------
ivd = {}
for s in SEEDS:
    for f in FOLDS:
        iv = C.get(s, f"iv_idx_{f}")
        ivd[(s, f)] = (iv, softmax(C.get(s, f"iv__{f}__clean__{kc}")), softmax(S.get(s, f"iv__{f}__clean__{ka}")))
st = {}
for (s, f), (iv, pc, pa) in ivd.items():
    for nm, p, cst in (("mn", pc, cc[iv]), ("ast", pa, ca[iv]), ("avg", (pc + pa) / 2, cc[iv] + ca[iv])):
        st[(nm, s, f)] = (macro_f1(y_all[iv], p.argmax(1)), p.argmax(1), cst.mean())
pts_iv = {nm: (np.mean([st[(nm, s, f)][2] for s in SEEDS for f in FOLDS]), np.mean([st[(nm, s, f)][0] for s in SEEDS for f in FOLDS]))
          for nm in ("mn", "ast", "avg")}
thr = {}
brow = []
for b, rate in BUDGETS.items():
    f1s, cs, lin, ex = [], [], [], []
    for (s, f), (iv, pc, pa) in ivd.items():
        t_ = thr_for_rate(margin(pc), rate)
        thr[(b, s, f)] = t_
        esc = margin(pc) < t_
        pred = np.where(esc, ((pc + pa) / 2).argmax(1), pc.argmax(1))
        f1s.append(macro_f1(y_all[iv], pred))
        cs.append(float(np.mean(cc[iv] + esc * ca[iv])))
    civ = float(np.mean(cs))
    v, lo, hi, lam = upper_hull_pair(pts_iv, civ)
    for (s, f), (iv, pc, pa) in ivd.items():
        flo, plo = st[(lo, s, f)][:2]
        if lam == 0:
            lin.append(flo)
            ex.append(flo)
            continue
        fhi, phi = st[(hi, s, f)][:2]
        lin.append((1 - lam) * flo + lam * fhi)
        ex.append(exact_mix_f1_present(y_all[iv], plo, phi, lam)[1])
    brow.append({"budget": b, "f1_iv": np.mean(f1s) * 100, "cost_iv": civ, "hull": f"{lo}|{hi}|{lam:.3f}",
                 "gain_linear": (np.mean(f1s) - np.mean(lin)) * 100, "gain_expected": (np.mean(f1s) - np.mean(ex)) * 100})
BS = pd.DataFrame(brow)
BS.to_csv(OUT / "audit_budget_selection.csv", index=False)
P(BS.round(4).to_string())
bstar = BS.sort_values(["gain_expected", "cost_iv"], ascending=[False, True]).budget.iloc[0]
bstar_lin = BS.sort_values(["gain_linear", "cost_iv"], ascending=[False, True]).budget.iloc[0]
P("chosen budget (expected-F1 IV hull):", bstar, " (linear IV hull):", bstar_lin)

# best single on inner validation of each family
best = {}
for fam in ("clean", "field"):
    v = {nm: np.mean([macro_f1(y_all[T.get(s, f"iv_idx_{f}")], T.get(s, f"iv__{f}__{fam}__{k}").argmax(1)) for s in SEEDS for f in FOLDS])
         for nm, T, k in (("mn", C, kc), ("ast", S, ka))}
    best[fam] = max(v, key=v.get)
    P("best single", fam, {k_: round(x * 100, 3) for k_, x in v.items()}, "->", best[fam])

# ---------- test ----------
rows, au, preds = [], [], {}
for c in CONDS:
    for s in SEEDS:
        pcf, paf = softmax(C.get(s, f"test__{c}__{kc}")), softmax(S.get(s, f"test__{c}__{ka}"))
        for f in FOLDS:
            te = np.flatnonzero(fold_all == f)
            pc, pa, yt = pcf[te], paf[te], y_all[te]
            m = margin(pc)
            esc = m < thr[(bstar, s, f)]
            pr = {"mn": pc.argmax(1), "ast": pa.argmax(1), "avg": ((pc + pa) / 2).argmax(1)}
            pr["cascade"] = np.where(esc, pr["avg"], pr["mn"])
            cost = {"mn": cc[te].mean(), "ast": ca[te].mean(), "avg": (cc[te] + ca[te]).mean(), "cascade": float(np.mean(cc[te] + esc * ca[te]))}
            for nm, p in pr.items():
                rows.append({"cond": c, "seed": s, "fold": f, "system": nm, "f1": macro_f1(yt, p) * 100, "f1_ypresent": macro_f1(yt, p, rule="y") * 100,
                             "cost": cost[nm], "rate": float(esc.mean()), "ast_share": float((esc * ca[te]).sum() / ca[te].sum())})
            preds[(c, s, f)] = (yt, pr, esc)
            fix = (pr["mn"] != yt) & (pr["ast"] == yt)
            au.append({"cond": c, "seed": s, "fold": f, "rate": float(esc.mean()), "auroc_fixable": roc_auc_score(fix, 1 - m),
                       "auroc_wrong": roc_auc_score(pr["mn"] != yt, 1 - m)})
    C.drop("test__"), S.drop("test__")
T_ = pd.DataFrame(rows)
AU = pd.DataFrame(au)

# ---------- hull, comparators, tests ----------
res, per_sf = [], []
for c in CONDS:
    g = T_[T_.cond == c]
    pts = {nm: (g[g.system == nm].cost.mean(), g[g.system == nm].f1.mean()) for nm in ("mn", "ast", "avg")}
    Cc = g[g.system == "cascade"].cost.mean()
    v, lo, hi, lam = upper_hull_pair(pts, Cc)
    out = {"cond": c, "hull": f"{lo}|{hi}|{lam:.4f}", "cascade_cost": Cc, "rate": g[g.system == "cascade"].rate.mean()}
    for nm in ("mn", "ast", "avg", "cascade"):
        out[f"{nm}_f1"] = g[g.system == nm].f1.mean()
    want_full = c in ("clean", "field")
    for s in SEEDS:
        for f in FOLDS:
            yt, pr, esc = preds[(c, s, f)]
            gg = g[(g.seed == s) & (g.fold == f)].set_index("system")
            r = {"cond": c, "seed": s, "fold": f, "cascade": gg.f1["cascade"], "cascade_yp": gg.f1_ypresent["cascade"],
                 "mn": gg.f1["mn"], "ast": gg.f1["ast"], "avg": gg.f1["avg"]}
            r["hull_lin"] = (1 - lam) * gg.f1[lo] + lam * gg.f1[hi] if lam > 0 else gg.f1[lo]
            if want_full and lam > 0:
                e_yp, e_sk_approx = exact_mix_f1_present(yt, pr[lo], pr[hi], lam)
                r["hull_exact_yp"], r["hull_sk_approx"] = e_yp * 100, e_sk_approx * 100
                r["hull_mc_sk"], r["hull_mc_se"] = (x * 100 for x in mc_mix_f1(yt, pr[lo], pr[hi], lam))
                rg_yp, rg_sk = exact_mix_f1_present(yt, pr["mn"], pr["avg"], float(esc.mean()))
                r["rgate_exact_yp"], r["rgate_sk_approx"] = rg_yp * 100, rg_sk * 100
                r["rgate_mc_sk"] = mc_mix_f1(yt, pr["mn"], pr["avg"], float(esc.mean()), seed=778)[0] * 100
            per_sf.append(r)
    D = pd.DataFrame([r for r in per_sf if r["cond"] == c])
    fm = D.groupby("fold").mean(numeric_only=True)
    comps = {"hull_lin": fm.cascade - fm.hull_lin, "best": fm.cascade - fm[best["field" if c == "field" else "clean"]], "avg": fm.cascade - fm.avg}
    if want_full:
        comps.update({"hull_mc_sk": fm.cascade - fm.hull_mc_sk, "hull_sk_approx": fm.cascade - fm.hull_sk_approx,
                      "hull_exact_yp(cascade_yp)": fm.cascade_yp - fm.hull_exact_yp, "rgate_mc_sk": fm.cascade - fm.rgate_mc_sk,
                      "rgate_sk_approx": fm.cascade - fm.rgate_sk_approx})
    for nm, d in comps.items():
        p, lo_, hi_ = nb_test(d.to_numpy(), ratio)
        out[f"d_{nm}"], out[f"p_{nm}"], out[f"ci_{nm}"] = d.mean(), p, f"{lo_:.3f}..{hi_:.3f}"
        out[f"pos_{nm}"] = int((d > 0).sum())
        out[f"folds_{nm}"] = json.dumps(np.round(d.to_numpy(), 3).tolist())
    out["signflip_hull"] = signflip((fm.cascade - fm.hull_lin).to_numpy())
    am = fm.ast - fm.mn
    out["ast_minus_mn"], out["ast_minus_mn_folds"] = am.mean(), json.dumps(np.round(am.to_numpy(), 3).tolist())
    if want_full:
        out["mc_se_max"] = D.hull_mc_se.max()
        # clip-level bootstrap within folds, linear hull at the fixed lambda
        rng = np.random.default_rng(4242)
        acc = np.zeros(2000)
        from audit_common import macro_f1_batch
        for f in FOLDS:
            n_f = fold_n[f]
            I = rng.integers(0, n_f, (2000, n_f))
            ds = np.zeros(2000)
            for s in SEEDS:
                yt, pr, _ = preds[(c, s, f)]
                Y = yt[I]

                def fb(p):
                    out_ = np.zeros(2000)
                    for j in range(0, 2000, 250):
                        yy, pp = Y[j:j + 250], p[I[j:j + 250]]
                        idx = (np.arange(len(yy))[:, None] * (K * K) + yy * K + pp).ravel()
                        cm = np.bincount(idx, minlength=len(yy) * K * K).reshape(len(yy), K, K)
                        tp = np.diagonal(cm, axis1=1, axis2=2).astype(float)
                        den = cm.sum(2) + cm.sum(1)
                        out_[j:j + 250] = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0).sum(1) / (den > 0).sum(1)
                    return out_
                ds += fb(pr["cascade"]) - ((1 - lam) * fb(pr[lo]) + lam * fb(pr[hi]))
            acc += ds / len(SEEDS)
        acc = acc / len(FOLDS) * 100
        out["boot_lo"], out["boot_hi"] = np.percentile(acc, 2.5), np.percentile(acc, 97.5)
    res.append(out)
    P(c, {k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in out.items()})
RES = pd.DataFrame(res)
RES.to_csv(OUT / "audit_by_condition.csv", index=False)
pd.DataFrame(per_sf).to_csv(OUT / "audit_by_seed_fold.csv", index=False)
T_.to_csv(OUT / "audit_test_rows.csv.gz", index=False)

# ---------- H3 ----------
rate = AU.groupby(["cond", "fold"]).rate.mean().unstack()
fmr = rate.mean(1)
for kind in ("bg", "pink", "white"):
    for a_, b_ in (("clean", f"{kind}__10db"), (f"{kind}__10db", f"{kind}__0db")):
        P(f"H3 {a_} -> {b_}: {fmr[a_] * 100:.2f} -> {fmr[b_] * 100:.2f}, folds rising {(rate.loc[b_] > rate.loc[a_]).sum()}")
P(f"H3 clean -> field: {fmr['clean'] * 100:.2f} -> {fmr['field'] * 100:.2f}, folds rising {(rate.loc['field'] > rate.loc['clean']).sum()}")
auc = AU.groupby("cond")[["auroc_fixable", "auroc_wrong", "rate"]].mean()
auc.to_csv(OUT / "audit_gate_auroc.csv")
P(auc.round(4).to_string())

# ---------- every budget on test (descriptive only) ----------
brs = []
for b in BUDGETS:
    for c in ("clean", "field"):
        dd = []
        rs = []
        for f in FOLDS:
            ds = []
            for s in SEEDS:
                yt, pr, _ = preds[(c, s, f)]
                te = np.flatnonzero(fold_all == f)
                pc = softmax(C.get(s, f"test__{c}__{kc}"))[te]
                esc = margin(pc) < thr[(b, s, f)]
                casc = np.where(esc, pr["avg"], pr["mn"])
                ds.append((macro_f1(yt, casc) * 100, float(np.mean(cc[te] + esc * ca[te])), esc.mean()))
            dd.append(ds)
        g = T_[T_.cond == c]
        pts = {nm: (g[g.system == nm].cost.mean(), g[g.system == nm].f1.mean()) for nm in ("mn", "ast", "avg")}
        Cb = np.mean([x[1] for ds in dd for x in ds])
        v, lo, hi, lam = upper_hull_pair(pts, Cb)
        fold_d = []
        for i, f in enumerate(FOLDS):
            dsf = []
            for j, s in enumerate(SEEDS):
                gg = g[(g.seed == s) & (g.fold == f)].set_index("system").f1
                h = (1 - lam) * gg[lo] + lam * gg[hi] if lam > 0 else gg[lo]
                dsf.append(dd[i][j][0] - h)
            fold_d.append(np.mean(dsf))
        p, l_, h_ = nb_test(fold_d, ratio)
        brs.append({"budget": b, "cond": c, "rate": np.mean([x[2] for ds in dd for x in ds]), "cascade_f1": np.mean([x[0] for ds in dd for x in ds]),
                    "d_hull_linear": np.mean(fold_d), "p_nb": p, "folds_pos": int((np.array(fold_d) > 0).sum())})
    C.drop("test__")
BR = pd.DataFrame(brs)
BR.to_csv(OUT / "audit_every_budget_test.csv", index=False)
P(BR.round(4).to_string())
log.close()
