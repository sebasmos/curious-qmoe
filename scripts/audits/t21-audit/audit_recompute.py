"""Independent recomputation of T21 D1 primary (pool design, mn10_w8a8 cheap tier).
Reads only stored score files (logits), pool.csv and the extraction length sidecars. Own code throughout.
Usage: python -I audit_recompute.py <results_dir> <out_dir>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SEEDS, FOLDS = [0, 1, 2, 3, 4], [1, 2, 3, 4, 5]
BUD = [0.05, 0.10, 0.20, 0.30, 0.50]
NOISE = ["bg__10db", "bg__0db", "pink__10db", "pink__0db", "white__10db", "white__0db"]
CONDS = ["clean", *NOISE, "field"]
K = 41
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)

def sm(z):
    z = z.astype(np.float64); z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def mf1(y, p):
    return 100 * f1_score(y, p, average="macro", zero_division=0)
def f1v(Y, Pp):
    """Own vectorised macro F1 over rows of Y, Pp [D, n]; classes present in y or p (sklearn rule)."""
    Dn = Y.shape[0]
    cm = np.bincount((np.arange(Dn)[:, None] * K * K + Y * K + Pp).ravel(), minlength=Dn * K * K).reshape(Dn, K, K)
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float); den = cm.sum(1) + cm.sum(2)
    return 100 * np.where(den > 0, 2 * tp / np.maximum(den, 1), 0).sum(1) / (den > 0).sum(1)
def marg(p):
    q = np.partition(p, -2, axis=1); return q[:, -1] - q[:, -2]
def thr_for(m, rate):
    s = np.sort(m); n = len(s); k = int(round(rate * n))
    return (s[k - 1] + s[k]) / 2 if 0 < k < n else (np.inf if k >= n else -np.inf)

pool = pd.read_csv(R / "csv/d1/pool.csv")
Z = {t: {s: dict(np.load(R / f"scores/d1/pool_{t}_s{s}.npz")) for s in SEEDS} for t in ("mn10_w8a8", "ast_w4g128")}
y = Z["mn10_w8a8"][0]["y"].astype(int)
fold = Z["mn10_w8a8"][0]["fold"].astype(int)

# ---- structural / leakage checks ----
_yy = np.random.default_rng(0).integers(0, K, (3, 500)); _pp = np.random.default_rng(1).integers(0, K, (3, 500))
assert np.allclose(f1v(_yy, _pp), [mf1(a, b) for a, b in zip(_yy, _pp)])
P("== structure ==")
P("y equals pool.csv class_id:", np.array_equal(y, pool.class_id.to_numpy()))
P("fold equals pool.csv folder:", np.array_equal(fold, pool.folder.to_numpy()))
for t in Z:
    for s in SEEDS:
        assert np.array_equal(Z[t][s]["y"], y) and np.array_equal(Z[t][s]["fold"], fold), (t, s)
P("y and fold identical across both tiers and all 5 seeds: True")
P("fold sizes:", np.bincount(fold)[1:].tolist())
ct = pd.crosstab(pool.label, pool.folder); P("per-class per-fold count min/max:", int(ct.values.min()), int(ct.values.max()),
   "; max within-class fold spread:", int((ct.max(1) - ct.min(1)).max()))
P("freesound_id unique:", pool.freesound_id.is_unique, "; name unique:", pool.name.is_unique, "; all manually verified:", bool((pool.manually_verified == 1).all()))
P("official test clips per fold:", pool[pool.official == "test"].groupby("folder").size().tolist())
for s in SEEDS:
    for f in FOLDS:
        a, b = Z["mn10_w8a8"][s][f"iv_idx_{f}"], Z["ast_w4g128"][s][f"iv_idx_{f}"]
        assert np.array_equal(a, b), (s, f)
        assert not np.isin(a, np.flatnonzero(fold == f)).any(), (s, f)
P("inner-val rows identical across tiers and disjoint from the test fold (25 seed-fold pairs): True")
P("inner-val size per fold:", len(a), "; iv sets differ across seeds:", not np.array_equal(Z["mn10_w8a8"][0]["iv_idx_1"], Z["mn10_w8a8"][1]["iv_idx_1"]))

# field composite check: each test__field row equals one stored cell (or none = a 5 dB cell)
def field_assign(n, seed):
    cells = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]
    o = np.random.default_rng(seed).permutation(n); out = np.array(["clean"] * n, dtype=object)
    for r, i in enumerate(o[(n + 1) // 2:]): out[i] = cells[r % 9]
    return out
fa = field_assign(len(y), 5001)
for t in Z:
    z = Z[t][0]; fld = z["test__field__refit"]; ok = 0; tot = 0
    for c in ["clean", *NOISE]:
        rows = np.flatnonzero(fa == c); tot += len(rows)
        ok += int(np.all(fld[rows] == z[f"test__{c}__refit"][rows], axis=1).sum())
    P(f"{t}: field rows matching the cell given by my own reimplementation of the assignment (seed 5001): {ok}/{tot} (5 dB rows not stored, {int(np.isin(fa, ['bg__5db','pink__5db','white__5db']).sum())})")

# ---- scaler choice on clean inner validation ----
P("== scaler choice (clean inner validation, mean over 25 seed-folds) ==")
scal = {}
for t in Z:
    m = {}
    for k in ("fp32fit", "refit"):
        m[k] = np.mean([mf1(y[Z[t][s][f"iv_idx_{f}"]], Z[t][s][f"iv__{f}__clean__{k}"].argmax(1)) for s in SEEDS for f in FOLDS])
    scal[t] = "refit" if m["refit"] > m["fp32fit"] else "fp32fit"
    P(t, {k: round(v, 4) for k, v in m.items()}, "->", scal[t])

# ---- costs: my own analytic AST count, their MN10 cost re-derived separately (see audit_costs.py) ----
costs = json.loads((OUT / "costs_audit.json").read_text())
cc = np.load(OUT / "cost_mn_per_clip.npy"); ca = np.load(OUT / "cost_ast_per_clip.npy")
P("== costs used ==  mean MN10 %.4e  mean AST %.4e  ratio %.1f" % (cc.mean(), ca.mean(), ca.mean() / cc.mean()))

def upper_hull(pts, c):
    """pts: dict name -> (cost, f1). Best static at cost <= c or two-point mix with lo<=c<hi."""
    best = None
    for lo, (cl, fl) in pts.items():
        if cl > c * (1 + 1e-12): continue
        cand = [(fl, lo, lo, 0.0)]
        for hi, (ch, fh) in pts.items():
            if ch > c * (1 + 1e-12):
                lam = (c - cl) / (ch - cl); cand.append(((1 - lam) * fl + lam * fh, lo, hi, lam))
        for v in cand:
            if best is None or v[0] > best[0] + 1e-12: best = v
    return best

def mc_f1(yt, plo, phi, lam, rng, draws=4000):
    if lam == 0: return mf1(yt, plo)
    pick = rng.random((draws, len(yt))) < lam
    return float(f1v(np.broadcast_to(yt, pick.shape), np.where(pick, phi[None], plo[None])).mean())

ks, ka = scal["mn10_w8a8"], scal["ast_w4g128"]
# ---- budget choice on clean inner validation ----
P("== budget choice on clean inner validation ==")
IV = {}
for s in SEEDS:
    for f in FOLDS:
        iv = Z["mn10_w8a8"][s][f"iv_idx_{f}"]
        IV[(s, f)] = (iv, sm(Z["mn10_w8a8"][s][f"iv__{f}__clean__{ks}"]), sm(Z["ast_w4g128"][s][f"iv__{f}__clean__{ka}"]))
st = {}
for (s, f), (iv, pc, pa) in IV.items():
    for nm, p, c_ in (("mn", pc, cc[iv]), ("ast", pa, ca[iv]), ("avg", (pc + pa) / 2, cc[iv] + ca[iv])):
        st[(nm, s, f)] = (mf1(y[iv], p.argmax(1)), p.argmax(1), c_.mean())
pts_iv = {nm: (np.mean([st[(nm, s, f)][2] for s, f in IV]), np.mean([st[(nm, s, f)][0] for s, f in IV])) for nm in ("mn", "ast", "avg")}
P("iv statics (cost, F1):", {k: (f"{v[0]:.3e}", round(v[1], 3)) for k, v in pts_iv.items()})
THR, brows = {}, []
for b in BUD:
    f1s, cs, rates = [], [], []
    for (s, f), (iv, pc, pa) in IV.items():
        m = marg(pc); t_ = thr_for(m, b); THR[(b, s, f)] = t_
        e = m < t_; rates.append(e.mean())
        f1s.append(mf1(y[iv], np.where(e, ((pc + pa) / 2).argmax(1), pc.argmax(1))))
        cs.append(np.mean(cc[iv] + e * ca[iv]))
    c_iv = np.mean(cs); h = upper_hull(pts_iv, c_iv)
    hl = np.mean([(1 - h[3]) * st[(h[1], s, f)][0] + h[3] * st[(h[2], s, f)][0] for s, f in IV])
    brows.append({"budget": b, "rate_iv": np.mean(rates), "f1_iv": np.mean(f1s), "cost_iv": c_iv, "hull_iv_linear": hl,
                  "gain_iv": np.mean(f1s) - hl, "hull": f"{h[1]}|{h[2]}|{h[3]:.3f}"})
BS = pd.DataFrame(brows); P(BS.round(4).to_string())
bstar = BS.sort_values(["gain_iv", "cost_iv"], ascending=[False, True]).iloc[0].budget
P("chosen budget:", bstar)
BS.to_csv(OUT / "audit_budget_selection.csv", index=False)

# ---- test ----
P("== test ==")
rows, PRED = [], {}
for c in CONDS:
    for s in SEEDS:
        for f in FOLDS:
            te = np.flatnonzero(fold == f); yt = y[te]
            pc = sm(Z["mn10_w8a8"][s][f"test__{c}__{ks}"][te]); pa = sm(Z["ast_w4g128"][s][f"test__{c}__{ka}"][te])
            e = marg(pc) < THR[(bstar, s, f)]
            pr = {"mn": pc.argmax(1), "ast": pa.argmax(1), "avg": ((pc + pa) / 2).argmax(1)}
            pr["cascade"] = np.where(e, pr["avg"], pr["mn"])
            co = {"mn": cc[te].mean(), "ast": ca[te].mean(), "avg": (cc[te] + ca[te]).mean(), "cascade": np.mean(cc[te] + e * ca[te]),
                  "cascade_cheap_not_charged": np.mean(np.where(e, ca[te], cc[te]))}
            PRED[(c, s, f)] = (yt, pr, e, te)
            for nm in ("mn", "ast", "avg", "cascade"):
                rows.append({"cond": c, "seed": s, "fold": f, "system": nm, "f1": mf1(yt, pr[nm]), "cost": co[nm],
                             "rate": e.mean(), "cost_nocheap": co["cascade_cheap_not_charged"]})
T = pd.DataFrame(rows); T.to_csv(OUT / "audit_by_seed_fold.csv", index=False)

def nb(dfold, ratio):
    k = len(dfold); m = dfold.mean(); sd = dfold.std(ddof=1)
    tt = m / np.sqrt((1 / k + ratio) * sd ** 2); p = 2 * stats.t.sf(abs(tt), k - 1)
    h = stats.t.ppf(0.975, k - 1) * np.sqrt((1 / k + ratio) * sd ** 2)
    return p, (m - h, m + h), 2 * stats.t.sf(abs(m / (sd / np.sqrt(k))), k - 1)
fn = np.bincount(fold)[1:]; ratio = float(np.mean(fn / (len(y) - fn)))
P("NB ratio n_test/n_train:", ratio)
res = []
rng = np.random.default_rng(7)
for c in CONDS:
    g = T[T.cond == c]
    pts = {nm: (g[g.system == nm].cost.mean(), g[g.system == nm].f1.mean()) for nm in ("mn", "ast", "avg")}
    Cc = g[g.system == "cascade"].cost.mean()
    h = upper_hull(pts, Cc)
    lo, hi, lam = h[1], h[2], h[3]
    sf = g.set_index(["system", "seed", "fold"]).f1
    d_lin, d_mc = [], []
    for s in SEEDS:
        for f in FOLDS:
            yt, pr, e, te = PRED[(c, s, f)]
            hl = (1 - lam) * sf[(lo, s, f)] + lam * (sf[(hi, s, f)] if lam > 0 else 0)
            hm = mc_f1(yt, pr[lo], pr[hi], lam, rng) if c in ("clean", "field") else np.nan
            d_lin.append((f, sf[("cascade", s, f)] - hl)); d_mc.append((f, sf[("cascade", s, f)] - hm))
    dl = pd.DataFrame(d_lin, columns=["f", "d"]).groupby("f").d.mean().to_numpy()
    dm = pd.DataFrame(d_mc, columns=["f", "d"]).groupby("f").d.mean().to_numpy()
    p, ci, p_plain = nb(dl, ratio)
    # sensitivity: hull at lambda = realised rate (length-independent AST cost), and at cascade cost without the cheap charge
    rate = g[g.system == "cascade"].rate.mean()
    h_rate = upper_hull({k: (v[0] if k == "mn" else None, v[1]) for k, v in pts.items()} if False else
                        {"mn": (0.0, pts["mn"][1]), "ast": (1.0, pts["ast"][1]), "avg": (1.0 + pts["mn"][0] / pts["ast"][0], pts["avg"][1])}, rate)
    Cn = g[g.system == "cascade"].cost_nocheap.mean(); h_n = upper_hull(pts, Cn)
    r = {"cond": c, "mn": pts["mn"][1], "ast": pts["ast"][1], "avg": pts["avg"][1], "cascade": g[g.system == "cascade"].f1.mean(),
         "rate": rate, "cascade_cost": Cc, "hull": f"{lo}|{hi}|{lam:.4f}", "hull_f1_linear": h[0], "d_linear": dl.mean(),
         "fold_d_linear": np.round(dl, 3).tolist(), "folds_pos": int((dl > 0).sum()), "p_nb": p, "nb_ci": [round(ci[0], 3), round(ci[1], 3)],
         "p_plain_t": p_plain, "d_mc_expected": dm.mean() if c in ("clean", "field") else np.nan,
         "fold_d_mc": np.round(dm, 3).tolist() if c in ("clean", "field") else None,
         "sens_d_hull_at_lambda_eq_rate": g[g.system == "cascade"].f1.mean() - h_rate[0],
         "sens_d_if_cheap_not_charged": g[g.system == "cascade"].f1.mean() - h_n[0],
         "ast_minus_mn_folds": np.round((sf["ast"] - sf["mn"]).groupby(level="fold").mean().to_numpy(), 3).tolist()}
    res.append(r)
RES = pd.DataFrame(res); RES.to_csv(OUT / "audit_by_condition.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
P(RES[["cond", "mn", "ast", "avg", "cascade", "rate", "hull", "hull_f1_linear", "d_linear", "d_mc_expected", "p_nb", "folds_pos"]].round(4).to_string())
for r in res:
    if r["cond"] in ("field", "clean"):
        P(r["cond"], "fold_d_linear", r["fold_d_linear"], "fold_d_mc", r["fold_d_mc"], "NB CI", r["nb_ci"], "p_nb %.5f" % r["p_nb"],
          "family17 %.4f" % min(1, 17 * r["p_nb"]), "plain t p %.5f" % r["p_plain_t"],
          "| sens lambda=rate %.3f, cheap not charged %.3f" % (r["sens_d_hull_at_lambda_eq_rate"], r["sens_d_if_cheap_not_charged"]),
          "| AST-MN folds", r["ast_minus_mn_folds"])

# ---- humility: escalation rate by condition at the clean threshold ----
rt = T[T.system == "cascade"].groupby(["cond", "fold"]).rate.mean().unstack()
fm = rt.mean(1) * 100
P("== escalation rate (fold mean, %) ==", fm.round(2).to_dict())
steps = [("clean", f"{k}__10db") for k in ("bg", "pink", "white")] + [(f"{k}__10db", f"{k}__0db") for k in ("bg", "pink", "white")] + [("clean", "field")]
P("all steps rise:", all(fm[b] > fm[a] for a, b in steps), "; folds rising per step:", [int((rt.loc[b] > rt.loc[a]).sum()) for a, b in steps])

# ---- clip-level bootstrap (2000 draws, clips resampled within folds, hull linear inside each draw) ----
bs = np.random.default_rng(99)
for c in ("field", "clean"):
    lo, hi, lam = [x for x in RES[RES.cond == c].hull.iloc[0].split("|")]; lam = float(lam)
    acc = np.zeros(2000)
    for f in FOLDS:
        n = int((fold == f).sum()); I = bs.integers(0, n, (2000, n))
        for s in SEEDS:
            yt, pr, e, te = PRED[(c, s, f)]
            Y = yt[I]
            fh = (1 - lam) * f1v(Y, pr[lo][I]) + (lam * f1v(Y, pr[hi][I]) if lam > 0 else 0)
            acc += (f1v(Y, pr["cascade"][I]) - fh) / (len(SEEDS) * len(FOLDS))
    P(f"bootstrap 95% {c}:", np.round(np.percentile(acc, [2.5, 97.5]), 3).tolist())
(OUT / "audit_recompute.log").write_text("\n".join(log) + "\n")
