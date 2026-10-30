"""P3 recompute for ESC-50/50 from the raw score files: the pick (mn20 W8A8 + 4-bit head -> AST w4g128@512 esc head,
rule avg, budget b05) on the field mixture against the hull of every static system (Stage 2's, Stage 3b's, the 3c
siblings), priced with stage 3c's costs.json. Own code for the field assignment (asserted against audio_shift), the
margin gate and its threshold, the cascade, the hull pair, the exact expected macro F1 of a two-system mixture
(checked by Monte Carlo), the fold t-test, Nadeau-Bengio and the sign-flip p. Then the clean inner-validation
selection: gain of the pick and of the runner-up with the same machinery."""
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
S2, S3B, R, I5, W, out = [Path(p) for p in sys.argv[1:7]]
sys.path.insert(0, str(W / "scripts/redesign/idea6/stage2"))
import audio_shift, analyze as A_  # noqa  (only BUDGETS and the field assignment, for assertions)
K, SEEDS, SL = 50, [0, 1, 2, 3, 4], 512
FIELD_CELLS = tuple((k, s) for k in ("bg", "pink", "white") for s in (10, 5, 0))
costs = json.loads((R / "analysis/esc50/costs.json").read_text())

def field_assignment(n, seed):
    order = np.random.default_rng(seed).permutation(n); o = ["clean"] * n
    for r, i in enumerate(order[(n + 1) // 2:]):
        k, s = FIELD_CELLS[r % 9]; o[i] = f"{k}__{s}db"
    return np.array(o)

def f1_macro(y, p, k=K):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float); den = cm.sum(0) + cm.sum(1); ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

class Src:
    """One run's npz files; field logits assembled from the per-condition files."""
    def __init__(self, d, cfg): self.d, self.cfg, self.cache = d, cfg, {}
    def load(self, cond):
        if cond in self.cache: return self.cache[cond]
        z = np.load(self.d / f"{self.cfg}__{cond}.npz")
        r = {"y": z["y"].astype(int), "fold": z["test_fold"].astype(int), "z": z["test_logits"].astype(float),
             "iv": {int(k[7:]): (z[k].astype(int), z["iv_logits_" + k[7:]].astype(float)) for k in z.files if k.startswith("iv_idx_")}}
        self.cache[cond] = r; return r
    def field(self):
        if "field" in self.cache: return self.cache["field"]
        b = self.load("clean"); n = len(b["y"]); at, av = field_assignment(n, 5001), field_assignment(n, 5002)
        z = b["z"].copy(); iv = {f: (idx, v.copy()) for f, (idx, v) in b["iv"].items()}
        for c in sorted(set(at) - {"clean"}):
            r = self.load(c); m = at == c; z[m] = r["z"][m]
            for f, (idx, v) in iv.items():
                sel = av[idx] == c; v[sel] = r["iv"][f][1][sel]
        assert np.isfinite(z).all()
        self.cache["field"] = {"y": b["y"], "fold": b["fold"], "z": z, "iv": iv}; return self.cache["field"]

# ── systems ──
cheap = {}  # name -> {seed: Src}
for k in ["esc", "8", "4", "bitnet"]:
    for bb in ("fp32", "w8a8"):
        cheap[f"eff:{bb}:{k}"] = {s: Src(S2 / "scores/esc50" / f"effp_esc50_s{s}" / k, bb) for s in SEEDS}
        for m, root in (("mn04", S3B), ("mn10", S3B), ("mn05", R), ("mn20", R), ("dymn04", R)):
            d = {s: Src(root / "scores_cnn/esc50" / f"{m}_esc50_s{s}" / k, f"{m}_{bb}") for s in SEEDS}
            if all((x.d / f"{x.cfg}__clean.npz").exists() for x in d.values()): cheap[f"cnn:{m}:{bb}:{k}"] = d
ast = {}
for h in ["esc", "4"]:
    for cfg in ["w4g128", "n4", "w4", "w5", "w6", "w8g64", "fp32"]:
        d = {s: Src(I5 / "scores/esc50" / f"astesc50_s{s}" / h, cfg) for s in SEEDS}
        if all((x.d / f"{cfg}__clean.npz").exists() for x in d.values()): ast[f"ast:{cfg}+{h}"] = d
    for cfg in ["fp32", "w8g64", "w4g128", "n4", "w3g64", "w4"]:
        ast[f"ast:{cfg}@{SL}+{h}"] = {s: Src(S2 / "scores_ast_short/esc50" / f"astshort_esc50_s{s}" / h, cfg) for s in SEEDS}
    for g, sfx in (("ast256", "@256"), ("ast384", "@384"), ("astm2", "@256x2")):
        for cfg in ["n4", "w4g128"]:
            ast[f"ast:{cfg}{sfx}+{h}"] = {s: Src(S3B / f"scores_{g}/esc50" / f"{g}_esc50_s{s}" / h, cfg) for s in SEEDS}
def cost(name):
    return costs["cheap"][name] if name in costs["cheap"] else costs["ast"][name[4:]]
systems = {**cheap, **ast}
for nm in systems: cost(nm)
print(f"{len(cheap)} cheap statics, {len(ast)} AST statics", flush=True)
ref = cheap["eff:fp32:esc"][0].load("clean"); y_all = ref["y"]; folds = sorted(ref["iv"]); n = len(y_all)
assert (field_assignment(n, 5001) == np.array(audio_shift.field_assignment(n, 5001))).all()
assert A_.BUDGETS["b05"] == 0.05, A_.BUDGETS
keys = [(s, f) for s in SEEDS for f in folds]
te_idx = {f: np.flatnonzero(ref["fold"] == f) for f in folds}

# ── statics on field and clean, per (seed, fold); predictions kept for the mixtures ──
pred, stat = {}, []
for nm, d in systems.items():
    for s in SEEDS:
        for cond in ("field", "clean"):
            r = d[s].field() if cond == "field" else d[s].load("clean")
            assert (r["y"] == y_all).all() and (r["fold"] == ref["fold"]).all()
            for f in folds:
                te = te_idx[f]; p = r["z"][te].argmax(1); pred[(nm, cond, s, f)] = p
                stat.append({"system": nm, "condition": cond, "seed": s, "fold": f, "f1": f1_macro(y_all[te], p), "cost": cost(nm)})
S = pd.DataFrame(stat)

def hull_pair(sp, c):
    lo_set, hi_set = sp[sp.cost <= c * (1 + 1e-12)], sp[sp.cost > c * (1 + 1e-12)]
    best = None
    for lo in lo_set.itertuples():
        cands = [(lo.f1, lo.system, lo.system, 0.0)] + [(((1 - (c - lo.cost) / (hi.cost - lo.cost)) * lo.f1 + (c - lo.cost) / (hi.cost - lo.cost) * hi.f1),
                                                      lo.system, hi.system, (c - lo.cost) / (hi.cost - lo.cost)) for hi in hi_set.itertuples()]
        for v in cands:
            if best is None or v[0] > best[0] + 1e-15: best = v
    return best

def mix_f1_exact(y, plo, phi, lam, k=K):
    """E[macro F1] when each clip independently takes phi with probability lam, else plo. Per class c the F1 is
    2 (T0 + U) / (n_c + N0 + U + V) with U, V the numbers of switched clips among the disagreeing ones that end up
    counted as true positives / as extra predictions of c; U and V are sums of independent Bernoullis."""
    agree = plo == phi; tot = 0.0
    for c in range(k):
        yc = y == c; n_c = int(yc.sum()); both = agree & (plo == c)
        T0, N0 = int((both & yc).sum()), int(both.sum())
        lo_only, hi_only = (plo == c) & ~agree, (phi == c) & ~agree
        a1, a2 = int((lo_only & yc).sum()), int((lo_only & ~yc).sum())   # lo predicts c (correct / wrong), kept with prob 1-lam
        b1, b2 = int((hi_only & yc).sum()), int((hi_only & ~yc).sum())   # hi predicts c, taken with prob lam
        pu = np.convolve(stats.binom.pmf(np.arange(a1 + 1), a1, 1 - lam), stats.binom.pmf(np.arange(b1 + 1), b1, lam))
        pv = np.convolve(stats.binom.pmf(np.arange(a2 + 1), a2, 1 - lam), stats.binom.pmf(np.arange(b2 + 1), b2, lam))
        u = np.arange(len(pu))[:, None]; v = np.arange(len(pv))[None, :]
        tot += float(pu @ (2 * (T0 + u) / (n_c + N0 + u + v)) @ pv)
    return tot / k

def mix_f1_mc(y, plo, phi, lam, draws=4000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(draws):
        take = rng.random(len(y)) < lam; out.append(f1_macro(y, np.where(take, phi, plo)))
    return float(np.mean(out)), float(np.std(out) / np.sqrt(draws))

def tests(d_pairs, keys, ratio):
    fm = np.array([np.mean([d for d, kk in zip(d_pairs, keys) if kk[1] == f]) for f in folds]); k = len(fm)
    sd = fm.std(ddof=1); t_nb = fm.mean() / np.sqrt((1 / k + ratio) * sd ** 2)
    signs = 1 - 2 * ((np.arange(2 ** k)[:, None] >> np.arange(k)[None, :]) & 1)
    p_sf = float(np.mean(np.abs((signs * fm[None, :]).mean(1)) >= abs(fm.mean()) - 1e-12))
    return {"diff_pts": 100 * fm.mean(), "fold_diffs_pts": (100 * fm).round(3).tolist(), "p_t_fold": float(stats.ttest_1samp(fm, 0).pvalue),
            "p_nb": float(2 * stats.t.sf(abs(t_nb), k - 1)), "p_signflip": p_sf}
n_te = np.array([len(te_idx[f]) for f in folds]); NB = float(np.mean(n_te / (n - n_te)))

# ── the cascade ──
def cascade(cheap_nm, tier_nm, rule, rate, cond):
    rows = []
    for s in SEEDS:
        rc = cheap[cheap_nm][s]; ra = ast[tier_nm][s]
        rcc = rc.load("clean"); rx = rc.field() if cond == "field" else rcc; rax = ra.field() if cond == "field" else ra.load("clean")
        for f in folds:
            iv_idx, iv_z = rcc["iv"][f]; m_iv = np.sort(np.unique(np.subtract(*np.sort(softmax(iv_z), 1)[:, [-1, -2]].T)))
            # threshold: the midpoint cut whose escalation rate on the clean inner validation is closest to the budget (ties: lower rate)
            piv = softmax(iv_z); miv = np.sort(piv, 1); miv = miv[:, -1] - miv[:, -2]
            u = np.unique(miv); cuts = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
            rates = (miv[None, :] < cuts[:, None]).mean(1); thr = cuts[np.lexsort((rates, np.abs(rates - rate)))[0]]
            te = te_idx[f]; p = softmax(rx["z"][te]); pe = softmax(rax["z"][te])
            sp = np.sort(p, 1); margin = sp[:, -1] - sp[:, -2]; escm = margin < thr
            de = pe.argmax(1) if rule == "ast" else ((p + pe) / 2).argmax(1); pr = np.where(escm, de, p.argmax(1))
            r_ = float(escm.mean())
            rows.append({"seed": s, "fold": f, "f1": f1_macro(y_all[te], pr), "rate": r_, "cost": cost(cheap_nm) + r_ * cost(tier_nm), "pred": pr,
                         "iv_rate": float((miv < thr).mean())})
    return rows

res = {}
for cond in ("field", "clean"):
    rows = cascade("cnn:mn20:w8a8:4", f"ast:w4g128@{SL}+esc", "avg", 0.05, cond)
    c_mean = float(np.mean([r["cost"] for r in rows]))
    sp = S[S.condition == cond].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
    v, lo, hi, lam = hull_pair(sp, c_mean)
    hull = []
    for r in rows:
        kk = (r["seed"], r["fold"]); te = te_idx[r["fold"]]
        hull.append(f1_macro(y_all[te], pred[(lo, cond, *kk)]) if lam == 0 else mix_f1_exact(y_all[te], pred[(lo, cond, *kk)], pred[(hi, cond, *kk)], lam))
    mc = mix_f1_mc(y_all[te_idx[1]], pred[(lo, cond, 0, 1)], pred[(hi, cond, 0, 1)], lam) if lam > 0 else None
    d = [r["f1"] - h for r, h in zip(rows, hull)]
    res[cond] = {"design": "cnn:mn20:w8a8:4 -> ast:w4g128@512+esc, avg, b05", "f1": float(np.mean([r["f1"] for r in rows])), "cost_T": c_mean / 1e12,
                 "rate": float(np.mean([r["rate"] for r in rows])), "hull_lo": lo, "hull_hi": hi, "lam": lam, "hull_f1_interp": v, "hull_f1_exact_mean": float(np.mean(hull)),
                 "mc_check_seed0_fold1": {"exact": hull[0], "mc": mc}, **tests(d, [(r["seed"], r["fold"]) for r in rows], NB)}
    print(cond, json.dumps(res[cond], indent=1), flush=True)
raw = {"esc50": res["field"]["p_t_fold"], "esc5": 0.036, "urban": 0.041}  # the other two raw p from stage 3c's summary P3 table
ps = np.array(list(raw.values())); o = np.argsort(ps); adj = np.empty(3); run_ = 0
for r_, i in enumerate(o): run_ = max(run_, min(1, (3 - r_) * ps[i])); adj[i] = run_
res["holm"] = dict(zip(raw, adj.round(4).tolist())); res["stage3c_summary_P3_esc50"] = {"diff_pts": 2.11, "p_t_fold": 0.0055, "holm": 0.016, "hull_f1": 0.860, "f1": 0.881}

# ── clean inner-validation selection: the pick and the runner-up ──
def iv_gain(cheap_nm, tier_nm, rule, rate):
    rows = []
    for s in SEEDS:
        rc, ra = cheap[cheap_nm][s].load("clean"), ast[tier_nm][s].load("clean")
        for f in folds:
            iv_idx, z = rc["iv"][f]; p = softmax(z); pe = softmax(ra["iv"][f][1]); assert (ra["iv"][f][0] == iv_idx).all()
            sp_ = np.sort(p, 1); miv = sp_[:, -1] - sp_[:, -2]; u = np.unique(miv); cuts = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
            rates = (miv[None, :] < cuts[:, None]).mean(1); thr = cuts[np.lexsort((rates, np.abs(rates - rate)))[0]]; escm = miv < thr
            de = pe.argmax(1) if rule == "ast" else ((p + pe) / 2).argmax(1); pr = np.where(escm, de, p.argmax(1)); yv = y_all[iv_idx]
            rows.append({"seed": s, "fold": f, "f1_iv": f1_macro(yv, pr), "rate": float(escm.mean()), "cost": cost(cheap_nm) + float(escm.mean()) * cost(tier_nm)})
    c_iv = float(np.mean([r["cost"] for r in rows]))
    # clean inner-validation statics
    st = []
    ivp = {}
    for nm, d in systems.items():
        for s in SEEDS:
            r = d[s].load("clean")
            for f in folds:
                idx, z = r["iv"][f]; pr = z.argmax(1); ivp[(nm, s, f)] = pr; st.append({"system": nm, "f1": f1_macro(y_all[idx], pr), "cost": cost(nm)})
    sp = pd.DataFrame(st).groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
    v, lo, hi, lam = hull_pair(sp, c_iv)
    gains = []
    for r in rows:
        kk = (r["seed"], r["fold"]); idx = cheap[cheap_nm][r["seed"]].load("clean")["iv"][r["fold"]][0]; yv = y_all[idx]
        # inner splits can lack a class, so the plan's 200 simulated draws (seed 12346) instead of the exact formula
        h = f1_macro(yv, ivp[(lo, *kk)]) if lam == 0 else mix_f1_mc(yv, ivp[(lo, *kk)], ivp[(hi, *kk)], lam, draws=200, seed=12346)[0]
        gains.append(r["f1_iv"] - h)
    return {"f1_iv": float(np.mean([r["f1_iv"] for r in rows])), "iv_rate": float(np.mean([r["rate"] for r in rows])), "cost_iv_T": c_iv / 1e12,
            "hull": [lo, hi, lam], "gain_iv_pts": 100 * float(np.mean(gains)), "gain_sd_over_pairs_pts": 100 * float(np.std(gains, ddof=1)), "gains": gains}
pick = iv_gain("cnn:mn20:w8a8:4", f"ast:w4g128@{SL}+esc", "avg", 0.05); runner = iv_gain("cnn:mn10:w8a8:4", f"ast:n4@{SL}+esc", "avg", 0.05)
dd = np.array(pick["gains"]) - np.array(runner["gains"])
res["selection_iv"] = {"pick": {k: v for k, v in pick.items() if k != "gains"}, "runner_up": {k: v for k, v in runner.items() if k != "gains"},
                       "pick_minus_runner_up_pts": 100 * float(dd.mean()), "pick_minus_runner_up_p_t_pairs": float(stats.ttest_1samp(dd, 0).pvalue),
                       "stage3c_picks_json": {"gain_iv": 0.003964659368887378, "runner_up_gain_iv": 0.0035844650122120124, "hull_iv": ["cnn:mn20:w8a8:4", "cnn:mn10:fp32:8", 0.6986]}}
print(json.dumps(res["selection_iv"], indent=1))
json.dump(res, open(out, "w"), indent=1)
