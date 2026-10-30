"""Finding 2: UrbanSound8K P5b recomputed with the exact-length mn10 whose FP32 head was retrained on exact-length
embeddings (R/scores_unpr; the P2 design uses that head kind), against the same hull as stage 3c's P5b (every static,
plus the exact-length mn10 statics at their own cost); also the padded P2 design and the exact-length design with the
padded heads, as reproduction checks. Field mixture, 3 seeds x 10 folds, fold t-test."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
S2, S3B, R, I5, out = [Path(p) for p in sys.argv[1:6]]
K, SEEDS, SL = 10, [0, 1, 2], 400
FIELD_CELLS = tuple((k, s) for k in ("bg", "pink", "white") for s in (10, 5, 0))
costs = json.loads((R / "analysis/urban/costs.json").read_text())
def field_assignment(n, seed):
    order = np.random.default_rng(seed).permutation(n); o = ["clean"] * n
    for r, i in enumerate(order[(n + 1) // 2:]):
        k, s = FIELD_CELLS[r % 9]; o[i] = f"{k}__{s}db"
    return np.array(o)
def f1_macro(y, p, k=K):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k); tp = np.diag(cm).astype(float); den = cm.sum(0) + cm.sum(1); ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
class Src:
    def __init__(self, d, cfg): self.d, self.cfg, self.cache = d, cfg, {}
    def load(self, cond):
        if cond in self.cache: return self.cache[cond]
        z = np.load(self.d / f"{self.cfg}__{cond}.npz")
        self.cache[cond] = {"y": z["y"].astype(int), "fold": z["test_fold"].astype(int), "z": z["test_logits"].astype(float),
                            "iv": {int(k[7:]): (z[k].astype(int), z["iv_logits_" + k[7:]].astype(float)) for k in z.files if k.startswith("iv_idx_")}}
        return self.cache[cond]
    def field(self):
        if "field" in self.cache: return self.cache["field"]
        b = self.load("clean"); n = len(b["y"]); at = field_assignment(n, 5001); z = b["z"].copy()
        for c in sorted(set(at) - {"clean"}):
            mk = at == c; z[mk] = self.load(c)["z"][mk]
        assert np.isfinite(z).all(); self.cache["field"] = {"y": b["y"], "fold": b["fold"], "z": z, "iv": b["iv"]}; return self.cache["field"]
cheap, ast = {}, {}
for k in ["esc", "8", "4", "bitnet"]:
    for bb in ("fp32", "w8a8"):
        cheap[f"eff:{bb}:{k}"] = {s: Src(S2 / "scores/urban" / f"effp_urban_s{s}" / k, bb) for s in SEEDS}
        for m, root in (("mn04", S3B), ("mn10", S3B), ("mn05", R), ("mn20", R), ("dymn04", R)):
            d = {s: Src(root / "scores_cnn/urban" / f"{m}_urban_s{s}" / k, f"{m}_{bb}") for s in SEEDS}
            if all((x.d / f"{x.cfg}__clean.npz").exists() for x in d.values()): cheap[f"cnn:{m}:{bb}:{k}"] = d
        cheap[f"cnn:mn10unp:{bb}:{k}"] = {s: Src(R / "scores_unp/urban" / f"mn10_urban_s{s}" / k, f"mn10_{bb}") for s in SEEDS}
    cheap[f"cnn:mn10unpr:{bb}:esc"] = {s: Src(R / "scores_unpr/urban" / f"mn10unp_urban_s{s}" / "esc", f"mn10_{bb}") for s in SEEDS}
for h in ["esc", "4"]:
    for cfg in ["w4g128", "n4", "w4", "w5", "w6", "w8g64", "fp32"]:
        d = {s: Src(I5 / "scores/urban" / f"asturban_s{s}" / h, cfg) for s in SEEDS}
        if all((x.d / f"{cfg}__clean.npz").exists() for x in d.values()): ast[f"ast:{cfg}+{h}"] = d
    for cfg in ["fp32", "w8g64", "w4g128", "n4", "w3g64", "w4"]:
        ast[f"ast:{cfg}@{SL}+{h}"] = {s: Src(S2 / "scores_ast_short/urban" / f"astshort_urban_s{s}" / h, cfg) for s in SEEDS}
    for g, sfx in (("ast256", "@256"), ("ast384", "@384")):
        for cfg in ["n4", "w4g128"]:
            ast[f"ast:{cfg}{sfx}+{h}"] = {s: Src(S3B / f"scores_{g}/urban" / f"{g}_urban_s{s}" / h, cfg) for s in SEEDS}
def cost(name):
    key = name.replace("mn10unpr", "mn10unp")
    return costs["cheap"][key] if key in costs["cheap"] else costs["ast"][name[4:]]
systems = {**cheap, **ast}; [cost(nm) for nm in systems]
ref = cheap["eff:fp32:esc"][0].load("clean"); y_all = ref["y"]; folds = sorted(ref["iv"]); n = len(y_all); te_idx = {f: np.flatnonzero(ref["fold"] == f) for f in folds}
print(f"{len(cheap)} cheap statics (incl. mn10unp and mn10unpr), {len(ast)} AST statics", flush=True)
pred, stat = {}, []
for nm, d in systems.items():
    for s in SEEDS:
        r = d[s].field(); assert (r["y"] == y_all).all() and (r["fold"] == ref["fold"]).all()
        for f in folds:
            te = te_idx[f]; p = r["z"][te].argmax(1); pred[(nm, s, f)] = p; stat.append({"system": nm, "seed": s, "fold": f, "f1": f1_macro(y_all[te], p), "cost": cost(nm)})
S = pd.DataFrame(stat)
def hull_pair(sp, c):
    lo_set, hi_set = sp[sp.cost <= c * (1 + 1e-12)], sp[sp.cost > c * (1 + 1e-12)]; best = None
    for lo in lo_set.itertuples():
        cands = [(lo.f1, lo.system, lo.system, 0.0)] + [((1 - (c - lo.cost) / (hi.cost - lo.cost)) * lo.f1 + (c - lo.cost) / (hi.cost - lo.cost) * hi.f1, lo.system, hi.system, (c - lo.cost) / (hi.cost - lo.cost)) for hi in hi_set.itertuples()]
        for v in cands:
            if best is None or v[0] > best[0] + 1e-15: best = v
    return best
def mix_f1_exact(y, plo, phi, lam, k=K):
    agree = plo == phi; tot = 0.0
    for c in range(k):
        yc = y == c; n_c = int(yc.sum()); assert n_c > 0; both = agree & (plo == c); T0, N0 = int((both & yc).sum()), int(both.sum())
        lo_only, hi_only = (plo == c) & ~agree, (phi == c) & ~agree
        a1, a2 = int((lo_only & yc).sum()), int((lo_only & ~yc).sum()); b1, b2 = int((hi_only & yc).sum()), int((hi_only & ~yc).sum())
        pu = np.convolve(stats.binom.pmf(np.arange(a1 + 1), a1, 1 - lam), stats.binom.pmf(np.arange(b1 + 1), b1, lam))
        pv = np.convolve(stats.binom.pmf(np.arange(a2 + 1), a2, 1 - lam), stats.binom.pmf(np.arange(b2 + 1), b2, lam))
        u = np.arange(len(pu))[:, None]; v = np.arange(len(pv))[None, :]; tot += float(pu @ (2 * (T0 + u) / (n_c + N0 + u + v)) @ pv)
    return tot / k
n_te = np.array([len(te_idx[f]) for f in folds]); NB = float(np.mean(n_te / (n - n_te)))
def tests(d_pairs, keys):
    fm = np.array([np.mean([d for d, kk in zip(d_pairs, keys) if kk[1] == f]) for f in folds]); k = len(fm); sd = fm.std(ddof=1)
    t_nb = fm.mean() / np.sqrt((1 / k + NB) * sd ** 2)
    return {"diff_pts": 100 * fm.mean(), "fold_diffs_pts": (100 * fm).round(2).tolist(), "p_t_fold": float(stats.ttest_1samp(fm, 0).pvalue), "p_nb": float(2 * stats.t.sf(abs(t_nb), k - 1))}
def cascade(cheap_nm, tier_nm, rule, rate):
    rows = []
    for s in SEEDS:
        rc = cheap[cheap_nm][s]; rcc = rc.load("clean"); rx = rc.field(); rax = ast[tier_nm][s].field()
        for f in folds:
            piv = softmax(rcc["iv"][f][1]); miv = np.sort(piv, 1); miv = miv[:, -1] - miv[:, -2]; u = np.unique(miv)
            cuts = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]]); rates = (miv[None, :] < cuts[:, None]).mean(1); thr = cuts[np.lexsort((rates, np.abs(rates - rate)))[0]]
            te = te_idx[f]; p = softmax(rx["z"][te]); pe = softmax(rax["z"][te]); sp = np.sort(p, 1); escm = (sp[:, -1] - sp[:, -2]) < thr
            de = pe.argmax(1) if rule == "ast" else ((p + pe) / 2).argmax(1); pr = np.where(escm, de, p.argmax(1)); r_ = float(escm.mean())
            rows.append({"seed": s, "fold": f, "f1": f1_macro(y_all[te], pr), "rate": r_, "cost": cost(cheap_nm) + r_ * cost(tier_nm)})
    return rows
res = {}
for label, cheap_nm, hull_set in (("P2 padded (S4 row)", "cnn:mn10:w8a8:esc", "no_unp"), ("P5b exact-length, padded heads (stage 3c P5b row)", "cnn:mn10unp:w8a8:esc", "unp"),
                                  ("exact-length, heads retrained at exact length (post hoc)", "cnn:mn10unpr:w8a8:esc", "unpr"), ("exact-length retrained, hull without any exact-length static", "cnn:mn10unpr:w8a8:esc", "no_unp")):
    rows = cascade(cheap_nm, f"ast:w8g64@{SL}+esc", "avg", 0.30); c_mean = float(np.mean([r["cost"] for r in rows]))
    keep = {"no_unp": lambda nm: "unp" not in nm, "unp": lambda nm: "unpr" not in nm, "unpr": lambda nm: "mn10unp:" not in nm}[hull_set]
    sp = S[S.system.map(keep)].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index(); v, lo, hi, lam = hull_pair(sp, c_mean)
    hull = [f1_macro(y_all[te_idx[r["fold"]]], pred[(lo, r["seed"], r["fold"])]) if lam == 0 else mix_f1_exact(y_all[te_idx[r["fold"]]], pred[(lo, r["seed"], r["fold"])], pred[(hi, r["seed"], r["fold"])], lam) for r in rows]
    d = [r["f1"] - h for r, h in zip(rows, hull)]
    res[label] = {"cheap": cheap_nm, "f1": float(np.mean([r["f1"] for r in rows])), "cost_T": c_mean / 1e12, "rate": float(np.mean([r["rate"] for r in rows])), "hull_lo": lo, "hull_hi": hi, "lam": lam,
                  "hull_f1": float(np.mean(hull)), **tests(d, [(r["seed"], r["fold"]) for r in rows])}
    print(label, json.dumps(res[label]), flush=True)
st = S.groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
res["statics_field"] = {nm: {"f1": float(st[st.system == nm].f1.iloc[0]), "cost_T": float(st[st.system == nm].cost.iloc[0]) / 1e12} for nm in [x for x in
                        ["cnn:mn10:w8a8:esc", "cnn:mn10unp:w8a8:esc", "cnn:mn10unpr:w8a8:esc", "cnn:mn10:fp32:esc", "cnn:mn10unpr:fp32:esc", "cnn:mn20:fp32:bitnet", "cnn:mn20:w8a8:esc", "ast:w8g64@400+esc", "ast:fp32@400+esc"] if (st.system == x).any()]}
print(json.dumps(res["statics_field"], indent=1)); json.dump(res, open(out, "w"), indent=1)
