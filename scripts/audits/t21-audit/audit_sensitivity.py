"""Sensitivity of the field and clean gains: (a) drop test clips with a cross-fold near duplicate (MN10 FP32 cosine),
(b) MN10 cost scaled x0.5, x2, x4, x10, (c) per-fold hull at the fold's own cascade cost, (d) hull at lambda = rate.
Own code. Usage: python -I audit_sensitivity.py <results_dir> <audit_dir>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
R, A = Path(sys.argv[1]), Path(sys.argv[2])
S, F = range(5), range(1, 6)
def sm(z):
    z = z.astype(np.float64); z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def mf1(y, p): return 100 * f1_score(y, p, average="macro", zero_division=0)
def marg(p): q = np.partition(p, -2, 1); return q[:, -1] - q[:, -2]
def thr(m, r): s = np.sort(m); k = int(round(r * len(s))); return (s[k - 1] + s[k]) / 2
Z = {t: {s: dict(np.load(R / f"scores/d1/pool_{t}_s{s}.npz")) for s in S} for t in ("mn10_w8a8", "ast_w4g128")}
y = Z["mn10_w8a8"][0]["y"].astype(int); fold = Z["mn10_w8a8"][0]["fold"].astype(int)
cc0 = np.load(A / "cost_mn_per_clip.npy"); ca = np.load(A / "cost_ast_per_clip.npy")
X = np.load(R / "emb/d1/mn_pool/mn10_fp32__clean.npy").astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True)
nn = np.zeros(len(y))
for i0 in range(0, len(y), 500):
    Sm = X[i0:i0 + 500] @ X.T
    Sm[fold[i0:i0 + 500, None] == fold[None, :]] = -1
    nn[i0:i0 + 500] = Sm.max(1)
def hull(pts, c):
    best = None
    for lo, (cl, fl) in pts.items():
        if cl > c: continue
        cand = [(fl, lo, lo, 0.0)] + [((1 - (c - cl) / (ch - cl)) * fl + (c - cl) / (ch - cl) * fh, lo, hi, (c - cl) / (ch - cl)) for hi, (ch, fh) in pts.items() if ch > c]
        for v in cand:
            if best is None or v[0] > best[0]: best = v
    return best
def run(cond, keep, cc, perfold=False, lam_rate=False):
    rows = []
    for s in S:
        for f in F:
            iv = Z["mn10_w8a8"][s][f"iv_idx_{f}"]
            t_ = thr(marg(sm(Z["mn10_w8a8"][s][f"iv__{f}__clean__refit"])), 0.20)
            te = np.flatnonzero((fold == f) & keep); yt = y[te]
            pc = sm(Z["mn10_w8a8"][s][f"test__{cond}__refit"][te]); pa = sm(Z["ast_w4g128"][s][f"test__{cond}__refit"][te])
            e = marg(pc) < t_
            pr = {"mn": pc.argmax(1), "ast": pa.argmax(1), "avg": ((pc + pa) / 2).argmax(1)}; pr["cas"] = np.where(e, pr["avg"], pr["mn"])
            co = {"mn": cc[te].mean(), "ast": ca[te].mean(), "avg": (cc[te] + ca[te]).mean(), "cas": (cc[te] + e * ca[te]).mean()}
            for k in pr: rows.append((s, f, k, mf1(yt, pr[k]), co[k], e.mean()))
    T = pd.DataFrame(rows, columns=["s", "f", "sys", "f1", "cost", "rate"])
    def d_for(g):
        pts = {k: (g[g.sys == k].cost.mean(), g[g.sys == k].f1.mean()) for k in ("mn", "ast", "avg")}
        c = g[g.sys == "cas"].cost.mean()
        if lam_rate:
            r = g[g.sys == "cas"].rate.mean(); h = (1 - r) * pts["mn"][1] + r * pts["avg"][1]
            return g[g.sys == "cas"].f1.mean() - h
        h = hull(pts, c); sf = g.set_index(["sys", "s", "f"]).f1
        return np.mean([sf[("cas", s, f)] - ((1 - h[3]) * sf[(h[1], s, f)] + h[3] * sf[(h[2], s, f)]) for s, f in set(zip(g.s, g.f))])
    if perfold or lam_rate:
        dfold = np.array([d_for(T[T.f == f]) for f in F])
    else:
        pts = {k: (T[T.sys == k].cost.mean(), T[T.sys == k].f1.mean()) for k in ("mn", "ast", "avg")}
        h = hull(pts, T[T.sys == "cas"].cost.mean()); sf = T.set_index(["sys", "s", "f"]).f1
        dfold = np.array([np.mean([sf[("cas", s, f)] - ((1 - h[3]) * sf[(h[1], s, f)] + h[3] * sf[(h[2], s, f)]) for s in S]) for f in F])
    k = 5; sd = dfold.std(ddof=1); tt = dfold.mean() / np.sqrt((1 / k + 0.25) * sd ** 2)
    return dfold.mean(), 2 * stats.t.sf(abs(tt), k - 1), int((dfold > 0).sum())
out = []
allk = np.ones(len(y), bool)
for cond in ("field", "clean"):
    out.append((cond, "baseline (my costs)", *run(cond, allk, cc0)))
    for th in (0.9999, 0.999):
        kp = nn <= th; out.append((cond, f"drop {int((~kp).sum())} clips with cross-fold MN10 cosine > {th}", *run(cond, kp, cc0)))
    for mult in (0.5, 2, 4, 10):
        out.append((cond, f"MN10 cost x{mult}", *run(cond, allk, cc0 * mult)))
    out.append((cond, "hull per fold at the fold's own cascade cost", *run(cond, allk, cc0, perfold=True)))
    out.append((cond, "per fold, hull at lambda = escalation rate (length-free AST cost)", *run(cond, allk, cc0, lam_rate=True)))
O = pd.DataFrame(out, columns=["cond", "variant", "d", "p_nb", "folds_pos"]); O.to_csv(A / "audit_sensitivity.csv", index=False)
pd.set_option("display.width", 200); print(O.round(4).to_string())
