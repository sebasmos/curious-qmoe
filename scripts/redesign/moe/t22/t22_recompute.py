"""T22 independent recompute (ANALYSIS_PLAN.md I.6). Reads only raw score files: the coarse expert's npz files and the stored npz files
listed in T13's file_list.csv (paths and sha256), plus the cost numbers. Own code throughout (sklearn macro F1, own thresholds, own field
assembly, own hull by Monte Carlo time-sharing). Takes the chosen budget pair and the two-tier budget from tables/verdict.json and
recomputes the hierarchy, the hull, the best single, the always-average and the two-tier contrasts on the H1 bed.
    python t22_recompute.py --r R"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); a = ap.parse_args()
Q = Path.home() / "Documents/qmoe-results"
FL = pd.read_csv(Q / "2026-10-25_0145_moe-t13_code-eaaedc9/inputs/file_list.csv")
V = json.loads((a.r / "tables/verdict.json").read_text())
NF = {"esc50": 5, "urban": 10, "tau": 5}; KK = {"esc50": 50, "urban": 10, "tau": 10}; SEEDS = [0, 1, 2]
CELLS = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]


def sm(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def marg(p):
    s = np.sort(p, 1); return s[:, -1] - s[:, -2]


def assign(n, seed):
    order = np.random.default_rng(seed).permutation(n); out = np.array(["clean"] * n, dtype=object)
    for r, i in enumerate(order[(n + 1) // 2:]): out[i] = CELLS[r % 9]
    return out


def thr(m, rate):
    """Cut point with the fraction of values below it closest to rate (ties to the smaller fraction): candidates midway between values."""
    u = np.unique(m); cut = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    fr = np.searchsorted(np.sort(m), cut, side="left") / len(m)
    best = np.argmin(np.abs(fr - rate) * 1e6 + fr * 1e-3)
    return cut[best]


def f1(y, p, k): return f1_score(y, p, average="macro", labels=np.unique(np.concatenate([y, p]))) * 100


_c = {}
def npz(path):
    if path not in _c:
        _c[path] = dict(np.load(path))
    return _c[path]


def stored(task, ex, cond, s):
    r = FL[(FL.task == task) & (FL.expert == ex) & (FL.cond == cond) & (FL.seed == s)]
    assert len(r) == 1, (task, ex, cond, s)
    p = r.iloc[0].path
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != r.iloc[0].sha256: raise SystemExit(f"sha256 differs: {p}")
    return npz(p)


def get(task, ex, cond, s):
    """(y, fold, test probs, {f: (iv idx, iv probs)}) with the field mix assembled from clean + nine cells."""
    if ex == "coarse":
        d = a.r / "scores" / task / f"coarse_{task}_s{s}"
        ld = lambda c: npz(str(d / f"coarse_w8a8__{c}.npz"))  # noqa: E731
    else:
        stored(task, ex, "clean", s)  # sha256 check of the clean file
        cp = FL[(FL.task == task) & (FL.expert == ex) & (FL.cond == "clean") & (FL.seed == s)].iloc[0].path
        # cells inside the field mix are read from the same folder (some are stored for the field rows only, so not in file_list)
        ld = lambda c: npz(cp.replace("__clean.npz", f"__{c}.npz"))  # noqa: E731
    if task == "tau":
        z = npz(str(a.r / "scores/tau" / f"coarse_tau_s{s}/coarse_w8a8__all.npz")) if ex == "coarse" else stored(task, ex, "clean", s)
        return z["y"], z["fold"].astype(int), sm(z["test_logits"].astype(np.float64)), {f: (z[f"iv_idx_{f}"], sm(z[f"iv_logits_{f}"].astype(np.float64))) for f in range(1, 6)}, z["device"].astype(str)
    if cond != "field":
        z = ld(cond)
        return z["y"], z["test_fold"].astype(int), sm(z["test_logits"].astype(np.float64)), {f: (z[f"iv_idx_{f}"], sm(z[f"iv_logits_{f}"].astype(np.float64))) for f in range(1, NF[task] + 1)}, None
    base = ld("clean"); n = len(base["y"]); at, av = assign(n, 5001), assign(n, 5002)
    Z = np.full(base["test_logits"].shape, np.nan); ivs = {f: (base[f"iv_idx_{f}"], np.full(base[f"iv_logits_{f}"].shape, np.nan)) for f in range(1, NF[task] + 1)}
    for c in ["clean"] + CELLS:
        z = ld(c); m = at == c; Z[m] = z["test_logits"][m]
        for f, (idx, v) in ivs.items():
            s_ = av[idx] == c; v[s_] = z[f"iv_logits_{f}"][s_]
    assert np.isfinite(Z).all()
    return base["y"], base["test_fold"].astype(int), sm(Z), {f: (i, sm(v)) for f, (i, v) in ivs.items()}, None


def hier(p1, p2, p3, t1, t2):
    e2 = marg(p1) < t1; p12 = (p1 + p2) / 2; e3 = e2 & (marg(p12) < t2)
    return np.where(e3[:, None], (p1 + p2 + p3) / 3, np.where(e2[:, None], p12, p1)).argmax(1), e2, e3


rng = np.random.default_rng(7)
out = {}
for task, T in V["tasks"].items():
    K = KK[task]; bed = T["bed"]; cond = "clean" if task == "tau" else bed
    r2, r3 = int(T["pair"][1:3]) / 100, int(T["pair"][4:6]) / 100; b2 = float(T["two_tier_budget"])
    C = json.loads((a.r / "inputs/p2_costs.json").read_text())["tasks"][task]["mn10"]["total_bitops"]
    ct = {}
    if task == "tau":
        cj = json.loads((Q / "2026-10-23_2133_redesign-idea6-optionB_code-f320295/analysis/tau/costs.json").read_text())
        ct = {"mn10": cj["cheap"]["cnn:mn10:w8a8:esc"], "mn20": cj["cheap"]["cnn:mn20:w8a8:esc"], "astfp32": cj["ast"]["ast:fp32+esc"], "astw4": cj["ast"]["ast:w4g128+esc"]}
    else:
        cj = json.loads((Q / f"2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1/analysis/{task}/costs.json").read_text()); L = 512 if task == "esc50" else 400
        ct = {"mn10": cj["cheap"]["cnn:mn10unp:w8a8:esc" if task == "urban" else "cnn:mn10:w8a8:esc"], "mn20": cj["cheap"]["cnn:mn20:w8a8:esc"],
              "astfp32": cj["ast"]["fp32+esc"], "astw4": cj["ast"]["w4g128+esc"], "astshort": cj["ast"][f"w4g128@{L}+esc"]}
    ct["coarse"] = C
    ex = list(ct)
    rows = []
    for s in SEEDS:
        G = {e: get(task, e, cond, s) for e in ex}
        Gf = {e: get(task, e, "field" if task != "tau" else "clean", s) for e in ex}
        Gc = {e: get(task, e, "clean", s) for e in ("coarse", "mn10", "astw4")}
        y, fold = G["coarse"][0], G["coarse"][1]
        for f in range(1, NF[task] + 1):
            idx = Gc["coarse"][3][f][0]; q1, q2, q3 = (Gc[e][3][f][1] for e in ("coarse", "mn10", "astw4"))
            t1 = thr(marg(q1), r2); e2 = marg(q1) < t1; t2 = thr(marg((q1 + q2) / 2)[e2], r3 / r2)
            tb = thr(marg(q2), b2)
            _, e2i, e3i = hier(q1, q2, q3, t1, t2); civ = np.mean(C + e2i * ct["mn10"] + e3i * ct["astw4"])
            tcm = thr(marg(q2), float(np.clip((civ - ct["mn10"]) / ct["astw4"], 0, 1)))
            te = fold == f
            if task == "tau": te &= (G["coarse"][4] != "a") if bed == "shift" else (G["coarse"][4] == "a")
            yt = y[te]; P = {e: G[e][2][te] for e in ex}
            ph, e2t, e3t = hier(P["coarse"], P["mn10"], P["astw4"], t1, t2)
            r = {"seed": s, "fold": f, "hier": f1(yt, ph, K), "cost": float(np.mean(C + e2t * ct["mn10"] + e3t * ct["astw4"]))}
            for e in ex: r[e] = f1(yt, P[e].argmax(1), K); r["pred_" + e] = P[e].argmax(1)
            for nm, parts in (("avg3", ("coarse", "mn10", "astw4")), ("avg_c_m", ("coarse", "mn10")), ("avg_c_a", ("coarse", "astw4")), ("avg_m_a", ("mn10", "astw4"))):
                pp = sum(P[e] for e in parts).argmax(1); r[nm] = f1(yt, pp, K); r["pred_" + nm] = pp
            for nm, t_ in (("two_own", tb), ("two_cm", tcm)):
                ee = marg(P["mn10"]) < t_; pp = np.where(ee, ((P["mn10"] + P["astw4"]) / 2).argmax(1), P["mn10"].argmax(1))
                r[nm] = f1(yt, pp, K); r[nm + "_cost"] = float(np.mean(ct["mn10"] + ee * ct["astw4"]))
            r["y"] = yt
            for e in ex:
                ii, pv = Gf[e][3][f]; r["iv_" + e] = f1(y[ii], pv.argmax(1), K)
            rows.append(r)
    D = pd.DataFrame(rows)
    sc = {"avg3": ct["coarse"] + ct["mn10"] + ct["astw4"], "avg_c_m": ct["coarse"] + ct["mn10"], "avg_c_a": ct["coarse"] + ct["astw4"], "avg_m_a": ct["mn10"] + ct["astw4"]} | ct
    mean_f1 = {k: D[k].mean() for k in sc}; Ch = D.cost.mean()
    best_v, best_h = -1, None
    for lo in sc:
        if sc[lo] > Ch * (1 + 1e-12): continue
        if mean_f1[lo] > best_v: best_v, best_h = mean_f1[lo], (lo, lo, 0.0)
        for hi in sc:
            if sc[hi] <= Ch * (1 + 1e-12): continue
            lam = (Ch - sc[lo]) / (sc[hi] - sc[lo]); v = (1 - lam) * mean_f1[lo] + lam * mean_f1[hi]
            if v > best_v + 1e-12: best_v, best_h = v, (lo, hi, lam)
    lo, hi, lam = best_h
    hull = []
    for _, r in D.iterrows():
        if lam == 0: hull.append(r[lo]); continue
        dr = rng.random((400, len(r["y"]))) < lam
        hull.append(np.mean([f1(r["y"], np.where(m, r["pred_" + hi], r["pred_" + lo]), K) for m in dr]))
    D["hull"] = hull
    cands = [e for e in ex if ct[e] <= Ch * (1 + 1e-12)]; ivm = {e: D["iv_" + e].mean() for e in cands}; best = max(cands, key=lambda e: ivm[e])
    fm = lambda col: (D.hier - D[col]).groupby(D.fold).mean()  # noqa: E731
    o = {"hier_f1": D.hier.mean(), "cost": Ch, "hull": f"{lo}|{hi}|{lam:.4f}", "hull_f1": D.hull.mean(), "best_single": best, "best_f1": D[best].mean(), "avg3_f1": D.avg3.mean(),
         "d_hull": fm("hull").mean(), "d_best": fm(best).mean(), "d_avg3": fm("avg3").mean(), "d_two_own": fm("two_own").mean(), "d_two_cm": fm("two_cm").mean(),
         "two_own_cost": D.two_own_cost.mean(), "two_cm_cost": D.two_cm_cost.mean(), "n_folds": int(D.fold.nunique())}
    H = T["H1"]; o["reported"] = {"hier_f1": H["f1"], "hull_f1": H["hull_f1"], "best_single": H["best_single"], "d_hull": H["hull_d"], "d_best": H["best_d"], "d_avg3": H["avg3_d"],
                                  "d_two_own": T["H2"]["d_two_own"], "d_two_cm": T["H2"]["d_two_cm"]}
    o["abs_diff"] = {k: abs(float(o[k]) - float(v)) for k, v in o["reported"].items() if k != "best_single"}
    o["best_single_same"] = best == H["best_single"]
    out[task] = o; print(task, json.dumps({k: v for k, v in o.items() if k not in ("reported",)}, default=float), flush=True)
(a.r / "tables/recompute.json").write_text(json.dumps(out, indent=1, default=float))
