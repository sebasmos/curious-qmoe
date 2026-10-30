"""T23 separate recompute of the primary numbers (ANALYSIS_PLAN.md C12), own code: reads only the raw score files (stored and ours),
the cost constants and the budgets chosen by the analysis (tables/verdict.json), and recomputes on the shifted bed and clean: both systems'
fold-level macro F1 and cost, the equal-cost hull (own envelope, own exact expectation of random time-sharing), the best single (own
inner-validation F1), the equal-cost two-tier cascade, and D_hull, D_best, D_two. Writes tables/recompute_check.csv against tables/primary.csv.
Does not import t23_analyze.py, stage 2's analyze.py or t13_common.py.    python t23_recompute.py --r R"""
import argparse, json
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
Q = Path.home() / "Documents/qmoe-results"
S2, S3B, S3C, I5 = (Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba", Q / "2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db",
                    Q / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1", Q / "2026-10-20_2359_redesign-idea5-stage2_code-3c18c38")
OPTB, UNP = Q / "2026-10-23_2133_redesign-idea6-optionB_code-f320295", Q / "2026-10-23_1918_redesign-idea6-stage3c-unpadded_code-cbf6be6"
TASKS = {"esc50": (50, [0, 1, 2], 5), "urban": (10, [0, 1, 2], 10), "tau": (10, [0, 1, 2], 5)}
AST = {"esc50": "astshort", "urban": "astshort", "tau": "astw4"}; EXTRA = {"esc50": ["mn20", "astw4", "astfp32"], "urban": ["mn20", "astw4", "astfp32"], "tau": ["mn20", "astfp32"]}
FIELD = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); a = ap.parse_args(); R = a.r
if not (lambda al: al.get("all_ok") and al.get("stage") == "post")(json.load(open(R / "inputs/alignment.json"))): raise SystemExit("alignment not all_ok")
V = json.load(open(R / "tables/verdict.json"))

def path(task, e, seed, cond):
    if task == "tau":
        if e == "wave": return R / f"scores/tau/wg_tau_s{seed}/esc/wg__all.npz"
        n = {"mn10": f"mn10_tau_s{seed}/esc/mn10_w8a8", "mn20": f"mn20_tau_s{seed}/esc/mn20_w8a8", "astw4": f"ast_tau_s{seed}/esc/w4g128", "astfp32": f"ast_tau_s{seed}/esc/fp32"}[e]
        return OPTB / f"scores/{n}__all.npz"
    if e == "wave": return R / f"scores/{task}/wg_{task}_s{seed}/esc/wg__{cond}.npz"
    if e == "mn10": d = (UNP / f"scores_unpr/urban/mn10unp_urban_s{seed}") if task == "urban" else (S3B / f"scores_cnn/{task}/mn10_{task}_s{seed}"); c = "mn10_w8a8"
    elif e == "mn20": d = S3C / f"scores_cnn/{task}/mn20_{task}_s{seed}"; c = "mn20_w8a8"
    elif e in ("astw4", "astfp32"): d = I5 / f"scores/{task}/ast{task}_s{seed}"; c = "w4g128" if e == "astw4" else "fp32"
    else: d = S2 / f"scores_ast_short/{task}/astshort_{task}_s{seed}"; c = "w4g128"
    return d / "esc" / f"{c}__{cond}.npz"

def field_assign(n, seed):
    order = np.random.default_rng(seed).permutation(n); out = np.array(["clean"] * n, dtype=object)
    for r, i in enumerate(order[(n + 1) // 2:]): out[i] = FIELD[r % 9]
    return out

def load(task, e, seed, cond):
    """(y, fold, device or None, test logits, {fold: (iv_idx, iv logits)})"""
    if task == "tau" or cond != "field":
        z = np.load(path(task, e, seed, "clean" if task == "tau" else cond))
        iv = {int(k[7:]): (z[k], z["iv_logits_" + k[7:]]) for k in z.files if k.startswith("iv_idx_")}
        return z["y"].astype(int), (z["fold"] if task == "tau" else z["test_fold"]).astype(int), (z["device"].astype(str) if task == "tau" else None), z["test_logits"].astype(float), iv
    y, fo, _, zc, ivc = load(task, e, seed, "clean"); at, av = field_assign(len(y), 5001), field_assign(len(y), 5002)
    z = zc.copy(); iv = {f: (i, l.copy()) for f, (i, l) in ivc.items()}
    for c in FIELD:
        _, _, _, zz, ivz = load(task, e, seed, c); m = at == c; z[m] = zz[m]
        for f, (i, l) in iv.items(): sel = av[i] == c; l[sel] = ivz[f][1][sel]
    return y, fo, None, z, iv

def sm(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def mg(p): s = np.sort(p, 1); return s[:, -1] - s[:, -2]
def thr(m, rate):
    u = np.unique(m); cand = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]]); rr = np.array([(m < c).mean() for c in cand])
    best = np.lexsort((rr, np.abs(rr - rate)))[0]; return cand[best]
def F1(y, p): return 100 * f1_score(y, p, average="macro", zero_division=0)

def exp_mix_f1(y, plo, phi, lam, k):
    """Exact E[macro F1] (every class in y) when each clip takes phi with probability lam: per class, the joint law of the variable
    true-positive and predicted counts by dynamic programming over the clips whose two predictions differ."""
    tot = 0.0
    for c in range(k):
        yc = y == c; n_c = yc.sum(); fixed = plo == phi; tp0 = int(((plo == c) & yc & fixed).sum()); pp0 = int(((plo == c) & fixed).sum())
        var = np.flatnonzero(~fixed & ((plo == c) | (phi == c)))
        dist = {(0, 0): 1.0}
        for i in var:
            nd = {}
            for (tp, pp), pr in dist.items():
                for take_hi, w in ((False, 1 - lam), (True, lam)):
                    pc = (phi[i] if take_hi else plo[i]) == c; key = (tp + int(pc and yc[i]), pp + int(pc)); nd[key] = nd.get(key, 0.0) + pr * w
            dist = nd
        tot += sum(pr * 2 * (tp0 + tp) / (n_c + pp0 + pp) for (tp, pp), pr in dist.items())
    return 100 * tot / k

def envelope(points, c):
    """Best static or two-point time-sharing at cost c: points {name: (cost, f1)} -> (lo, hi, lam)."""
    best = None
    for lo, (cl, fl) in points.items():
        if cl > c * (1 + 1e-12): continue
        cands = [(fl, lo, lo, 0.0)] + [(fl + (c - cl) / (ch - cl) * (fh - fl), lo, hi, (c - cl) / (ch - cl)) for hi, (ch, fh) in points.items() if ch > c * (1 + 1e-12)]
        for v in cands:
            if best is None or v[0] > best[0] + 1e-15: best = v
    return best[1:]

cost_esc = {t: json.load(open(S3C / f"analysis/{t}/costs.json")) for t in ("esc50", "urban")}; cost_tau = json.load(open(OPTB / "analysis/tau/costs.json"))
def costs(task):
    if task == "tau": c = dict(mn10=cost_tau["cheap"]["cnn:mn10:w8a8:esc"], mn20=cost_tau["cheap"]["cnn:mn20:w8a8:esc"], astw4=cost_tau["ast"]["ast:w4g128+esc"], astfp32=cost_tau["ast"]["ast:fp32+esc"])
    else:
        cc = cost_esc[task]; L = 512 if task == "esc50" else 400
        c = dict(mn10=cc["cheap"]["cnn:mn10unp:w8a8:esc" if task == "urban" else "cnn:mn10:w8a8:esc"], mn20=cc["cheap"]["cnn:mn20:w8a8:esc"], astfp32=cc["ast"]["fp32+esc"],
                 astw4=cc["ast"]["w4g128+esc"], astshort=cc["ast"][f"w4g128@{L}+esc"])
    c["wave"] = json.load(open(R / "inputs/p2_costs.json"))[task]["wave_q8_bitops_padded"]; return c

rows = []
for task, (K, seeds, nf) in TASKS.items():
    ast = AST[task]; C = costs(task); core = ["mn10", "wave", ast]; bud = V["budgets_chosen"][task]
    ra = float(bud["a"]); r2, r3 = [float(x) for x in bud["b"].strip("()").split(",")]
    for bed in (("shift" if task == "tau" else "field"), "clean"):
        Lc = {(e, s): load(task, e, s, "clean") for e in core for s in seeds}
        Lb = {(e, s): load(task, e, s, bed if task != "tau" else "clean") for e in core + EXTRA[task] for s in seeds}
        fam = "field" if bed == "field" else "clean"
        Lf = Lb if fam == "field" else Lc
        bs = {e: np.mean([F1(Lf[(e, s)][0][Lf[(e, s)][4][f][0]], Lf[(e, s)][4][f][1].argmax(1)) for s in seeds for f in range(1, nf + 1)]) for e in core}
        best = max(bs, key=bs.get)
        per = {}
        for s in seeds:
            y, fo, dev = Lc[("mn10", s)][0], Lc[("mn10", s)][1], Lc[("mn10", s)][2]
            for f in range(1, nf + 1):
                iv = Lc[("mn10", s)][4][f][0]; pm, pw, pa = (sm(Lc[(e, s)][4][f][1]) for e in core)
                ta = thr(mg(pm), ra); ea = mg(pm) < ta; c_a = C["mn10"] + ea.mean() * C["wave"]
                t2 = thr(mg(pm), r2); e2 = mg(pm) < t2; t3 = thr(mg((pm + pw) / 2)[e2], min(1.0, r3 * len(pm) / e2.sum())); e3 = e2 & (mg((pm + pw) / 2) < t3)
                c_b = C["mn10"] + e2.mean() * C["wave"] + e3.mean() * C[ast]
                tq = {sy: thr(mg(pm), float(np.clip((cs - C["mn10"]) / C[ast], 0, 1))) for sy, cs in (("a", c_a), ("b", c_b))}
                te = np.flatnonzero((fo == f) & ((dev == "a") if (task == "tau" and bed == "clean") else (dev != "a") if task == "tau" else True))
                yt = y[te]; P = {e: sm(Lb[(e, s)][3][te]) for e in core + EXTRA[task]}
                for n_ in (2, 3):
                    for cmb in combinations(core, n_): P["avg:" + "+".join(cmb)] = np.mean([P[e] for e in cmb], 0); C["avg:" + "+".join(cmb)] = sum(C[e] for e in cmb)
                m1 = mg(P["mn10"]); xa = m1 < ta; q = (P["mn10"] + P["wave"]) / 2
                pred_a = np.where(xa, q.argmax(1), P["mn10"].argmax(1))
                x2 = m1 < t2; x3 = x2 & (mg(q) < t3); pred_b = np.where(x3, P["avg:" + "+".join(core)].argmax(1), np.where(x2, q.argmax(1), P["mn10"].argmax(1)))
                out = {"sys_a": (F1(yt, pred_a), C["mn10"] + xa.mean() * C["wave"]), "sys_b": (F1(yt, pred_b), C["mn10"] + x2.mean() * C["wave"] + x3.mean() * C[ast])}
                for sy in ("a", "b"):
                    xt = m1 < tq[sy]; out[f"two_{sy}"] = (F1(yt, np.where(xt, ((P["mn10"] + P[ast]) / 2).argmax(1), P["mn10"].argmax(1))), 0)
                per[(s, f)] = dict(out=out, yt=yt, pred={m: P[m].argmax(1) for m in P}, statf1={m: F1(yt, P[m].argmax(1)) for m in P})
        for sysn in ("sys_a", "sys_b"):
            Cc = np.mean([v["out"][sysn][1] for v in per.values()])
            pts = {m: (C[m], np.mean([v["statf1"][m] for v in per.values()])) for m in per[(seeds[0], 1)]["statf1"]}
            lo, hi, lam = envelope(pts, Cc)
            d = {"hull": [], "best": [], "two_eq": []}; fold = []
            for (s, f), v in per.items():
                x = v["out"][sysn][0]; h = v["statf1"][lo] if lam == 0 else exp_mix_f1(v["yt"], v["pred"][lo], v["pred"][hi], lam, K)
                d["hull"].append(x - h); d["best"].append(x - v["statf1"][best]); d["two_eq"].append(x - v["out"]["two_" + sysn[-1]][0]); fold.append(f)
            fold = np.array(fold)
            r = dict(task=task, cond=bed, system=sysn, f1=np.mean([v["out"][sysn][0] for v in per.values()]), cost=Cc, hull=f"{lo}|{hi}|{lam:.4f}", best_single=best)
            for k_, vv in d.items(): vv = np.array(vv); r[f"d_{k_}"] = float(np.mean([vv[fold == f].mean() for f in np.unique(fold)]))
            rows.append(r)
RC = pd.DataFrame(rows)
P_ = pd.concat([pd.read_csv(R / f"tables/{t}_by_condition.csv") for t in TASKS])
M = RC.merge(P_[["task", "cond", "system", "f1", "cost", "hull", "best_single", "d_hull", "d_best", "d_two_eq"]], on=["task", "cond", "system"], suffixes=("_recompute", "_analysis"))
for c in ("f1", "cost", "d_hull", "d_best", "d_two_eq"): M[f"absdiff_{c}"] = (M[f"{c}_recompute"] - M[f"{c}_analysis"]).abs()
M["max_absdiff_points"] = M[[f"absdiff_{c}" for c in ("f1", "d_hull", "d_best", "d_two_eq")]].max(1); M["ok"] = (M.max_absdiff_points <= 0.01) & (M.hull_recompute == M.hull_analysis)
M.to_csv(R / "tables/recompute_check.csv", index=False)
print(M[["task", "cond", "system", "f1_recompute", "f1_analysis", "d_hull_recompute", "d_hull_analysis", "d_best_recompute", "d_two_eq_recompute", "max_absdiff_points", "ok"]].round(4).to_string())
