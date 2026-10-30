"""T23 analysis (ANALYSIS_PLAN.md C9 to C11): budget choice on clean inner validation, the two systems, the equal-cost hull, best single,
the equal-cost two-tier cascade, random gates, tests, H1, H2, H3, stop rule, verdict. READS TEST LABELS; refuses to run unless
R/inputs/alignment.json says all_ok. Stored arrays are read through t13_common (read only); ours from R/scores.
    python t23_analyze.py --r R [--tasks esc50 urban tau]"""
from __future__ import annotations
import argparse, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
for p_ in (ROOT, HERE.parent, ROOT / "scripts/redesign/idea6/stage2"): sys.path.insert(0, str(p_))
from t13_common import A, CORR, NOISE, cost_table, tau_rows, available as s_avail, load as s_load  # noqa: E402
from idea5b_common import holm, mix_expected_f1  # noqa: E402

TASKS = {"esc50": dict(K=50, seeds=[0, 1, 2], nf=5), "urban": dict(K=10, seeds=[0, 1, 2], nf=10), "tau": dict(K=10, seeds=[0, 1, 2], nf=5)}
AST = {"esc50": "astshort", "urban": "astshort", "tau": "astw4"}
EXTRA = {"esc50": ["mn20", "astw4", "astfp32"], "urban": ["mn20", "astw4", "astfp32"], "tau": ["mn20", "astfp32"]}
BUD_A = [0.05, 0.10, 0.20, 0.30, 0.50]
BUD_B = [(r2, r3) for r2 in (0.30, 0.50, 0.70) for r3 in (0.05, 0.10, 0.20, 0.30)]
BUD_T21 = [0.05, 0.10, 0.20, 0.30, 0.50]
BED = {"esc50": "field", "urban": "field", "tau": "shift"}
FAMILY, NB_BOOT, BOOT_SEED, RG_DRAWS, RG_SEED, MC_SEED = 19, 2000, 20261028, 200, 20261029, 20261030


def conds_of(task): return ["clean", "shift"] if task == "tau" else ["clean", *NOISE, "field", *CORR]


def exp_f1(y, plo, phi, lam, k):
    """Exact expected macro F1 of random time-sharing (T21's exp_f1; Monte Carlo when a class is absent from y)."""
    if lam <= 0: return A.f1_fast(y, plo, k)
    if len(np.unique(y)) == k: return float(mix_expected_f1(y, plo, phi, lam, k)[0])
    rng = np.random.default_rng(MC_SEED); acc = []
    for _ in range(8):
        dr = rng.random((500, len(y))) < lam
        acc.append(A.f1_batch(y, np.where(dr, phi[None, :], plo[None, :]), k))
    return float(np.concatenate(acc).mean())


def f1_rows(Y, P, k):
    Dn = Y.shape[0]
    idx = (np.arange(Dn)[:, None] * (k * k) + Y * k + P).ravel()
    cm = np.bincount(idx, minlength=Dn * k * k).reshape(Dn, k, k)
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float); den = cm.sum(1) + cm.sum(2)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).sum(1) / (den > 0).sum(1)


def nb_ci(fm, ratio):
    fm = np.asarray(fm, float); k = len(fm); se = np.sqrt((1 / k + ratio) * fm.var(ddof=1)); h = stats.t.ppf(0.975, k - 1) * se
    return float(fm.mean() - h), float(fm.mean() + h)


class Data:
    def __init__(self, R, task):
        self.R, self.task, self.c = R, task, {}
        self.experts = ["mn10", "wave", AST[task]] + EXTRA[task]

    def avail(self, e, cond):
        if e == "wave":
            if self.task == "tau": return all((self.R / f"scores/tau/wg_tau_s{s}/esc/wg__all.npz").exists() for s in TASKS["tau"]["seeds"])
            return all(A.ScoreDir(self.R / f"scores/{self.task}/wg_{self.task}_s{s}/esc").has("wg", cond) for s in TASKS[self.task]["seeds"])
        return s_avail(self.task, e, cond)

    def get(self, e, cond, seed):
        key = (e, cond if self.task != "tau" else "all", seed)
        if key not in self.c:
            if e != "wave": self.c[key] = s_load(self.task, e, cond, seed)
            elif self.task == "tau":
                z = np.load(self.R / f"scores/tau/wg_tau_s{seed}/esc/wg__all.npz"); iv = {int(k[7:]): z[k] for k in z.files if k.startswith("iv_idx_")}
                self.c[key] = {"y": z["y"].astype(np.int64), "fold": z["fold"].astype(int), "device": z["device"].astype(str), "z": z["test_logits"].astype(np.float64),
                               "iv": {f: (iv[f], z[f"iv_logits_{f}"].astype(np.float64)) for f in sorted(iv)}}
            else:
                if key not in self.c:
                    sd = A.ScoreDir(self.R / f"scores/{self.task}/wg_{self.task}_s{seed}/esc"); self.c[key] = sd.get("wg", cond)
        return self.c[key]


def avg_name(c): return "avg:" + "+".join(c)


def system_a(pm, pw, thr):
    esc = A.margin(pm) < thr
    return np.where(esc, ((pm + pw) / 2).argmax(1), pm.argmax(1)), esc


def system_b(pm, pw, pa, t2, t3):
    e2 = A.margin(pm) < t2; q = (pm + pw) / 2; e3 = e2 & (A.margin(q) < t3)
    return np.where(e3, ((pm + pw + pa) / 3).argmax(1), np.where(e2, q.argmax(1), pm.argmax(1))), e2, e3


def thr_b(pm, pw, r2, r3):
    """Tier 2 at rate r2 of all clips; tier 3 on the tier-2 clips with the lowest margin of mean(MN10, expert) so that r3 of all clips reach it."""
    m1 = A.margin(pm); t2 = A.thr_rate(m1, r2); e2 = m1 < t2
    mq = A.margin((pm + pw) / 2)
    if e2.sum() == 0: return t2, -np.inf
    t3 = A.thr_rate(mq[e2], min(1.0, r3 * len(pm) / e2.sum()))
    return t2, t3


def two_tier(pm, pa, thr):
    esc = A.margin(pm) < thr
    return np.where(esc, ((pm + pa) / 2).argmax(1), pm.argmax(1)), esc


def analyze_task(R, task, out):
    T = TASKS[task]; K = T["K"]; seeds = T["seeds"]; folds = list(range(1, T["nf"] + 1)); D = Data(R, task); ast = AST[task]
    C = cost_table(task); C["wave"] = json.load(open(R / "inputs/p2_costs.json"))[task]["wave_q8_bitops_padded"]
    core = ("mn10", "wave", ast); avgs = [c for n in (2, 3) for c in combinations(core, n)]
    for c in avgs: C[avg_name(c)] = sum(C[e] for e in c)
    keys = [(s, f) for s in seeds for f in folds]
    y = D.get("mn10", "clean", seeds[0])["y"]
    # ── clean inner validation ──
    IV = {}
    for s, f in keys:
        idx = D.get("mn10", "clean", s)["iv"][f][0]; P = {}
        for e in D.experts:
            if not D.avail(e, "clean"): continue
            i2, lg = D.get(e, "clean", s)["iv"][f]
            if not np.array_equal(i2, idx): raise SystemExit(f"{task} {e} s{s} f{f}: inner-validation rows differ from MN10-Q8's")
            P[e] = A.softmax(lg)
        for c in avgs: P[avg_name(c)] = np.mean([P[e] for e in c], 0)
        if task != "tau" and np.intersect1d(idx, np.flatnonzero(D.get("mn10", "clean", s)["fold"] == f)).size: raise SystemExit("inner validation overlaps test")
        IV[(s, f)] = (idx, P)
    members = [m for m in IV[keys[0]][1]]
    sp_iv = pd.DataFrame([{"system": m, "f1": np.mean([A.f1_fast(y[IV[k][0]], IV[k][1][m].argmax(1), K) for k in keys]), "cost": C[m]} for m in members])
    rng = np.random.default_rng(A.RNG_IV)

    def iv_hull_mc(c_iv, sp):
        v, lo, hi, lam = A.hull_pair(sp, c_iv); h = []
        for k in keys:
            idx, P = IV[k]; plo = P[lo].argmax(1)
            if lam == 0: h.append(A.f1_fast(y[idx], plo, K)); continue
            dr = rng.random((A.DRAWS, len(plo))) < lam
            h.append(float(A.f1_batch(y[idx], np.where(dr, P[hi].argmax(1)[None, :], plo[None, :]), K).mean()))
        return float(np.mean(h)), f"{lo}|{hi}|{lam:.3f}"

    sel, TH = [], {}
    for r in BUD_A:
        f1s, cs = [], []
        for k in keys:
            idx, P = IV[k]; t = A.thr_rate(A.margin(P["mn10"]), r); TH[("a", r, k)] = t
            pr, esc = system_a(P["mn10"], P["wave"], t); f1s.append(A.f1_fast(y[idx], pr, K)); cs.append(C["mn10"] + esc.mean() * C["wave"])
        h, hs = iv_hull_mc(float(np.mean(cs)), sp_iv)
        sel.append(dict(system="a", budget=str(r), f1_iv=100 * np.mean(f1s), cost_iv=np.mean(cs), hull_iv=100 * h, gain_iv=100 * (np.mean(f1s) - h), hull=hs))
    for r2, r3 in BUD_B:
        f1s, cs = [], []
        for k in keys:
            idx, P = IV[k]; t2, t3 = thr_b(P["mn10"], P["wave"], r2, r3); TH[("b", (r2, r3), k)] = (t2, t3)
            pr, e2, e3 = system_b(P["mn10"], P["wave"], P[ast], t2, t3); f1s.append(A.f1_fast(y[idx], pr, K))
            cs.append(C["mn10"] + e2.mean() * C["wave"] + e3.mean() * C[ast])
        h, hs = iv_hull_mc(float(np.mean(cs)), sp_iv)
        sel.append(dict(system="b", budget=f"{r2}|{r3}", f1_iv=100 * np.mean(f1s), cost_iv=np.mean(cs), hull_iv=100 * h, gain_iv=100 * (np.mean(f1s) - h), hull=hs))
    sp21 = sp_iv[sp_iv.system.isin(["mn10", ast, avg_name(("mn10", ast))])]
    for r in BUD_T21:
        f1s, cs = [], []
        for k in keys:
            idx, P = IV[k]; t = A.thr_rate(A.margin(P["mn10"]), r); TH[("t21", r, k)] = t
            pr, esc = two_tier(P["mn10"], P[ast], t); f1s.append(A.f1_fast(y[idx], pr, K)); cs.append(C["mn10"] + esc.mean() * C[ast])
        h, hs = iv_hull_mc(float(np.mean(cs)), sp21)
        sel.append(dict(system="two_tier_own", budget=str(r), f1_iv=100 * np.mean(f1s), cost_iv=np.mean(cs), hull_iv=100 * h, gain_iv=100 * (np.mean(f1s) - h), hull=hs))
    SEL = pd.DataFrame(sel); SEL.insert(0, "task", task)
    pick = {}
    for sysn in ("a", "b", "two_tier_own"):
        g = SEL[SEL.system == sysn].sort_values(["gain_iv", "cost_iv"], ascending=[False, True]); pick[sysn] = g.iloc[0].budget
    bud = {"a": float(pick["a"]), "b": tuple(float(x) for x in pick["b"].split("|")), "two_tier_own": float(pick["two_tier_own"])}
    thr = {"a": {k: TH[("a", bud["a"], k)] for k in keys}, "b": {k: TH[("b", bud["b"], k)] for k in keys}, "two_tier_own": {k: TH[("t21", bud["two_tier_own"], k)] for k in keys}}
    # equal-cost two-tier: its clean inner-validation cost equals the system's, per (seed, fold)
    eq = {}
    for sysn in ("a", "b"):
        for k in keys:
            idx, P = IV[k]
            if sysn == "a": _, esc = system_a(P["mn10"], P["wave"], thr["a"][k]); csys = C["mn10"] + esc.mean() * C["wave"]
            else: _, e2, e3 = system_b(P["mn10"], P["wave"], P[ast], *thr["b"][k]); csys = C["mn10"] + e2.mean() * C["wave"] + e3.mean() * C[ast]
            rate = float(np.clip((csys - C["mn10"]) / C[ast], 0, 1)); eq[(sysn, k)] = (A.thr_rate(A.margin(P["mn10"]), rate), rate, csys)
    # best single per condition family, on inner validation
    def best_single(fam):
        v = {}
        for e in core:
            if not D.avail(e, fam): v[e] = -np.inf; continue
            vals = []
            for s, f in keys:
                idx, lg = D.get(e, fam, s)["iv"][f]; ok = np.isfinite(lg).all(1); vals.append(A.f1_fast(y[idx][ok], lg[ok].argmax(1), K))
            v[e] = float(np.mean(vals))
        return max(v, key=v.get), v
    # ── test ──
    rows, au, preds = [], [], {}
    for cond in conds_of(task):
        av = [e for e in D.experts if D.avail(e, "clean" if (task == "tau") else cond)]
        if not {"mn10", "wave"} <= set(av): continue
        for s, f in keys:
            Dm = D.get("mn10", cond, s)
            te = tau_rows(Dm, f, cond) if task == "tau" else np.flatnonzero(Dm["fold"] == f)
            P = {}
            for e in av:
                De = D.get(e, cond, s)
                if not (np.array_equal(De["y"], y) and np.array_equal(De["fold"], Dm["fold"])): raise SystemExit(f"{task} {e} {cond}: rows differ")
                P[e] = A.softmax(De["z"][te])
                if not np.isfinite(P[e]).all(): raise SystemExit(f"{task} {e} {cond} s{s} f{f}: missing test logits")
            for c in avgs:
                if all(e in P for e in c): P[avg_name(c)] = np.mean([P[e] for e in c], 0)
            yt = y[te]; pr = {m: P[m].argmax(1) for m in P}; cost = {m: C[m] for m in P}
            pa_, esc = system_a(P["mn10"], P["wave"], thr["a"][(s, f)]); pr["sys_a"] = pa_; cost["sys_a"] = C["mn10"] + esc.mean() * C["wave"]
            t2_, t3_ = thr["b"][(s, f)]; e2_ = A.margin(P["mn10"]) < t2_; e3_ = e2_ & (A.margin((P["mn10"] + P["wave"]) / 2) < t3_)
            rec = {"rate_a": esc.mean(), "rate_b2": e2_.mean(), "rate_b3": e3_.mean()}  # escalation of (b) needs no AST logits
            if ast in P:
                pb, e2, e3 = system_b(P["mn10"], P["wave"], P[ast], *thr["b"][(s, f)]); pr["sys_b"] = pb; cost["sys_b"] = C["mn10"] + e2.mean() * C["wave"] + e3.mean() * C[ast]
                for nm, tk in (("two_eq_a", ("a", (s, f))), ("two_eq_b", ("b", (s, f)))):
                    p2, e = two_tier(P["mn10"], P[ast], eq[tk][0]); pr[nm] = p2; cost[nm] = C["mn10"] + e.mean() * C[ast]
                p2, e = two_tier(P["mn10"], P[ast], thr["two_tier_own"][(s, f)]); pr["two_own"] = p2; cost["two_own"] = C["mn10"] + e.mean() * C[ast]
                assert np.array_equal(e2, e2_) and np.array_equal(e3, e3_)
                # random gate for (b): same counts, Monte Carlo
                rg = np.random.default_rng(RG_SEED + 1000 * s + f); n2, n3 = int(e2.sum()), int(e3.sum()); vals = []
                q2 = P[avg_name(("mn10", "wave"))].argmax(1); q3 = P[avg_name(core)].argmax(1)
                for _ in range(RG_DRAWS):
                    perm = rg.permutation(len(yt)); r2 = np.zeros(len(yt), bool); r2[perm[:n2]] = True; r3 = np.zeros(len(yt), bool); r3[perm[:n3]] = True
                    vals.append(A.f1_fast(yt, np.where(r3, q3, np.where(r2, q2, pr["mn10"])), K))
                rows.append(dict(task=task, cond=cond, seed=s, fold=f, system="rgate_b", f1=100 * np.mean(vals), cost=cost["sys_b"]))
            for m in pr:
                rows.append(dict(task=task, cond=cond, seed=s, fold=f, system=m, f1=100 * A.f1_fast(yt, pr[m], K), acc=100 * float((pr[m] == yt).mean()), cost=cost[m], n=len(te), **rec))
            rows.append(dict(task=task, cond=cond, seed=s, fold=f, system="rgate_a", f1=100 * exp_f1(yt, pr["mn10"], pr[avg_name(("mn10", "wave"))], esc.mean(), K), cost=cost["sys_a"]))
            right = {e: pr[e] == yt for e in core if e in pr}
            rows.append(dict(task=task, cond=cond, seed=s, fold=f, system="oracle", f1=np.nan, acc=100 * float(np.any(list(right.values()), 0).mean())))
            m1 = A.margin(P["mn10"]); fix = (pr["mn10"] != yt) & (pr["wave"] == yt); wr = pr["mn10"] != yt
            rec_au = dict(task=task, cond=cond, seed=s, fold=f, auroc_fixable_by_wave=roc_auc_score(fix, 1 - m1) if 0 < fix.sum() < len(fix) else np.nan,
                          auroc_wrong=roc_auc_score(wr, 1 - m1) if 0 < wr.sum() < len(wr) else np.nan, **rec)
            for a_, b_ in combinations([e for e in core if e in pr], 2):
                x1, x2 = (~right[a_]).astype(float), (~right[b_]).astype(float)
                rec_au[f"phi_{a_}_{b_}"] = float(np.corrcoef(x1, x2)[0, 1]) if x1.std() > 0 and x2.std() > 0 else np.nan
            au.append(rec_au); preds[(cond, s, f)] = (yt, pr)
    TT = pd.DataFrame(rows); AU = pd.DataFrame(au)
    # ── per condition: hull, comparators, tests ──
    fold_n = {}; res = []
    for cond in TT.cond.unique():
        g = TT[TT.cond == cond]
        fam = cond if cond != "shift" else "clean"
        bs_name, bs_vals = best_single(fam)
        stat = [m for m in g.system.unique() if m in C]
        sp = g[g.system.isin(stat)].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index(); sp["f1"] /= 100
        ntr = []
        for f in folds:
            yt, _ = preds[(cond, seeds[0], f)]; n_te = len(yt)
            if task == "tau": n_tr = int(((D.get("mn10", "clean", seeds[0])["device"] == "a") & (D.get("mn10", "clean", seeds[0])["fold"] != f)).sum())
            else: n_tr = len(y) - n_te
            ntr.append(n_te / n_tr)
        ratio = float(np.mean(ntr))
        for sysn in ("sys_a", "sys_b"):
            if sysn not in g.system.values: continue
            xs = g[g.system == sysn].set_index(["seed", "fold"]).sort_index(); Cc = float(xs.cost.mean())
            v, lo, hi, lam = A.hull_pair(sp, Cc)
            hull = {k: 100 * (A.f1_fast(preds[(cond,) + k][0], preds[(cond,) + k][1][lo], K) if lam == 0 else exp_f1(preds[(cond,) + k][0], preds[(cond,) + k][1][lo], preds[(cond,) + k][1][hi], lam, K)) for k in xs.index}
            vals = lambda nm: g[g.system == nm].set_index(["seed", "fold"]).sort_index().f1  # noqa: E731
            r = dict(task=task, cond=cond, system=sysn, budget=str(bud["a"] if sysn == "sys_a" else bud["b"]), f1=xs.f1.mean(), cost=Cc,
                     rate_tier2=float(xs.rate_a.mean() if sysn == "sys_a" else xs.rate_b2.mean()), rate_tier3=float(xs.rate_b3.mean()) if sysn == "sys_b" else 0.0,
                     hull=f"{lo}|{hi}|{lam:.4f}", hull_f1=float(np.mean(list(hull.values()))), best_single=bs_name, best_single_f1=float(vals(bs_name).mean()) if bs_name in g.system.values else np.nan,
                     ratio_nb=ratio, n_folds=len(folds), n_seeds=len(seeds), **{f"mean_{m}": float(vals(m).mean()) for m in ("mn10", "wave", ast) if m in g.system.values})
            comps = {"hull": np.array([hull[k] for k in xs.index]), "best": vals(bs_name).to_numpy() if bs_name in g.system.values else None,
                     "two_eq": vals("two_eq_" + sysn[-1]).to_numpy() if ("two_eq_" + sysn[-1]) in g.system.values else None,
                     "two_own": vals("two_own").to_numpy() if "two_own" in g.system.values else None,
                     "rgate": vals("rgate_" + sysn[-1]).to_numpy() if ("rgate_" + sysn[-1]) in g.system.values else None,
                     "avg3": vals(avg_name(core)).to_numpy() if avg_name(core) in g.system.values else None}
            if "two_eq_" + sysn[-1] in g.system.values: r["two_eq_cost"] = float(g[g.system == "two_eq_" + sysn[-1]].cost.mean()); r["two_eq_f1"] = float(vals("two_eq_" + sysn[-1]).mean())
            if "two_own" in g.system.values: r["two_own_cost"] = float(g[g.system == "two_own"].cost.mean()); r["two_own_f1"] = float(vals("two_own").mean()); r["two_own_budget"] = bud["two_tier_own"]
            for nm, z in comps.items():
                if z is None: continue
                d = xs.f1.to_numpy() - z; fm = pd.Series(d, index=xs.index).groupby(level="fold").mean().to_numpy()
                t = A.tests(xs.f1.to_numpy(), z, list(xs.index), ratio)
                r[f"d_{nm}"] = float(fm.mean()); r[f"folds_pos_{nm}"] = int((fm > 0).sum()); r[f"p_nb_{nm}"] = t["p_nb"]; r[f"p_signflip_{nm}"] = t["p_signflip"]
                r[f"nb_lo_{nm}"], r[f"nb_hi_{nm}"] = nb_ci(fm, ratio); r[f"fold_d_{nm}"] = json.dumps(np.round(fm, 3).tolist())
            r["signflip_floor"] = 2.0 / 2 ** len(folds)
            if cond in ("clean", BED[task]):
                bsr = np.random.default_rng(BOOT_SEED); acc = np.zeros(NB_BOOT)
                for f in folds:
                    n_f = len(preds[(cond, seeds[0], f)][0]); I = bsr.integers(0, n_f, (NB_BOOT, n_f)); ds_ = np.zeros(NB_BOOT)
                    for s in seeds:
                        yt, pr = preds[(cond, s, f)]; Y = yt[I]
                        fh = (1 - lam) * f1_rows(Y, pr[lo][I], K) + (lam * f1_rows(Y, pr[hi][I], K) if lam > 0 else 0)
                        ds_ += f1_rows(Y, pr[sysn][I], K) - fh
                    acc += ds_ / len(seeds)
                acc = acc / len(folds) * 100; r["boot_lo_hull"], r["boot_hi_hull"] = float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))
            res.append(r)
    RES = pd.DataFrame(res)
    TT.to_csv(out / f"{task}_by_seed_fold.csv", index=False); AU.to_csv(out / f"{task}_gate_by_seed_fold.csv", index=False)
    RES.to_csv(out / f"{task}_by_condition.csv", index=False); SEL.to_csv(out / f"{task}_budget_selection.csv", index=False)
    eqd = pd.DataFrame([dict(task=task, system=k[0], seed=k[1][0], fold=k[1][1], two_tier_rate_iv=v[1], system_cost_iv=v[2]) for k, v in eq.items()])
    eqd.to_csv(out / f"{task}_two_tier_equal_cost.csv", index=False)
    return RES, AU, bud


def humility(task, AU):
    """H3: fold-mean escalation rates rise at every noise step; field and TAU shift above clean."""
    cols = [c for c in ("rate_a", "rate_b2", "rate_b3") if c in AU]
    fm = AU.groupby(["cond", "fold"])[cols].mean().groupby("cond").mean()
    exc, steps = [], 0
    if task == "tau":
        for c in cols:
            steps += 1
            if not fm.loc["shift", c] > fm.loc["clean", c]: exc.append(f"{c}: shift {fm.loc['shift', c]:.3f} <= clean {fm.loc['clean', c]:.3f}")
    else:
        for c in cols:
            for kind in ("bg", "pink", "white"):
                seq = ["clean"] + [f"{kind}__{s}db" for s in (20, 10, 5, 0)]
                for a_, b_ in zip(seq[:-1], seq[1:]):
                    if b_ not in fm.index or pd.isna(fm.loc[b_, c]): continue
                    steps += 1
                    if not fm.loc[b_, c] > fm.loc[a_, c]: exc.append(f"{c}: {b_} {fm.loc[b_, c]:.3f} <= {a_} {fm.loc[a_, c]:.3f}")
            steps += 1
            if not fm.loc["field", c] > fm.loc["clean", c]: exc.append(f"{c}: field <= clean")
    return {"steps": steps, "exceptions": exc, "holds": not exc, "rates": fm.round(4).to_dict()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--tasks", nargs="+", default=list(TASKS)); a = ap.parse_args()
    al = json.load(open(a.r / "inputs/alignment.json"))
    if not (al.get("all_ok") and al.get("stage") == "post"): raise SystemExit("alignment.json is not all_ok: refusing to score")
    out = a.r / "tables"; out.mkdir(exist_ok=True)
    allres, H3, buds = [], {}, {}
    for t in a.tasks:
        RES, AU, bud = analyze_task(a.r, t, out); allres.append(RES); H3[t] = humility(t, AU); buds[t] = {k: str(v) for k, v in bud.items()}
        print(t, "done", flush=True)
    RES = pd.concat(allres)
    prim = RES[[r.cond == BED[r.task] for r in RES.itertuples()]].copy()
    prim["p_max"] = prim[["p_nb_hull", "p_nb_best"]].max(1); prim["p_holm6"] = holm(prim.p_max.to_numpy()) if len(prim) else []
    prim["p_family19"] = np.minimum(1, FAMILY * prim.p_max)
    nf = {t: TASKS[t]["nf"] for t in TASKS}
    prim["h1_task_pass"] = [(r.d_hull >= 1.0) and (r.d_best >= 1.0) and (r.folds_pos_hull > nf[r.task] / 2) and (r.folds_pos_best > nf[r.task] / 2) and (r.p_holm6 < 0.05) for r in prim.itertuples()]
    prim["h2_task_pass"] = [(r.d_two_eq >= 0.5) and (r.folds_pos_two_eq > nf[r.task] / 2) for r in prim.itertuples()]
    prim.to_csv(out / "primary.csv", index=False)
    v = {"family": FAMILY, "beds": BED, "budgets_chosen": buds, "H1": {}, "H2": {}, "H3": H3}
    for sysn in ("sys_a", "sys_b"):
        g = prim[prim.system == sysn]
        v["H1"][sysn] = {"tasks_pass": g[g.h1_task_pass].task.tolist(), "pass": int(g.h1_task_pass.sum()) >= 2}
        v["H2"][sysn] = {"tasks_pass": g[g.h2_task_pass].task.tolist(), "pass": int(g.h2_task_pass.sum()) >= 2}
    v["H1_pass"] = any(x["pass"] for x in v["H1"].values())
    v["H2_pass"] = any(v["H2"][s]["pass"] for s in v["H1"] if v["H1"][s]["pass"]) if v["H1_pass"] else any(x["pass"] for x in v["H2"].values())
    v["H3_holds_all_tasks"] = all(h["holds"] for h in H3.values())
    v["stop_fires"] = bool(not (prim.d_two_eq > 0.5).any())
    v["rules"] = {"H1": "a system beats the equal-cost hull and the best single by 1.0 point on two of three tasks, majority of folds, Nadeau-Bengio, Holm over 3 tasks x 2 systems",
                  "H2": "the same system beats the equal-cost two-tier MN10-Q8 to AST-Q4 cascade by 0.5 points on two of three tasks (majority of folds)",
                  "H3": "escalation rises with shift at every noise step; field and TAU shift above clean",
                  "Stop": "neither system more than 0.5 points above the two-tier cascade on any task"}
    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=str))
    cols = ["task", "system", "budget", "f1", "cost", "rate_tier2", "rate_tier3", "hull_f1", "d_hull", "nb_lo_hull", "nb_hi_hull", "folds_pos_hull", "best_single", "d_best", "folds_pos_best",
            "d_two_eq", "folds_pos_two_eq", "p_max", "p_holm6", "h1_task_pass", "h2_task_pass"]
    print(prim[cols].round(3).to_string()); print(json.dumps({k: v[k] for k in ("H1_pass", "H2_pass", "H3_holds_all_tasks", "stop_fires")}))


if __name__ == "__main__":
    main()
