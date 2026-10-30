"""T22 analysis (ANALYSIS_PLAN.md E to G): budget pair chosen on clean inner validation, the three-tier hierarchy, the equal-cost hull,
the best single at equal or lower cost, the always-average of all tiers, the two-tier cascade (own budget and cost-matched), H1, H2,
H3, the stop rule and the verdict. Reads only score files (the coarse expert's and the stored tiers'), the cost tables and
inputs/alignment.json (refuses to run unless it says ok).     python t22_analyze.py --r R"""
import argparse, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t22_common import *  # noqa
import t13_common as T13
import analyze as A
from idea5b_common import holm, mix_expected_f1
from scipy import stats

R2, R3 = (0.30, 0.50, 0.70), (0.05, 0.10, 0.20, 0.30)
PAIRS = [(a, b) for a in R2 for b in R3]
BUD2 = (0.05, 0.10, 0.20, 0.30, 0.50)
SINGLES = ["coarse", "mn10", "mn20", "astw4", "astfp32", "astshort"]
TIERS = ("coarse", "mn10", "astw4")
AVGS = [c for n in (2, 3) for c in combinations(TIERS, n)]
FAMILY, NB_BOOT, BOOT_SEED = 19, 2000, 20261028
BED = {"esc50": "field", "urban": "field", "tau": "shift"}
ALL3 = {"esc50": ["clean"] + NOISE + ["field"], "urban": ["clean", "bg__10db", "bg__0db", "field"], "tau": ["clean", "shift"]}


def pname(p): return f"h{int(round(p[0] * 100)):02d}_{int(round(p[1] * 100)):02d}"
def aname(c): return "avg:" + "+".join(c)


class Data:
    """Probabilities of every expert, condition and seed for one task; TAU conditions are row masks of one array."""

    def __init__(self, r, task):
        self.r, self.task, self.K, self.nf = r, task, TASKS[task]["K"], TASKS[task]["nf"]
        self.B = bed(task); self.y = self.B["y"]; self.fold = self.B["fold"]; self.c = {}
        self.experts = [e for e in SINGLES if not (task == "tau" and e == "astshort")]

    def raw(self, ex, cond, s):
        key = (ex, cond, s)
        if key not in self.c:
            if ex == "coarse":
                d = self.r / "scores" / self.task / f"coarse_{self.task}_s{s}"
                if self.task == "tau":
                    z = np.load(d / "coarse_w8a8__all.npz"); iv = {int(k[7:]): (z[k], z[f"iv_logits_{k[7:]}"].astype(np.float64)) for k in z.files if k.startswith("iv_idx_")}
                    D = {"y": z["y"].astype(np.int64), "fold": z["fold"].astype(int), "z": z["test_logits"].astype(np.float64), "iv": iv}
                else:
                    D = A.ScoreDir(d).get("coarse_w8a8", cond)
            else:
                D = T13.load(self.task, ex, "clean" if self.task == "tau" else cond, s)
            if not (np.array_equal(D["y"], self.y) and np.array_equal(D["fold"], self.fold)):
                raise SystemExit(f"{self.task} {ex} {cond} s{s}: rows differ")
            self.c[key] = {"z": A.softmax(D["z"]), "iv": {f: (np.asarray(i, int), A.softmax(v)) for f, (i, v) in D["iv"].items()}}
        return self.c[key]

    def has(self, ex, cond):
        if ex == "coarse":
            return True
        return T13.available(self.task, ex, "clean" if self.task == "tau" else cond, SEEDS)

    def test_rows(self, cond, f):
        m = self.fold == f
        if self.task == "tau":
            dev = self.B["device"]
            m &= (dev == "a") if cond == "clean" else ((dev != "a") if cond == "shift" else (dev == cond))
        return np.flatnonzero(m)

    def test(self, ex, cond, s, f):
        rc = "clean" if self.task == "tau" else cond
        return self.raw(ex, rc, s)["z"][self.test_rows(cond, f)]

    def iv(self, ex, cond, s, f):
        rc = "clean" if self.task == "tau" else cond
        return self.raw(ex, rc, s)["iv"][f]


def costs(r, task):
    c = T13.cost_table(task)
    c["coarse"] = json.load(open(r / "inputs/p2_costs.json"))["tasks"][task]["mn10"]["total_bitops"]
    return c


def hier(p1, p2, p3, t1, t2, own=False):
    """Coarse to fine: tier 2 if margin(p1) < t1; tier 3 if margin(p12) < t2 (own=True: margin(p2)); output averages the tiers that ran."""
    e2 = A.margin(p1) < t1
    p12 = (p1 + p2) / 2
    e3 = e2 & ((A.margin(p2) if own else A.margin(p12)) < t2)
    out = np.where(e3[:, None], (p1 + p2 + p3) / 3, np.where(e2[:, None], p12, p1))
    return out.argmax(1), e2, e3


def thresholds(p1, p2, r2, r3, own=False):
    m1 = A.margin(p1); t1 = A.thr_rate(m1, r2); e2 = m1 < t1
    m = A.margin(p2) if own else A.margin((p1 + p2) / 2)
    t2 = A.thr_rate(m[e2], r3 / r2) if e2.any() else -np.inf
    return t1, t2


def mc_hull_iv(y, plo, phi, lam, K, rng):
    if lam == 0.0:
        return A.f1_fast(y, plo, K)
    dr = rng.random((A.DRAWS, len(plo))) < lam
    return float(A.f1_batch(y, np.where(dr, phi[None, :], plo[None, :]), K).mean())


def nb_ci(fm, ratio):
    fm = np.asarray(fm, float); k = len(fm); se = np.sqrt((1 / k + ratio) * fm.var(ddof=1)); h = stats.t.ppf(0.975, k - 1) * se
    return float(fm.mean() - h), float(fm.mean() + h)


def nb_ratio(D, cond):
    """Mean over folds of n_test / n_train (TAU: rows of the bed in the fold over device A training rows)."""
    out = []
    for f in range(1, D.nf + 1):
        nt = len(D.test_rows(cond, f))
        ntr = int(((D.fold != f) & D.B["is_a"]).sum())
        out.append(nt / ntr)
    return float(np.mean(out))


def analyze_task(r, task, out):
    D = Data(r, task); K = D.K; C = costs(r, task); keys = [(s, f) for s in SEEDS for f in range(1, D.nf + 1)]
    statics = [e for e in D.experts] + [aname(c) for c in AVGS]
    cost_of = {e: C[e] for e in D.experts} | {aname(c): sum(C[e] for e in c) for c in AVGS}

    def probs(kind, which, cond, s, f):
        """which: an expert or an avg:... name; kind 'test' or 'iv'. Returns (idx or None, probs)."""
        if which.startswith("avg:"):
            parts = [probs(kind, e, cond, s, f) for e in which[4:].split("+")]
            return parts[0][0], sum(p for _, p in parts) / len(parts)
        if kind == "test":
            return None, D.test(which, cond, s, f)
        return D.iv(which, cond, s, f)

    # ── clean inner validation: thresholds, pair choice, two-tier budget ──
    rng = np.random.default_rng(A.RNG_IV)
    ivp, thr, thr_own, thr2 = {}, {}, {}, {}
    for s, f in keys:
        idx, p1 = probs("iv", "coarse", "clean", s, f); i2, p2 = probs("iv", "mn10", "clean", s, f); i3, p3 = probs("iv", "astw4", "clean", s, f)
        if not (np.array_equal(idx, i2) and np.array_equal(idx, i3)): raise SystemExit(f"{task} s{s} f{f}: inner splits differ")
        ivp[(s, f)] = (idx, p1, p2, p3)
        for pr in PAIRS:
            thr[(pr, s, f)] = thresholds(p1, p2, *pr); thr_own[(pr, s, f)] = thresholds(p1, p2, *pr, own=True)
        for b in BUD2:
            thr2[(b, s, f)] = A.thr_rate(A.margin(p2), b)
    st_iv = {}
    for s, f in keys:
        idx = ivp[(s, f)][0]; y = D.y[idx]
        for e in statics:
            _, p = probs("iv", e, "clean", s, f); st_iv[(e, s, f)] = (A.f1_fast(y, p.argmax(1), K), p.argmax(1))
    sp_iv = pd.DataFrame([{"system": e, "f1": np.mean([st_iv[(e, s, f)][0] for s, f in keys]), "cost": cost_of[e]} for e in statics])
    sel, ivcost = [], {}
    for pr in PAIRS:
        f1s, cs = [], []
        for s, f in keys:
            idx, p1, p2, p3 = ivp[(s, f)]
            pred, e2, e3 = hier(p1, p2, p3, *thr[(pr, s, f)])
            f1s.append(A.f1_fast(D.y[idx], pred, K)); c_ = float(np.mean(C["coarse"] + e2 * C["mn10"] + e3 * C["astw4"])); cs.append(c_); ivcost[(pr, s, f)] = c_
        civ = float(np.mean(cs)); v, lo, hi, lam = A.hull_pair(sp_iv, civ)
        hl = [mc_hull_iv(D.y[ivp[k][0]], st_iv[(lo, *k)][1], st_iv[(hi, *k)][1], lam, K, rng) for k in keys]
        sel.append({"task": task, "pair": pname(pr), "r2": pr[0], "r3": pr[1], "f1_iv": np.mean(f1s) * 100, "cost_iv": civ, "hull_iv": np.mean(hl) * 100,
                    "gain_iv": (np.mean(f1s) - np.mean(hl)) * 100, "hull": f"{lo}|{hi}|{lam:.4f}"})
    SEL = pd.DataFrame(sel).sort_values(["gain_iv", "cost_iv"], ascending=[False, True]); pstar = tuple(SEL.iloc[0][["r2", "r3"]]); SEL.to_csv(out / f"{task}_pair_selection.csv", index=False)
    sp2 = sp_iv[sp_iv.system.isin(["mn10", "astw4", aname(("mn10", "astw4"))])]
    sel2 = []
    for b in BUD2:
        f1s, cs = [], []
        for s, f in keys:
            idx, p1, p2, p3 = ivp[(s, f)]; e = A.margin(p2) < thr2[(b, s, f)]
            f1s.append(A.f1_fast(D.y[idx], np.where(e, ((p2 + p3) / 2).argmax(1), p2.argmax(1)), K)); cs.append(float(np.mean(C["mn10"] + e * C["astw4"])))
        civ = float(np.mean(cs)); v, lo, hi, lam = A.hull_pair(sp2, civ)
        hl = [mc_hull_iv(D.y[ivp[k][0]], st_iv[(lo, *k)][1], st_iv[(hi, *k)][1], lam, K, rng) for k in keys]
        sel2.append({"task": task, "budget": b, "f1_iv": np.mean(f1s) * 100, "cost_iv": civ, "gain_iv": (np.mean(f1s) - np.mean(hl)) * 100})
    SEL2 = pd.DataFrame(sel2).sort_values(["gain_iv", "cost_iv"], ascending=[False, True]); bstar = float(SEL2.iloc[0].budget); SEL2.to_csv(out / f"{task}_two_tier_budget.csv", index=False)
    # cost-matched two-tier thresholds (per pair, per seed and fold, on clean inner validation)
    thr_cm = {}
    for pr in PAIRS:
        for s, f in keys:
            rate = float(np.clip((ivcost[(pr, s, f)] - C["mn10"]) / C["astw4"], 0.0, 1.0)); thr_cm[(pr, s, f)] = A.thr_rate(A.margin(ivp[(s, f)][2]), rate)
    # best single at equal or lower cost: inner validation of the bed's family (field for ESC-50 and UrbanSound8K; TAU clean device A)
    fam = "field" if task != "tau" else "clean"
    iv_single = {}
    for e in D.experts:
        if not D.has(e, fam): continue
        v = []
        for s, f in keys:
            idx, p = probs("iv", e, fam, s, f); v.append(A.f1_fast(D.y[idx], p.argmax(1), K))
        iv_single[e] = float(np.mean(v))
    # ── test ──
    conds = [c for c in ALL3[task] if all(D.has(e, c) for e in TIERS)]
    rows, preds, rates = [], {}, []
    for c in conds:
        for s, f in keys:
            te = D.test_rows(c, f); y = D.y[te]
            p1, p2, p3 = (probs("test", e, c, s, f)[1] for e in TIERS)
            pr_ = {}
            for e in statics:
                if e in D.experts and not D.has(e, c): continue
                pr_[e] = probs("test", e, c, s, f)[1].argmax(1)
                rows.append({"cond": c, "seed": s, "fold": f, "system": e, "f1": A.f1_fast(y, pr_[e], K) * 100, "acc": float((pr_[e] == y).mean()) * 100, "cost": cost_of[e]})
            for pr in PAIRS:
                for own in (False, True):
                    if own and pr != pstar: continue
                    pred, e2, e3 = hier(p1, p2, p3, *(thr_own if own else thr)[(pr, s, f)], own=own)
                    nm = pname(pr) + ("_own" if own else "")
                    pr_[nm] = pred
                    rows.append({"cond": c, "seed": s, "fold": f, "system": nm, "f1": A.f1_fast(y, pred, K) * 100, "acc": float((pred == y).mean()) * 100,
                                 "cost": float(np.mean(C["coarse"] + e2 * C["mn10"] + e3 * C["astw4"])), "rate2": float(e2.mean()), "rate3": float(e3.mean())})
                e = A.margin(p2) < thr_cm[(pr, s, f)]; nm = "two_cm_" + pname(pr); pr_[nm] = np.where(e, ((p2 + p3) / 2).argmax(1), p2.argmax(1))
                rows.append({"cond": c, "seed": s, "fold": f, "system": nm, "f1": A.f1_fast(y, pr_[nm], K) * 100, "acc": float((pr_[nm] == y).mean()) * 100,
                             "cost": float(np.mean(C["mn10"] + e * C["astw4"])), "rate3": float(e.mean())})
            for b in BUD2:
                e = A.margin(p2) < thr2[(b, s, f)]; nm = f"two_b{int(b * 100):02d}"; pr_[nm] = np.where(e, ((p2 + p3) / 2).argmax(1), p2.argmax(1))
                rows.append({"cond": c, "seed": s, "fold": f, "system": nm, "f1": A.f1_fast(y, pr_[nm], K) * 100, "acc": float((pr_[nm] == y).mean()) * 100,
                             "cost": float(np.mean(C["mn10"] + e * C["astw4"])), "rate3": float(e.mean())})
            oracle = (p1.argmax(1) == y) | (p2.argmax(1) == y) | (p3.argmax(1) == y)
            w1, w2, w3 = (p.argmax(1) != y for p in (p1, p2, p3))
            phi = lambda u, v: float(np.corrcoef(u, v)[0, 1]) if u.std() > 0 and v.std() > 0 else np.nan  # noqa: E731
            rates.append({"cond": c, "seed": s, "fold": f, "oracle_acc": float(oracle.mean()) * 100, "phi_t1_t2": phi(w1, w2), "phi_t1_t3": phi(w1, w3), "phi_t2_t3": phi(w2, w3)})
            preds[(c, s, f)] = (y, pr_)
    T_ = pd.DataFrame(rows); T_.to_csv(out / f"{task}_by_seed_fold.csv", index=False); pd.DataFrame(rates).to_csv(out / f"{task}_oracle_phi.csv", index=False)
    PS = pname(pstar)
    res = []
    for c in conds:
        g = T_[T_.cond == c]; ratio = nb_ratio(D, c)
        sp = g[g.system.isin(statics)].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index(); sp["f1"] /= 100
        vals = lambda nm: g[g.system == nm].set_index(["seed", "fold"]).sort_index()  # noqa: E731
        for hn in [pname(p) for p in PAIRS] + [PS + "_own"]:
            xh = vals(hn); Ch = float(xh.cost.mean()); v, lo, hi, lam = A.hull_pair(sp, Ch)
            hull = np.array([(A.f1_fast(preds[(c, s, f)][0], preds[(c, s, f)][1][lo], K) if lam == 0 else
                              float(mix_expected_f1(preds[(c, s, f)][0], preds[(c, s, f)][1][lo], preds[(c, s, f)][1][hi], lam, K)[0])) * 100 for s, f in xh.index])
            cands = [e for e in iv_single if e in sp.system.values and cost_of[e] <= Ch * (1 + 1e-12)]
            best = max(cands, key=lambda e: iv_single[e]) if cands else None
            two_own = f"two_b{int(bstar * 100):02d}"; two_cm = "two_cm_" + hn.replace("_own", "")
            r = {"task": task, "cond": c, "system": hn, "chosen": hn == PS, "f1": xh.f1.mean(), "acc": xh.acc.mean(), "cost": Ch, "rate2": xh.rate2.mean(), "rate3": xh.rate3.mean(),
                 "hull": f"{lo}|{hi}|{lam:.4f}", "hull_f1": hull.mean(), "best_single": best, "best_single_f1": vals(best).f1.mean() if best else np.nan,
                 "avg3_f1": vals(aname(TIERS)).f1.mean(), "two_own_budget": bstar, "two_own_f1": vals(two_own).f1.mean(), "two_own_cost": vals(two_own).cost.mean(),
                 "two_cm_f1": vals(two_cm).f1.mean(), "two_cm_cost": vals(two_cm).cost.mean(), "n_folds": D.nf, "nb_ratio": ratio}
            for k_, z in (("hull", hull), ("best", vals(best).f1.to_numpy() if best else None), ("avg3", vals(aname(TIERS)).f1.to_numpy()),
                          ("two_own", vals(two_own).f1.to_numpy()), ("two_cm", vals(two_cm).f1.to_numpy())):
                if z is None: continue
                d = xh.f1.to_numpy() - z; fm = pd.Series(d, index=xh.index).groupby(level="fold").mean().to_numpy()
                t = A.tests(xh.f1.to_numpy(), z, list(xh.index), ratio); lo_, hi_ = nb_ci(fm, ratio)
                r |= {f"d_{k_}": float(fm.mean()), f"p_nb_{k_}": t["p_nb"], f"p_signflip_{k_}": t["p_signflip"], f"nb_lo_{k_}": lo_, f"nb_hi_{k_}": hi_,
                      f"folds_ge1_{k_}": int((fm >= 1.0).sum()), f"folds_ge05_{k_}": int((fm >= 0.5).sum()), f"folds_pos_{k_}": int((fm > 0).sum()), f"fold_d_{k_}": json.dumps(np.round(fm, 3).tolist())}
            if hn == PS and c in (BED[task], "clean"):  # clip-level bootstrap of hierarchy minus hull
                bs = np.random.default_rng(BOOT_SEED); acc = np.zeros(NB_BOOT)
                for f in range(1, D.nf + 1):
                    n_f = len(preds[(c, SEEDS[0], f)][0]); I = bs.integers(0, n_f, (NB_BOOT, n_f)); ds = np.zeros(NB_BOOT)
                    for s in SEEDS:
                        y, pr_ = preds[(c, s, f)]; Y = y[I]
                        fh = (1 - lam) * A.f1_batch(Y, pr_[lo][I], K) + (lam * A.f1_batch(Y, pr_[hi][I], K) if lam > 0 else 0)
                        ds += A.f1_batch(Y, pr_[hn][I], K) - fh
                    acc += ds / len(SEEDS)
                acc = acc / D.nf * 100; r["boot_lo_hull"], r["boot_hi_hull"] = float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))
            res.append(r)
    RES = pd.DataFrame(res); RES.to_csv(out / f"{task}_by_condition.csv", index=False)
    # ── H3: escalation rates need tiers 1 and 2 only, so every noise cell (and the field mix) where both exist ──
    h3c = (["clean"] + NOISE + ["field"]) if task != "tau" else ["clean"] + sorted(set(D.B["device"]) - {"a"})
    hr = []
    for c in h3c:
        for s, f in keys:
            p1 = D.test("coarse", c, s, f); p2 = D.test("mn10", c, s, f)
            _, e2, e3 = hier(p1, p2, p2, *thr[(pstar, s, f)])
            hr.append({"cond": c, "seed": s, "fold": f, "rate2": float(e2.mean()), "rate3": float(e3.mean())})
    HR = pd.DataFrame(hr).groupby(["cond", "fold"])[["rate2", "rate3"]].mean()
    steps = []
    if task != "tau":
        for kind in ("bg", "pink", "white"):
            seq = ["clean"] + [f"{kind}__{s}db" for s in SNRS]
            steps += list(zip(seq[:-1], seq[1:]))
        steps.append(("clean", "field"))
    else:
        steps = [("clean", d) for d in h3c[1:]]
    h3, exc = [], []
    for a_, b_ in steps:
        for rk in ("rate2", "rate3"):
            fa, fb = HR.loc[a_][rk], HR.loc[b_][rk]; up = bool(fb.mean() > fa.mean())
            h3.append({"task": task, "step": f"{a_} -> {b_}", "rate": rk, "from": fa.mean() * 100, "to": fb.mean() * 100, "rises": up, "folds_rising": int((fb.values > fa.values).sum()), "n_folds": len(fa)})
            if not up: exc.append(f"{rk} {a_} -> {b_}")
    pd.DataFrame(h3).to_csv(out / f"{task}_h3.csv", index=False)
    # corruptions and the other noise cells: tiers 1 and 2 only, descriptive
    if task != "tau":
        dsc = []
        for c in NOISE + CORR:
            if not (D.has("mn10", c)): continue
            for s, f in keys:
                te = D.test_rows(c, f); y = D.y[te]; p1 = D.test("coarse", c, s, f); p2 = D.test("mn10", c, s, f)
                dsc.append({"cond": c, "seed": s, "fold": f, "coarse_f1": A.f1_fast(y, p1.argmax(1), K) * 100, "mn10_f1": A.f1_fast(y, p2.argmax(1), K) * 100,
                            "avg12_f1": A.f1_fast(y, ((p1 + p2) / 2).argmax(1), K) * 100})
        pd.DataFrame(dsc).groupby("cond").mean(numeric_only=True).drop(columns=["seed", "fold"]).to_csv(out / f"{task}_tiers12_all_cells.csv")
    return RES, SEL, SEL2, pstar, bstar, exc, iv_single


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--tasks", nargs="+", default=["esc50", "urban", "tau"]); a = ap.parse_args()
    al = json.loads((a.r / "inputs/alignment.json").read_text())
    if not al.get("ok") or any(t not in al["tasks"] for t in a.tasks):
        raise SystemExit("alignment check missing or failed: refusing to score")
    out = a.r / "tables"; out.mkdir(exist_ok=True)
    V = {"tasks": {}}
    for task in a.tasks:
        RES, SEL, SEL2, pstar, bstar, exc, ivs = analyze_task(a.r, task, out)
        b = RES[(RES.cond == BED[task]) & RES.chosen].iloc[0]; cl = RES[(RES.cond == "clean") & RES.chosen].iloc[0]
        maj = TASKS[task]["nf"] // 2 + 1
        h1_parts = all(b[f"d_{k}"] >= 1.0 and b[f"folds_ge1_{k}"] >= maj for k in ("hull", "best", "avg3"))
        cost_ok = bool(b.cost <= 1.05 * b.two_own_cost); k2 = "two_own" if cost_ok else "two_cm"
        g = RES[RES.cond == BED[task]]; pairs_only = g[~g.system.str.endswith("_own")]
        stop_margin = float((pairs_only.f1 - pairs_only.two_cm_f1).max())
        V["tasks"][task] = {"pair": b.system, "two_tier_budget": bstar, "bed": BED[task], "iv_best_single_scores": ivs,
            "H1": {"f1": b.f1, "cost": b.cost, "hull": b.hull, "hull_f1": b.hull_f1, "best_single": b.best_single, "best_single_f1": b.best_single_f1, "avg3_f1": b.avg3_f1,
                   **{f"{k}_{m}": b[f"{m}_{k}"] for k in ("hull", "best", "avg3") for m in ("d", "p_nb", "nb_lo", "nb_hi", "folds_ge1", "fold_d", "p_signflip")},
                   "boot_ci_hull": [b.get("boot_lo_hull"), b.get("boot_hi_hull")], "p_max": float(max(b.p_nb_hull, b.p_nb_best, b.p_nb_avg3)), "parts_pass": bool(h1_parts), "majority": maj},
            "H2": {"comparator": k2, "cost_condition_hierarchy_le_1p05_two_tier": cost_ok, "hierarchy_cost": b.cost, "two_own_cost": b.two_own_cost, "two_cm_cost": b.two_cm_cost,
                   "d_two_own": b.d_two_own, "p_two_own": b.p_nb_two_own, "d_two_cm": b.d_two_cm, "p_two_cm": b.p_nb_two_cm, "d": b[f"d_{k2}"], "p_nb": b[f"p_nb_{k2}"],
                   "nb_ci": [b[f"nb_lo_{k2}"], b[f"nb_hi_{k2}"]], "folds_ge05": int(b[f"folds_ge05_{k2}"]), "fold_d": b[f"fold_d_{k2}"]},
            "H3": {"exceptions": exc, "holds": not exc},
            "stop_margin_best_pair_minus_cost_matched_two_tier": stop_margin,
            "clean_secondary": {"f1": cl.f1, "d_hull": cl.d_hull, "p_hull": cl.p_nb_hull, "d_best": cl.d_best, "d_avg3": cl.d_avg3, "d_two_own": cl.d_two_own}}
    T = V["tasks"]; ts = list(T)
    ph1 = holm(np.array([T[t]["H1"]["p_max"] for t in ts])); ph2 = holm(np.array([T[t]["H2"]["p_nb"] for t in ts]))
    for t, p1_, p2_ in zip(ts, ph1, ph2):
        maj = TASKS[t]["nf"] // 2 + 1
        T[t]["H1"]["p_holm"] = float(p1_); T[t]["H1"]["p_family19"] = min(1.0, FAMILY * T[t]["H1"]["p_max"]); T[t]["H1"]["passes"] = bool(T[t]["H1"]["parts_pass"] and p1_ < 0.05)
        T[t]["H2"]["p_holm"] = float(p2_); T[t]["H2"]["passes"] = bool(T[t]["H2"]["d"] >= 0.5 and T[t]["H2"]["folds_ge05"] >= maj and p2_ < 0.05)
    V["H1"] = {"rule": "hierarchy beats hull, best single (<= cost) and always-average by 1.0 on the shifted bed, majority of folds, max NB p Holm over tasks < 0.05; two of three tasks",
               "tasks_pass": [t for t in ts if T[t]["H1"]["passes"]], "passes": sum(T[t]["H1"]["passes"] for t in ts) >= 2}
    V["H2"] = {"rule": "hierarchy minus two-tier cascade >= 0.5 (own budget if the hierarchy costs at most 1.05x, else cost-matched), majority of folds, NB Holm < 0.05; two of three",
               "tasks_pass": [t for t in ts if T[t]["H2"]["passes"]], "passes": sum(T[t]["H2"]["passes"] for t in ts) >= 2}
    V["H3"] = {"rule": "both escalation rates rise at every noise step (fold mean), field above clean; TAU every device above A; all three tasks",
               "tasks_hold": [t for t in ts if T[t]["H3"]["holds"]], "passes": all(T[t]["H3"]["holds"] for t in ts)}
    V["Stop"] = {"rule": "no budget pair more than 0.5 above the cost-matched two-tier cascade on the H1 bed on any task (P1, P2 passed)",
                 "max_margin": max(T[t]["stop_margin_best_pair_minus_cost_matched_two_tier"] for t in ts),
                 "fires": all(T[t]["stop_margin_best_pair_minus_cost_matched_two_tier"] <= 0.5 for t in ts)}
    (out / "verdict.json").write_text(json.dumps(V, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(V, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:6000])


if __name__ == "__main__":
    main()
