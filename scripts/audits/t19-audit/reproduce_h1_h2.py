"""Checks 1, 2, 4, 7: pick on clean inner validation, cascade, hull (several memberships), Nadeau-Bengio, every budget and rule.
Own code over the raw score files. Usage: python reproduce_h1_h2.py"""
import json
import sys
import numpy as np
import pandas as pd
from audit_core import OUT, TASK, NEW, LEG, get, sys_dirs, cost, mf1, softmax, margin, nb, holm, hull, thr_closest, thr_quantile

BUDGETS = [0.05, 0.10, 0.20, 0.30]
RULES = ["B", "avg"]
ABSENT_ZERO = "--absent-zero" in sys.argv     # check 4: the old F1 (absent classes scored as 0)
TAG = "_absentzero" if ABSENT_ZERO else ""


def F(y, p, K):
    return mf1(y, p, K, ABSENT_ZERO)


# hull memberships
def members(task, which):
    allsys = list(sys_dirs(task))
    if which == "B3":                      # ANALYSIS_PLAN Part B3 (as run by T19)
        return allsys
    if which == "B3_own":                  # B3, own heads only (no transferred w8a8 head, no stored FP32-trained head)
        return [s for s in allsys if not (s.endswith(":w8head") or s.endswith(":stored"))]
    strict = ["w8a8", "w8a6", "w8a4", "w4a8", "w4a6", "n4a4", "w4a4", "w6a6", "mp6", "mp8"]
    if which == "plan41":                  # round plan 4.1 list only; stored W4A8 = L:w4g128, stored W3A8 = L:w3g64; both head kinds
        return [s for s in allsys if s.split(":")[0] in strict and not s.startswith("L:")] + \
               [s for s in allsys if s.startswith("L:w4g128") or s.startswith("L:w3g64")] + ["mn10", "mn20"]
    if which == "plan41_own":
        return [s for s in members(task, "plan41") if not (s.endswith(":w8head") or s.endswith(":stored"))]
    if which == "B3_no_legacy":
        return [s for s in allsys if not s.startswith("L:")]
    if which == "B3_no_n4a6":
        return [s for s in allsys if not s.startswith("n4a6")]
    raise ValueError(which)


def folds(task):
    return list(range(1, TASK[task]["nf"] + 1))


CACHE = {}


def static_test(task, name, bed):
    """fold-level F1 (seeds averaged) on test rows of the bed."""
    k = (task, name, bed)
    if k not in CACHE:
        K = TASK[task]["K"]
        out = []
        for f in folds(task):
            v = []
            for s in TASK[task]["seeds"]:
                D = get(task, name, s, bed)
                if D is None:
                    CACHE[k] = None
                    return None
                r = np.flatnonzero(D["fold"] == f)
                v.append(F(D["y"][r], D["z"][r].argmax(1), K))
            out.append(np.mean(v))
        CACHE[k] = np.array(out)
    return CACHE[k]


def static_iv(task, name):
    k = (task, name, "iv")
    if k not in CACHE:
        K = TASK[task]["K"]
        v = []
        for s in TASK[task]["seeds"]:
            D = get(task, name, s, "clean")
            if D is None:
                CACHE[k] = None
                return None
            for f in folds(task):
                idx, z = D["iv"][f]
                v.append(F(D["y"][idx], z.argmax(1), K))
        CACHE[k] = float(np.mean(v))
    return CACHE[k]


def cascade(task, bed, budget, rule, thr_fn=thr_closest, iv_mode=False, tail="low", cheap="n4a4:own", exp="w8a8:own", rand=0):
    K = TASK[task]["K"]
    rows = []
    for s in TASK[task]["seeds"]:
        Cc, Ce = get(task, cheap, s, "clean"), get(task, exp, s, "clean")
        Dc, De = get(task, cheap, s, bed), get(task, exp, s, bed)
        for f in folds(task):
            ivr, ivz = Cc["iv"][f]
            assert np.array_equal(ivr, Ce["iv"][f][0])
            m_iv = margin(softmax(ivz))
            if tail == "low":
                thr = thr_fn(m_iv, budget)
            else:                                   # wrong-tail control: escalate the most confident clips
                thr = thr_fn(-m_iv, budget)
            if iv_mode:
                y, zc, ze = Cc["y"][ivr], ivz, Ce["iv"][f][1]
            else:
                r = np.flatnonzero(Dc["fold"] == f)
                assert not np.isin(r, ivr).any()
                y, zc, ze = Dc["y"][r], Dc["z"][r], De["z"][r]
            pc, pe = softmax(zc), softmax(ze)
            m = margin(pc)
            esc = (m < thr) if tail == "low" else (-m < thr)
            fused = pe if rule == "B" else (pc + pe) / 2
            pred = np.where(esc, fused.argmax(1), pc.argmax(1))
            row = dict(seed=s, fold=f, f1=F(y, pred, K), rate=esc.mean())
            if rand:
                g = np.random.default_rng(7 * s + 1000 * f + 1)
                v = []
                for _ in range(rand):
                    e2 = np.zeros(len(y), bool)
                    e2[g.choice(len(y), esc.sum(), replace=False)] = True
                    v.append(F(y, np.where(e2, fused.argmax(1), pc.argmax(1)), K))
                row["rand"] = np.mean(v)
            rows.append(row)
    return pd.DataFrame(rows)


def fm(df, col):
    return df.groupby("fold")[col].mean().sort_index().to_numpy()


def always_avg(task, bed):
    K = TASK[task]["K"]
    out = []
    for f in folds(task):
        v = []
        for s in TASK[task]["seeds"]:
            Dc, De = get(task, "n4a4:own", s, bed), get(task, "w8a8:own", s, bed)
            r = np.flatnonzero(Dc["fold"] == f)
            v.append(F(Dc["y"][r], (softmax(Dc["z"][r]) + softmax(De["z"][r])).argmax(1), K))
        out.append(np.mean(v))
    return np.array(out)


def hull_folds(task, bed, c, mem, priced=True):
    pts, arr = [], {}
    for nm in mem:
        a = static_test(task, nm, bed)
        if a is None:
            continue
        arr[nm] = a
        pts.append((nm, cost(task, nm, priced), a.mean()))
    av = always_avg(task, bed)
    arr["always-average"] = av
    pts.append(("always-average", cost(task, "n4a4:own", priced) + cost(task, "w8a8:own", priced), av.mean()))
    v, lo, hi, lam = hull(pts, c)
    h = arr[lo] if hi is None else (1 - lam) * arr[lo] + lam * arr[hi]
    return h, f"{lo}" + (f" + {hi} lam {lam:.3f}" if hi else "")


def ratio(task):
    D = get(task, "w8a8:own", 0, "clean")
    return float(np.mean([(D["fold"] == f).sum() / (D["fold"] != f).sum() for f in folds(task)]))


def pick(task, mem, thr_fn=thr_closest):
    pts = [(nm, cost(task, nm), static_iv(task, nm)) for nm in mem if static_iv(task, nm) is not None]
    cc, ce = cost(task, "n4a4:own"), cost(task, "w8a8:own")
    rows = []
    for b in BUDGETS:
        for rule in RULES:
            df = cascade(task, "clean", b, rule, thr_fn=thr_fn, iv_mode=True)
            c = cc + df.rate.mean() * ce
            hv = hull(pts, c)[0]
            rows.append(dict(budget=b, rule=rule, iv_f1=df.f1.mean(), iv_rate=df.rate.mean(), iv_cost_T=c / 1e12, iv_hull=hv,
                             adv=df.f1.mean() - hv))
    P = pd.DataFrame(rows)
    best = P.sort_values(["adv", "iv_cost_T"], ascending=[False, True]).iloc[0]
    return P, float(best.budget), best.rule


def evaluate(task, bed, b, rule, mem, thr_fn=thr_closest, rand=0, priced=True):
    cc, ce = cost(task, "n4a4:own", priced), cost(task, "w8a8:own", priced)
    df = cascade(task, bed, b, rule, thr_fn=thr_fn, rand=rand)
    fc = fm(df, "f1")
    c = cc + df.rate.mean() * ce
    h, desc = hull_folds(task, bed, c, mem, priced)
    d = fc - h
    K = len(d)
    out = dict(task=task, bed=bed, budget=b, rule=rule, f1=fc.mean(), rate=df.rate.mean(), cost_T=c / 1e12, hull=h.mean(), pair=desc,
               diff=d.mean(), **{f"nb_{k}": v for k, v in nb(d, ratio(task)).items()}, folds_ge1=int((d >= 1).sum()),
               folds_pos=int((d > 0).sum()), n_folds=K, fold_diffs=json.dumps(np.round(d, 4).tolist()),
               fold_cascade=json.dumps(np.round(fc, 4).tolist()), fold_hull=json.dumps(np.round(h, 4).tolist()))
    if rand:
        out["random_gate"] = fm(df, "rand").mean()
    return out


if __name__ == "__main__":
    res = {}
    allrows = []
    for task in ("esc50", "urban"):
        # 1. the T19 rule as run (B3 hull, thr_closest)
        for which in ("B3", "B3_own", "plan41", "plan41_own", "B3_no_legacy", "B3_no_n4a6"):
            mem = members(task, which)
            P, b, rule = pick(task, mem)
            P.to_csv(OUT / f"pick_{task}_{which}{TAG}.csv", index=False)
            for bed in ("field", "clean"):
                r = evaluate(task, bed, b, rule, mem, rand=100 if which == "B3" else 0)
                r["hull_set"] = which
                r["picked"] = True
                allrows.append(r)
                print(f"{task} {which:13s} pick b{b} {rule:3s} {bed:5s} cascade {r['f1']:.3f} rate {r['rate']:.4f} cost {r['cost_T']:.4f} "
                      f"hull {r['hull']:.3f} [{r['pair']}] diff {r['diff']:+.3f} NB [{r['nb_lo']:+.2f},{r['nb_hi']:+.2f}] p {r['nb_p']:.3f} "
                      f">=1: {r['folds_ge1']}/{r['n_folds']}" + (f" rand {r['random_gate']:.2f}" if 'random_gate' in r else ""), flush=True)
            # every budget and rule on the field mix and clean, this hull
            for bed in ("field", "clean"):
                for bb in BUDGETS:
                    for rr in RULES:
                        r = evaluate(task, bed, bb, rr, mem)
                        r["hull_set"] = which
                        r["picked"] = (bb == b and rr == rule)
                        allrows.append(r)
        # quantile threshold variant of the T19 pick
        mem = members(task, "B3")
        P, b, rule = pick(task, mem, thr_fn=thr_quantile)
        r = evaluate(task, "field", b, rule, mem, thr_fn=thr_quantile)
        r["hull_set"] = "B3_quantile_thr"
        r["picked"] = True
        allrows.append(r)
        print(f"{task} quantile-threshold pick b{b} {rule}: field diff {r['diff']:+.3f} cascade {r['f1']:.3f} rate {r['rate']:.4f}")
        # unpriced view of the B3 pick
        P, b, rule = pick(task, mem)
        r = evaluate(task, "field", b, rule, mem, priced=False)
        r["hull_set"] = "B3_unpriced_view"
        r["picked"] = True
        allrows.append(r)
        print(f"{task} unpriced view field diff {r['diff']:+.3f} hull {r['hull']:.3f} [{r['pair']}]")
    df = pd.DataFrame(allrows)
    df.to_csv(OUT / f"h1_h2_all{TAG}.csv", index=False)
    # Holm over the two tasks for the picked B3 rows
    for bed in ("field", "clean"):
        sel = df[(df.hull_set == "B3") & df.picked & (df.bed == bed)].drop_duplicates(["task"])
        ph = holm(sel.nb_p.to_numpy())
        for (_, r), p in zip(sel.iterrows(), ph):
            print(f"Holm(2 tasks) {bed} {r.task}: diff {r['diff']:+.3f} p_nb {r.nb_p:.4f} p_holm {p:.4f} family17 {min(1, 17 * p):.3f}")
