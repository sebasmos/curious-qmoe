"""T15 and T18 pre-selection on INNER-VALIDATION clips only (ANALYSIS_PLAN_T15.md section 6, ANALYSIS_PLAN_T18.md section 3). Reads no test logit and no test label.
T15: per task, among all uniform mixtures of 2 to 4 experts of the pool, the one with the largest inner-validation accuracy minus the hull's at its priced cost; the best single
at equal or lower cost. T18: per task, the (q1, q2) setting of the hierarchy with the same criterion.
Needs <R>/precond/survivors.json: {"views": {"urban": [...], "esc50": [...], "tau": [...]}, "lr": "lra"|"lrb"|null}.
   python t15_preselect.py --out <R>/inputs"""
import argparse, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t15_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=R / "inputs"); ap.add_argument("--tasks", nargs="+", default=list(TASKS)); a = ap.parse_args()
SV = json.load(open(R / "precond/survivors.json")); QS = [0.1, 0.3, 0.5]
def ivdata(task, experts, cells):
    """{(seed, fold): (y_iv, {cell: {expert: probs}})} for the inner-validation clips."""
    out = {}
    for seed in TASKS[task]["seeds"]:
        for fold in range(1, TASKS[task]["nf"] + 1):
            per = {}
            for c in cells:
                per[c] = {}
                for e in experts:
                    D = load(task, e, c if task != "tau" else "clean", seed); idx, lg = D["iv"][fold]; per[c][e] = softmax(lg)
                    y = D["y"][idx]
            out[(seed, fold)] = (y, per)
    return out
def mean_acc(fn, data, cells):
    """mean over folds of (mean over seeds of (mean over cells of fn(y, probs_by_expert)))"""
    nf = max(f for _, f in data); accs = []
    for f in range(1, nf + 1):
        accs.append(np.mean([np.mean([fn(y, per[c]) for c in cells]) for (s, ff), (y, per) in data.items() if ff == f]))
    return float(np.mean(accs))
def margin(p): s = np.sort(p, 1); return s[:, -1] - s[:, -2]
res15, res18, tables15, tables18 = {}, {}, [], []
for task in a.tasks:
    cells = ["clean"] if task == "tau" else POOL[task]
    new = SV["views"][task]; pool = [E1, E2] + new
    univ = [e for e in ["mn10", "mn20", "astfp32", "astw4", "astshort"] if not (task == "tau" and e == "astshort")] + new
    lr = SV.get("lr")
    need = sorted(set(univ + ([lr] if lr else [])))
    ok = [e for e in need if all(available(task, e, c if task != "tau" else "clean") for c in cells)]
    univ = [e for e in univ if e in ok]; CT = costs(task)
    data = ivdata(task, ok, cells)
    sing = {e: 100 * mean_acc(lambda y, P, e=e: (P[e].argmax(1) == y).mean(), data, cells) for e in ok}
    # ---------- T15 ----------
    def hull(c, acc_by_sys):
        sp = pd.DataFrame([dict(system=e, f1=acc_by_sys[e], cost=CT[e]) for e in acc_by_sys if CT[e] > 0]); return hull_value(sp, c)[0]
    rows = []
    for c in subsets([e for e in pool if e in ok]):
        ac = 100 * mean_acc(lambda y, P, c=c: (np.mean([P[e] for e in c], 0).argmax(1) == y).mean(), data, cells); cost = mean_cost(task, c)
        h = hull(cost, {e: sing[e] for e in univ}); rows.append(dict(task=task, combo=cname(c), size=len(c), cost=cost, iv_acc=ac, iv_hull=h, gain=ac - h, has_new=any(is_new(e) for e in c)))
    T = pd.DataFrame(rows).sort_values(["gain", "cost"], ascending=[False, True]); tables15.append(T)
    best = T.iloc[0]; c = best.combo.split("+"); cost = best.cost
    cand = [e for e in univ if CT[e] <= cost * (1 + 1e-12)]; bs = max(cand, key=lambda e: sing[e])
    pair = T[T.combo == cname((E1, E2))].iloc[0]
    res15[task] = dict(mixture=best.combo, cost=float(cost), iv_acc=float(best.iv_acc), iv_gain=float(best.gain), best_single=bs, best_single_iv_acc=float(sing[bs]), pool=pool, singles_universe=univ,
                       iv_singles={e: float(sing[e]) for e in ok}, baseline_pair_iv_acc=float(pair.iv_acc), baseline_pair_iv_gain=float(pair.gain), pool_cells=cells,
                       selection_basis="clean device-A inner validation (no shifted inner-validation rows exist)" if task == "tau" else "shifted-pool inner validation")
    # ---------- T18 ----------
    if lr:
        tiers = [lr, E1, E2]
        if all(t in ok for t in tiers):
            c1, c2, c3 = [CT[t] for t in tiers]; clean_cells = ["clean"]
            cleaniv = ivdata(task, tiers, clean_cells) if cells != ["clean"] else data
            rows = []
            for q1 in QS:
                for q2 in QS:
                    accs, costs_ = [], []
                    for f in range(1, TASKS[task]["nf"] + 1):
                        a_f, c_f = [], []
                        for (s, ff), (y, per) in data.items():
                            if ff != f: continue
                            yc, perc = cleaniv[(s, ff)]; p1 = perc["clean"][tiers[0]]; p12 = (p1 + perc["clean"][tiers[1]]) / 2
                            m1 = margin(p1); t1 = np.quantile(m1, q1); esc = m1 < t1
                            t2 = np.quantile(margin(p12)[esc], q2) if esc.sum() > 0 else -1.0
                            ac_c, co_c = [], []
                            for cl in cells:
                                pp = [per[cl][t] for t in tiers]; q12 = (pp[0] + pp[1]) / 2; q123 = (pp[0] + pp[1] + pp[2]) / 3
                                e1 = margin(pp[0]) < t1; e2 = e1 & (margin(q12) < t2)
                                out = np.where(e2[:, None], q123, np.where(e1[:, None], q12, pp[0]))
                                ac_c.append((out.argmax(1) == y).mean()); co_c.append(c1 + e1.mean() * c2 + e2.mean() * c3)
                            a_f.append(np.mean(ac_c)); c_f.append(np.mean(co_c))
                        accs.append(np.mean(a_f)); costs_.append(np.mean(c_f))
                    ac, co = 100 * float(np.mean(accs)), float(np.mean(costs_))
                    h = hull(co, {e: sing[e] for e in univ + [lr] if e in sing}); rows.append(dict(task=task, q1=q1, q2=q2, cost=co, iv_acc=ac, iv_hull=h, gain=ac - h))
            T8 = pd.DataFrame(rows).sort_values(["gain", "cost"], ascending=[False, True]); tables18.append(T8); b = T8.iloc[0]
            res18[task] = dict(q1=float(b.q1), q2=float(b.q2), cost=float(b.cost), iv_acc=float(b.iv_acc), iv_gain=float(b.gain), tiers=tiers, tier_costs=[c1, c2, c3], pool_cells=cells,
                               selection_basis="clean device-A inner validation" if task == "tau" else "shifted-pool inner validation")
    print(task, json.dumps(res15[task], default=float)[:600], flush=True)
a.out.mkdir(exist_ok=True, parents=True)
def merge(name, new):
    f = a.out / name; old = json.load(open(f)) if f.exists() else {}; old.update(new); json.dump(old, open(f, "w"), indent=1, default=float)
merge("t15_preselected.json", res15); merge("t18_preselected.json", res18)
pd.concat(tables15).to_csv(a.out / f"t15_preselect_table_{'_'.join(a.tasks)}.csv", index=False)

