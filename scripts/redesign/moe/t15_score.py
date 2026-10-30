"""T15 and T18 scoring (ANALYSIS_PLAN_T15.md sections 6 to 8, ANALYSIS_PLAN_T18.md sections 3 and 4). READS TEST LABELS. Run after t15_preselect.py was committed.
Per (task, cell, seed, fold): accuracy, macro-F1 (percent) of every single expert, every uniform mixture of 2 to 4 experts of the task's pool (with the any-correct oracle),
the T18 hierarchy for all nine (q1, q2) settings (running-average margin, primary, and the own-tier margin variant), the always-average of the three tiers, the margin router on
the selected T15 mixture; the pairwise error correlation (phi) of the pool's experts.   python t15_score.py --out DIR [--tasks urban ...]"""
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t15_common import *
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--tasks", nargs="+", default=list(TASKS)); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
SV = json.load(open(R / "precond/survivors.json")); P15 = json.load(open(R / "inputs/t15_preselected.json")); P18 = json.load(open(R / "inputs/t18_preselected.json")); QS = [0.1, 0.3, 0.5]
def margin(p): s = np.sort(p, 1); return s[:, -1] - s[:, -2]
RW, phis = [], []
for task in a.tasks:
    T = TASKS[task]; K = T["K"]; CT = costs(task); lr = SV.get("lr"); new = SV["views"][task]; pool = [E1, E2] + new
    univ = [e for e in ["mn10", "mn20", "astfp32", "astw4", "astshort"] if not (task == "tau" and e == "astshort")] + new + ([lr] if lr else [])
    sel = P15[task]["mixture"].split("+"); o18 = P18.get(task)
    t0 = time.time()
    for cell in CELLS[task]:
        cl = "clean" if task == "tau" else cell
        av = [e for e in univ if available(task, e, cl)]
        if not av: continue
        for seed in T["seeds"]:
            D = {e: load(task, e, cl, seed) for e in av}; D0 = D[av[0]]; y, fold = D0["y"], D0["fold"]
            for e in av[1:]: assert np.array_equal(D[e]["y"], y) and np.array_equal(D[e]["fold"], fold), (task, e)
            P = {e: softmax(D[e]["z"]) for e in av}; pred = {e: P[e].argmax(1) for e in av}; right = {e: pred[e] == y for e in av}
            poolav = [e for e in pool if e in av]; cands = subsets(poolav)
            for f in range(1, T["nf"] + 1):
                te = rows(task, D0, f, cell); yt = y[te]
                for e in av: RW.append(dict(task=task, cell=cell, seed=seed, fold=f, kind="single", system=e, n=len(te), acc=acc(yt, pred[e][te]), f1=f1(yt, pred[e][te], K), cost=CT[e]))
                for c in cands:
                    pm = np.mean([P[e][te] for e in c], 0).argmax(1); orc = np.any([right[e][te] for e in c], 0).mean() * 100
                    RW.append(dict(task=task, cell=cell, seed=seed, fold=f, kind="mix", system=cname(c), n=len(te), acc=acc(yt, pm), f1=f1(yt, pm, K), cost=mean_cost(task, c), oracle=orc))
                for e, e2 in __import__("itertools").combinations(poolav, 2):
                    x, z = (~right[e][te]).astype(float), (~right[e2][te]).astype(float)
                    ph = float(np.corrcoef(x, z)[0, 1]) if x.std() > 0 and z.std() > 0 else np.nan
                    phis.append(dict(task=task, cell=cell, seed=seed, fold=f, a=e, b=e2, phi=ph))
                # ---- T18 ----
                if lr and o18 and all(t in av for t in (lr, E1, E2)):
                    tiers = [lr, E1, E2]; c1, c2, c3 = [CT[t] for t in tiers]
                    iv = {t: softmax(load(task, t, "clean" if task == "tau" else "clean", seed)["iv"][f][1]) for t in tiers}
                    p1c = iv[lr]; p12c = (iv[lr] + iv[E1]) / 2; m1c = margin(p1c); m2c = margin(iv[E1])
                    p1, p2, p3 = [P[t][te] for t in tiers]; q12 = (p1 + p2) / 2; q123 = (p1 + p2 + p3) / 3
                    for q1 in QS:
                        t1 = np.quantile(m1c, q1); esc_c = m1c < t1
                        for q2 in QS:
                            for var, mg, mgc in (("hier", margin(q12), margin(p12c)), ("hier_own", margin(p2), m2c)):
                                t2 = np.quantile(mgc[esc_c], q2) if esc_c.sum() else -1.0
                                e1 = margin(p1) < t1; e2_ = e1 & (mg < t2); out = np.where(e2_[:, None], q123, np.where(e1[:, None], q12, p1)).argmax(1)
                                RW.append(dict(task=task, cell=cell, seed=seed, fold=f, kind=var, system=f"q{q1}_{q2}", n=len(te), acc=acc(yt, out), f1=f1(yt, out, K),
                                                 cost=c1 + e1.mean() * c2 + e2_.mean() * c3, esc1=100 * e1.mean(), esc2=100 * e2_.mean()))
                    out = q123.argmax(1); RW.append(dict(task=task, cell=cell, seed=seed, fold=f, kind="avg3", system="+".join(tiers), n=len(te), acc=acc(yt, out), f1=f1(yt, out, K), cost=c1 + c2 + c3))
                # ---- margin router on the selected T15 mixture (secondary) ----
                if E2 in sel and len(sel) > 1 and all(e in av for e in sel):
                    cheap = [e for e in sel if e != E2]; ivp = {e: softmax(load(task, e, "clean", seed)["iv"][f][1]) for e in cheap}
                    thr = np.quantile(margin(np.mean([ivp[e] for e in cheap], 0)), 0.30)
                    ch = np.mean([P[e][te] for e in cheap], 0); full = np.mean([P[e][te] for e in sel], 0); esc = margin(ch) < thr; out = np.where(esc[:, None], full, ch).argmax(1)
                    RW.append(dict(task=task, cell=cell, seed=seed, fold=f, kind="router", system=cname(sel), n=len(te), acc=acc(yt, out), f1=f1(yt, out, K),
                                     cost=sum(CT[e] for e in cheap) + esc.mean() * CT[E2], esc1=100 * esc.mean()))
        print(task, cell, "%.0fs" % (time.time() - t0), flush=True)
pd.DataFrame(RW).to_csv(a.out / "cell_fold_acc.csv", index=False); pd.DataFrame(phis).to_csv(a.out / "phi_fold.csv", index=False); print("rows", len(RW), len(phis))
