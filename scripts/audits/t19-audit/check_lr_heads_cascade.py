"""Check 7b (sensitivity, not pre-registered): does the H1 failure depend on the T19 MLP heads? Every AST static and both tiers get a
standardised logistic-regression head (sklearn, C=1) fitted on the fit rows of T19's seed-0 grouped split of each fold; budgets and rule
picked on clean inner validation against the inner-validation hull, as in B7; the cascade is evaluated on the field mix and clean against
the test hull (all LR-head AST statics, stored CNN tiers seed 0, always-average). One seed, so fold values are noisier than T19's."""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from audit_core import R, S2, TASK, NEW, LEG, field_assign, mf1, softmax, margin, cost, hull, thr_closest, nb, get

CF = ["clean"] + [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]
BUD, RULES = [0.05, 0.10, 0.20, 0.30], ["B", "avg"]
task = sys.argv[1]
emb = {"esc50": "esc50", "urban": "urban8k"}[task]
t = TASK[task]
K = t["K"]
meta = pd.read_csv(t["csv"], usecols=["folder", t["label"]])
y, fold = meta[t["label"]].to_numpy(), meta["folder"].to_numpy()
n = len(y)
at, av = field_assign(n, 5001), field_assign(n, 5002)
FOLDS = list(range(1, t["nf"] + 1))
splits = {k: json.loads((R / "heads" / task / "w8a8" / "s0" / f"fold_{k}" / "split.json").read_text()) for k in FOLDS}


def X_of(cfg):
    d = (S2 / "emb_ast_short" / emb) if cfg.startswith("L:") else (R / "emb" / emb)
    st = cfg.replace("L:", "")
    Xc = np.load(d / f"{st}__clean.npy")
    Xf = Xc.copy()
    for c in CF[1:]:
        Xn = np.load(d / f"{st}__{c}.npy", mmap_mode="r")
        m = at == c
        Xf[m] = Xn[m]
    return Xc, Xf


P = {}   # cfg -> dict(test_clean [n,K], test_field [n,K], iv {k: (idx, probs)})
for cfg in NEW + ["L:" + l for l in LEG]:
    Xc, Xf = X_of(cfg)
    pc, pf = np.full((n, K), np.nan), np.full((n, K), np.nan)
    iv = {}
    for k in FOLDS:
        ivi = np.array(splits[k]["inner_val_idx"])
        te = np.flatnonzero(fold == k)
        fit = np.setdiff1d(np.flatnonzero(fold != k), ivi)
        sc = StandardScaler().fit(Xc[fit])
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Xc[fit]), y[fit])
        assert list(lr.classes_) == list(range(K))
        pc[te] = lr.predict_proba(sc.transform(Xc[te]))
        pf[te] = lr.predict_proba(sc.transform(Xf[te]))
        iv[k] = (ivi, lr.predict_proba(sc.transform(Xc[ivi])))
    P[cfg] = dict(clean=pc, field=pf, iv=iv)
    print(cfg, "done", flush=True)

# stored CNN tiers, seed 0
for c in ("mn10", "mn20"):
    Dc, Df = get(task, c, 0, "clean"), get(task, c, 0, "field")
    P[c] = dict(clean=softmax(Dc["z"]), field=softmax(Df["z"]), iv={k: (Dc["iv"][k][0], softmax(Dc["iv"][k][1])) for k in FOLDS})


def cst(c):
    return cost(task, c if c in ("mn10", "mn20") or c.startswith("L:") else c + ":own")


def test_f1(p, bed):
    return np.array([mf1(y[fold == k], p[bed][fold == k].argmax(1), K) for k in FOLDS])


def iv_f1(p):
    return float(np.mean([mf1(y[p["iv"][k][0]], p["iv"][k][1].argmax(1), K) for k in FOLDS]))


cc, ce = cst("n4a4"), cst("w8a8")


def casc(bed, b, rule, iv_mode=False):
    f1s, rates = [], []
    for k in FOLDS:
        ivi, ivp = P["n4a4"]["iv"][k]
        thr = thr_closest(margin(ivp), b)
        if iv_mode:
            yy, pc, pe = y[ivi], ivp, P["w8a8"]["iv"][k][1]
        else:
            m = fold == k
            yy, pc, pe = y[m], P["n4a4"][bed][m], P["w8a8"][bed][m]
        e = margin(pc) < thr
        fu = pe if rule == "B" else (pc + pe) / 2
        f1s.append(mf1(yy, np.where(e, fu.argmax(1), pc.argmax(1)), K))
        rates.append(e.mean())
    return np.array(f1s), float(np.mean(rates))


avg = {bed: dict(clean=None) for bed in ("clean", "field")}
pts_iv = [(c, cst(c), iv_f1(P[c])) for c in P]
rows = []
for b in BUD:
    for rule in RULES:
        f, r = casc("clean", b, rule, iv_mode=True)
        c = cc + r * ce
        rows.append(dict(budget=b, rule=rule, iv_f1=f.mean(), iv_rate=r, adv=f.mean() - hull(pts_iv, c)[0], iv_cost=c))
PK = pd.DataFrame(rows).sort_values(["adv", "iv_cost"], ascending=[False, True])
pb, pr = PK.iloc[0].budget, PK.iloc[0].rule
out = dict(task=task, pick=dict(budget=float(pb), rule=pr), statics={}, points=[])
ratio = float(np.mean([(fold == k).sum() / (fold != k).sum() for k in FOLDS]))
for bed in ("field", "clean"):
    arr = {c: test_f1(P[c], bed) for c in P}
    arr["always-average"] = np.array([mf1(y[fold == k], (P["n4a4"][bed][fold == k] + P["w8a8"][bed][fold == k]).argmax(1), K) for k in FOLDS])
    costs = {c: cst(c) for c in P}
    costs["always-average"] = cc + ce
    pts = [(c, costs[c], arr[c].mean()) for c in arr]
    out["statics"][bed] = {c: round(float(arr[c].mean()), 3) for c in arr}
    for b in BUD:
        for rule in RULES:
            f, r = casc(bed, b, rule)
            c = cc + r * ce
            v, lo, hi, lam = hull(pts, c)
            h = arr[lo] if hi is None else (1 - lam) * arr[lo] + lam * arr[hi]
            s = nb(f - h, ratio)
            out["points"].append(dict(bed=bed, budget=b, rule=rule, picked=bool(b == pb and rule == pr), f1=f.mean(), rate=r, cost_T=c / 1e12,
                                      hull=h.mean(), pair=f"{lo}" + (f" + {hi} lam {lam:.3f}" if hi else ""), diff=s["mean"], p=s["p"],
                                      folds_ge1=int((f - h >= 1).sum())))
df = pd.DataFrame(out["points"])
print(f"pick b{pb} {pr}")
print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("statics field:", sorted(out["statics"]["field"].items(), key=lambda kv: -kv[1])[:10])
json.dump(out, open(f"lr_heads_cascade_{task}.json", "w"), indent=1, default=float)
