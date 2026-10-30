"""T16 step 3: scoring (the first script that reads test labels; the held-out labels it reads are training labels). Stage A: AUROC for "E1 wrong and strong model right" of the margin, the output-only MLP, the
hidden-state routers at every depth and T2b's stored gate. Stage B: the early-exit system on an escalation-rate grid with held-out rate and comparator choice (ANALYSIS_PLAN_T16.md sections 4, 5).
    python t16_score.py --bed B --variant fp32|w4g128 --heads DIR --router DIR --costs FILE --out DIR [--seeds ..]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.metrics import roc_auc_score
from t16_common import *
from heads import macro_f1

ap = argparse.ArgumentParser(); ap.add_argument("--bed", required=True); ap.add_argument("--variant", default="fp32"); ap.add_argument("--heads", required=True, type=Path); ap.add_argument("--router", required=True, type=Path)
ap.add_argument("--costs", required=True, type=Path); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS); ap.add_argument("--conds", nargs="*", default=None); ap.add_argument("--folds", type=int, nargs="+", default=None); a = ap.parse_args()
bed, var = a.bed, a.variant; K = BEDS[bed]["ncls"]; D = load_bed(bed); y = D["y"]; CONDS = a.conds or list(EVAL[bed]); a.out.mkdir(parents=True, exist_ok=True)
C = json.load(open(a.costs))[bed]; V = C["fp32" if var == "fp32" else "w4g128"]; STRONG = "S" if var == "fp32" else V["strong"]
MODELS = ["E1"] + EXPERTS2[bed] + ["S"]; SC = C["singles"]; MC = np.array([SC[m] for m in MODELS])
KINDS = [f"d{d}" for d in DEPTHS] + ["out"] + [f"p{d}" for d in DEPTHS]; RATES = [*np.arange(.05, .951, .05), .99]
T2B_CONDS = {"U-field", "U-0dB", "E-field", "E-0dB", "T-shift"}


def auc(lab, s):
    ok = np.isfinite(s)
    return float(roc_auc_score(lab[ok], s[ok])) if ok.sum() > 1 and 0 < lab[ok].sum() < ok.sum() else np.nan


def hull(acc, budget):
    """Time-sharing hull of the single models (acc, cost arrays aligned with MODELS) at `budget`: value, lo, hi, lam. Singles above the budget are excluded."""
    best = (-1.0, None, None, 0.0)
    for i in range(len(MODELS)):
        if MC[i] <= budget * (1 + 1e-12) and acc[i] > best[0]: best = (acc[i], i, i, 0.0)
        for j in range(len(MODELS)):
            if MC[i] < budget < MC[j]:
                lam = (budget - MC[i]) / (MC[j] - MC[i]); v = (1 - lam) * acc[i] + lam * acc[j]
                if v > best[0]: best = (v, i, j, lam)
    return best


def sys_cost(kind, esc):
    b = int(kind[1:]) if kind != "out" else None
    if kind == "out": return C["E1"] + C["router_out"] + esc * V["S_tot"]
    return C["E1"] + C["fbank"] + C["patch"] + V["enc_per_block"] * b + C["router"] + esc * (V["enc_per_block"] * (12 - b) + C["head"])


rowsA, rowsI, rowsB, rowsR = [], [], [], []
for seed in a.seeds:
    for fold in (a.folds or range(1, BEDS[bed]["nf"] + 1)):
        H = np.load(a.heads / bed / f"s{seed}_f{fold}.npz"); R = np.load(a.router / bed / f"s{seed}_f{fold}.npz"); T2 = np.load(T1C / bed / f"t2b_fit/s{seed}_f{fold}.npz")
        for cn in CONDS:
            P = {}
            for part in ("te", "iv"):
                rows = H[f"{cn}|{part}_rows"]; yy = y[rows]; Z = {m: H[f"L|{m}|{cn}|{part}"] for m in MODELS}; ok = np.isfinite(Z["E1"]).all(1) & np.isfinite(Z[STRONG]).all(1)
                z1p = Z["E1"].argmax(1); zsp = Z[STRONG].argmax(1); lab = ((z1p != yy) & (zsp == yy)).astype(int)
                sc = {"margin": -pmargin(softmax(Z["E1"].astype(np.float64)))}
                for k in KINDS: sc[k] = R[f"sc|{k}|{cn}|{part}"]
                if part == "te" and cn in T2B_CONDS and f"{cn}|A3h_w" in T2.files:
                    w = T2[f"{cn}|A3h_w"]; assert len(w) == len(rows); sc["T2b_A3h"] = 1.0 - w[:, 0]
                P[part] = dict(yy=yy, Z=Z, z1p=z1p, zsp=zsp, lab=lab, sc=sc, ok=ok)
                for k, s in sc.items():
                    (rowsA if part == "te" else rowsI).append(dict(variant=var, bed=bed, cond=cn, kind=k, seed=seed, fold=fold, auroc=auc(lab, np.where(ok, s, np.nan)), n=int(ok.sum()), n_pos=int(lab[ok].sum())))
            # ---- Stage B ----
            te, iv = P["te"], P["iv"]
            acc_te = np.array([(te["Z"][m].argmax(1) == te["yy"])[te["ok"]].mean() for m in MODELS]) * 100; acc_iv = np.array([(iv["Z"][m].argmax(1) == iv["yy"])[iv["ok"]].mean() for m in MODELS]) * 100
            for i, m in enumerate(MODELS): rowsR.append(dict(variant=var, bed=bed, cond=cn, ref=m, seed=seed, fold=fold, acc=acc_te[i], cost=MC[i]))
            if cn in T2B_CONDS and f"{cn}|A1_p" in T2.files:
                for arm in ("A1", "A2h", "A3h", "A4"):
                    p = T2[f"{cn}|{arm}_p"]; rowsR.append(dict(variant=var, bed=bed, cond=cn, ref="T2b_" + arm, seed=seed, fold=fold, acc=(p.argmax(1) == te["yy"]).mean() * 100, cost=float(np.mean(T2[f"{cn}|{arm}_cost"]))))
            oh_budget = lambda c: hull(acc_te, c)[0]                          # test-chosen hull (T13 convention, favours the hull), descriptive
            for k in KINDS:
                if k == "d12" or True:
                    s_te, s_iv = te["sc"][k], iv["sc"][k]
                    for r in RATES:
                        thr = np.quantile(s_iv[iv["ok"]], 1 - r); out = {}
                        for nm, Pp, s in (("iv", iv, s_iv), ("te", te, s_te)):
                            esc = (s >= thr) & Pp["ok"]; pred = np.where(esc, Pp["zsp"], Pp["z1p"]); cor = (pred == Pp["yy"])[Pp["ok"]]; er = esc[Pp["ok"]].mean()
                            out[nm] = dict(acc=cor.mean() * 100, cost=sys_cost(k, er), er=er, pred=pred[Pp["ok"]], yy=Pp["yy"][Pp["ok"]])
                        hv_iv = hull(acc_iv, out["iv"]["cost"])[0]; budget = out["te"]["cost"]; hv, lo, hi, lam = hull(acc_iv, budget)       # pair and weight fixed on held-out rows at the test budget
                        h_te = (1 - lam) * acc_te[lo] + lam * acc_te[hi]
                        okm = [i for i in range(len(MODELS)) if MC[i] <= budget * (1 + 1e-12)]; bs = max(okm, key=lambda i: acc_iv[i])
                        rowsB.append(dict(variant=var, bed=bed, cond=cn, kind=k, seed=seed, fold=fold, rate=r, acc=out["te"]["acc"], cost=budget, esc_rate=out["te"]["er"], hull=h_te, hull_lo=MODELS[lo], hull_hi=MODELS[hi], hull_lam=lam,
                                          best_single=acc_te[bs], best_single_name=MODELS[bs], hull_oracle=oh_budget(budget), iv_gain=out["iv"]["acc"] - hv_iv, f1=macro_f1(out["te"]["yy"], out["te"]["pred"], K) * 100))
    print(f"{bed} {var} s{seed}: scored", flush=True)
pd.DataFrame(rowsA).to_csv(a.out / f"auroc_test_{bed}_{var}.csv", index=False); pd.DataFrame(rowsI).to_csv(a.out / f"auroc_iv_{bed}_{var}.csv", index=False)
pd.DataFrame(rowsB).to_csv(a.out / f"system_{bed}_{var}.csv", index=False); pd.DataFrame(rowsR).to_csv(a.out / f"refs_{bed}_{var}.csv", index=False)
