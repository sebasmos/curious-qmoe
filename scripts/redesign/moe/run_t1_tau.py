"""TAU 2020 Mobile version of run_t1.py (ANALYSIS_PLAN.md Amendment 1): train on device-A rows of the training folds, test on the fold's device-A rows
(clean) and its other-device rows (shift). E1 mn10 W8A8, E2 AST w4g128 at 1024 frames, no E3.
Writes <out>/raw/s<seed>_f<fold>.npz; analysis scripts read only those files.
    python run_t1.py --out DIR --seeds 0 1 2 --folds 1 2 ... 10"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.model_selection import StratifiedGroupKFold
from common import pmargin, softmax
from heads import fit_head, macro_f1, NCLS
from QWave.datasets import inner_split

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, 6)))
ap.add_argument("--epochs_cf", type=int, default=30)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
(a.out / "raw").mkdir(parents=True, exist_ok=True)

TAU = Path.home() / "Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv"
OPTB = Path.home() / "Documents/qmoe-results/2026-10-23_2133_redesign-idea6-optionB_code-f320295"
d = pd.read_csv(TAU, usecols=["folder", "name", "class_id"])
y = d.class_id.values.astype(np.int64)
rec = d.name.str.split("-").str[2].astype(int).values            # location id, token 2
dev = d.name.str.replace(".wav", "", regex=False).str.split("-").str[-1].values
fold_of = d.folder.values
is_a = dev == "a"
X1 = np.load(OPTB / "emb_cnn/tau/mn10_w8a8__clean.npy").astype(np.float32)
X2 = np.load(OPTB / "emb_ast/tau/w4g128__clean.npy").astype(np.float32)
NCLS_ = int(y.max()) + 1
assert len(d) == 23035 and X1.shape[0] == 23035 and X2.shape[0] == 23035 and NCLS_ == 10
T = a.epochs_cf


def crossfit(X, tr, seed, keep_traj):
    """Out-of-fold logits per epoch for the training clips tr (positions in tr). Returns [T, len(tr), C]."""
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    out = np.zeros((T, len(tr), NCLS), np.float32)
    for fit_pos, ho_pos in sgkf.split(np.zeros(len(tr)), y[tr], rec[tr]):
        _, tj = fit_head(X[tr[fit_pos]], y[tr[fit_pos]], np.ones(len(fit_pos)), seed, fixed_epochs=T, Xrec=X[tr[ho_pos]])
        out[:, ho_pos] = tj
    return out


def difficulty(traj, ytr):
    P = softmax(traj.reshape(-1, NCLS).astype(np.float64)).reshape(traj.shape)
    n = len(ytr)
    ar = np.arange(n)
    ptrue = P[:, ar, ytr]                                  # [T, n]
    zt = traj[:, ar, ytr].astype(np.float64)
    zo = traj.astype(np.float64).copy(); zo[:, ar, ytr] = -np.inf
    aum = (zt - zo.max(2)).mean(0)
    onehot = np.zeros((n, NCLS)); onehot[ar, ytr] = 1
    el2n = np.linalg.norm(P[4] - onehot, axis=1)           # epoch 5
    correct = (P.argmax(2) == ytr[None, :])
    forget = ((correct[:-1]) & (~correct[1:])).sum(0)
    return dict(d=1 - ptrue.mean(0), aum=aum, el2n=el2n, forget=forget)


for seed in a.seeds:
    for fold in a.folds:
        f_out = a.out / "raw" / f"s{seed}_f{fold}.npz"
        if f_out.exists():
            continue
        t0 = time.time()
        tr = np.flatnonzero((fold_of != fold) & is_a); te = np.flatnonzero(fold_of == fold)
        cond = (~is_a[te]).astype(np.int8)   # 0 clean (device A), 1 shift (other devices)
        ytr, yte = y[tr], y[te]
        R = {"tr": tr, "te": te, "cond": cond, "dev": dev[te]}
        # B3: out-of-fold trajectories on the training clips
        tj1 = crossfit(X1, tr, seed, True)
        tj2 = crossfit(X2, tr, seed, True)
        dif1, dif2 = difficulty(tj1, ytr), difficulty(tj2, ytr)
        for k, v in dif1.items(): R[f"e1_{k}"] = v
        for k, v in dif2.items(): R[f"e2_{k}"] = v
        z1o, z2o = tj1[-1].astype(np.float64), tj2[-1].astype(np.float64)
        p1o, p2o = softmax(z1o), softmax(z2o)
        m1o, m2o = pmargin(p1o), pmargin(p2o)
        R.update(z1_oof=z1o.astype(np.float32), z2_oof=z2o.astype(np.float32))
        q50, q80 = np.quantile(dif1["d"], [0.5, 0.8])
        tier = np.where(dif1["d"] < q50, 0, np.where(dif1["d"] >= q80, 2, 1))
        R["tier"] = tier.astype(np.int8)
        tau1_oof = np.quantile(m1o, 0.30)
        S2 = m1o < tau1_oof
        R["S2"] = S2
        print(f"s{seed} f{fold}: crossfit done {time.time()-t0:.0f}s, oof acc E1 {(p1o.argmax(1)==ytr).mean():.3f} E2 {(p2o.argmax(1)==ytr).mean():.3f}, |S2| {S2.sum()}", flush=True)
        # B4: final E1 with the protocol 3 inner split
        fit_g, iv_g = inner_split(tr, y, 0.1, seed, rec)
        pos = {g: i for i, g in enumerate(tr)}
        fit_p = np.array([pos[g] for g in fit_g]); iv_p = np.array([pos[g] for g in iv_g])
        h1 = fit_head(X1[fit_g], y[fit_g], np.ones(len(fit_g)), seed, X1[iv_g], y[iv_g], metric="f1")
        z1_iv, z1_te = h1.logits(X1[iv_g]), h1.logits(X1[te])
        p1_iv, p1_te = softmax(z1_iv), softmax(z1_te)
        m1_iv, m1_te = pmargin(p1_iv), pmargin(p1_te)
        u = np.unique(m1_iv); c = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
        rates = (m1_iv[None, :] < c[:, None]).mean(1)
        tau1 = c[np.lexsort((rates, np.abs(rates - 0.30)))[0]]       # analyze.thr_rate(m_iv, 0.30)
        flag = m1_te < tau1
        R.update(fit_p=fit_p, iv_p=iv_p, tau1=tau1, tau1_oof=tau1_oof, z1_te=z1_te.astype(np.float32), z1_iv=z1_iv.astype(np.float32),
                 flag=flag, e1_epochs=h1.epochs_run)
        # B5: selection set = inner validation clips with m1_oof < tau1_oof
        sel_p = iv_p[S2[iv_p]]
        # inputs
        def inp_B(idx_tr_pos=None):
            tr_feat = np.hstack([X2[tr], z1o, m1o[:, None]]).astype(np.float32)
            te_feat = np.hstack([X2[te], z1_te, m1_te[:, None]]).astype(np.float32)
            return tr_feat, te_feat
        XB_tr, XB_te = inp_B()
        XA_tr, XA_te = X2[tr], X2[te]
        # U-A first (strong model of T1a)
        def run_arm(Xtr_, Xte_, w, gamma, sel_pos, name, out):
            keep = w > 0
            fp = fit_p[keep[fit_p]]
            h = fit_head(Xtr_[fp], ytr[fp], w[fp], seed, Xtr_[sel_pos], ytr[sel_pos], metric="acc", gamma=gamma)
            out[name + "_te"] = h.logits(Xte_).astype(np.float32)
            out[name + "_epochs"] = h.epochs_run
            return h
        ones = np.ones(len(tr))
        hUA = run_arm(XA_tr, XA_te, ones, 0, sel_p, "UA", R)
        z2_te = R["UA_te"].astype(np.float64); m2_te = pmargin(softmax(z2_te))
        z2_iv = hUA.logits(XA_tr[iv_p]); m2_iv = pmargin(softmax(z2_iv))
        R.update(z2_iv=z2_iv.astype(np.float32))
        run_arm(XA_tr, XA_te, np.where(S2, 1.0, 0.0), 0, sel_p, "M0A", R)
        run_arm(XA_tr, XA_te, np.where(S2, 1.0, 0.1), 0, sel_p, "R10A", R)
        W = {"U": (ones, 0), "M0": (np.where(S2, 1.0, 0.0), 0), "R05": (np.where(S2, 1.0, 0.05), 0), "R10": (np.where(S2, 1.0, 0.10), 0),
             "R20": (np.where(S2, 1.0, 0.20), 0), "F": (ones, 2.0), "T": (np.where(tier >= 1, 1.0, 0.0), 0)}
        for nm, (w, gm) in W.items():
            run_arm(XB_tr, XB_te, w, gm, sel_p, nm, R)
        R.update(m2_te=m2_te, m1_te=m1_te)
        np.savez_compressed(f_out, **R)
        print(f"s{seed} f{fold}: done {time.time()-t0:.0f}s flagged clean {flag[cond==0].mean():.3f} shift {flag[cond==1].mean():.3f}", flush=True)
