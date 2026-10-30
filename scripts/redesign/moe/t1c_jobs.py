"""T1c jobs (ANALYSIS_PLAN_T1C.md sections 4 and 9). One (seed, fold) per file; no job reads the label of a test clip.
    python t1c_jobs.py --bed urban|esc50|tau --phase strength|oof|arms --out DIR [--seeds 0 1 2] [--folds ...] [--threads 1]
strength: E1 (retrained), tau_1, U-A heads on every second-tier expert, logits of all of them on the test rows of every cell. ESC-50: also the cross-fit (difficulty, tiers, S_2).
oof: clean out-of-fold logits of the second-tier experts on the training clips (T2b recipe). arms: the T1 arms on E2a."""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.model_selection import StratifiedGroupKFold
from t1c_common import *
from heads import fit_head
from QWave.datasets import inner_split

ap = argparse.ArgumentParser()
ap.add_argument("--bed", choices=list(BEDS), required=True); ap.add_argument("--phase", choices=["strength", "oof", "arms"], required=True)
ap.add_argument("--out", required=True, type=Path); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--folds", type=int, nargs="+", default=None)
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--epochs_cf", type=int, default=30)
a = ap.parse_args(); torch.set_num_threads(a.threads)
bed, K, T = a.bed, BEDS[a.bed]["ncls"], a.epochs_cf
folds = a.folds or list(range(1, BEDS[bed]["nf"] + 1))
D = load_bed(bed); y, rec, fold_of = D["y"], D["rec"], D["fold"]
CELLS = cells_of(bed); EX2 = EXPERTS2[bed]
out_dir = a.out / bed / f"raw_{a.phase}"; out_dir.mkdir(parents=True, exist_ok=True)
strength_dir = a.out / bed / "raw_strength"
mm = lambda ex, cell: np.load(emb_path(bed, ex, cell), mmap_mode="r")
X1c = np.load(emb_path(bed, "E1", "clean")).astype(np.float32)
XC = {e: np.load(emb_path(bed, e, "clean")).astype(np.float32) for e in EX2}


def crossfit(X, tr, seed):
    """Out-of-fold logits per epoch for the training clips tr. [T, len(tr), K]."""
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    o = np.zeros((T, len(tr), K), np.float32)
    for fit_pos, ho_pos in sgkf.split(np.zeros(len(tr)), y[tr], rec[tr]):
        _, tj = fit_head(X[tr[fit_pos]], y[tr[fit_pos]], np.ones(len(fit_pos)), seed, fixed_epochs=T, Xrec=X[tr[ho_pos]], ncls=K)
        o[:, ho_pos] = tj
    return o


def thr_rate(m_iv, rate=0.30):
    u = np.unique(m_iv); c = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    rates = (m_iv[None, :] < c[:, None]).mean(1)
    return c[np.lexsort((rates, np.abs(rates - rate)))[0]]


def difficulty(traj, ytr):
    P = softmax(traj.reshape(-1, K).astype(np.float64)).reshape(traj.shape); n = len(ytr); ar = np.arange(n)
    return 1 - P[:, ar, ytr].mean(0)


for seed in a.seeds:
    for fold in folds:
        f_out = out_dir / f"s{seed}_f{fold}.npz"
        if f_out.exists(): continue
        t0 = time.time(); R = dict(seed=seed, fold=fold)
        if bed == "tau":
            tr = np.flatnonzero((fold_of != fold) & D["is_a"]); te = np.flatnonzero(fold_of == fold)
        else:
            tr = np.flatnonzero(fold_of != fold); te = np.flatnonzero(fold_of == fold)
        ytr = y[tr]
        if a.phase == "strength":
            if bed in ("urban", "tau"):
                P = np.load(PILOT / ("t1/raw" if bed == "urban" else "tau/raw") / f"s{seed}_f{fold}.npz")
                assert (P["tr"] == tr).all() and (P["te"] == te).all()
                fit_p, iv_p, S2m, tier, z1o, tau1_oof = P["fit_p"], P["iv_p"], P["S2"], P["tier"], P["z1_oof"], float(P["tau1_oof"])
                if bed == "tau": R["cond"] = P["cond"]
            else:
                tj1 = crossfit(X1c, tr, seed); d1 = difficulty(tj1, ytr); z1o = tj1[-1]
                m1o = pmargin(softmax(z1o.astype(np.float64))); q50, q80 = np.quantile(d1, [0.5, 0.8])
                tier = np.where(d1 < q50, 0, np.where(d1 >= q80, 2, 1)).astype(np.int8); tau1_oof = float(np.quantile(m1o, 0.30)); S2m = m1o < tau1_oof
                fit_g, iv_g = inner_split(tr, y, 0.1, seed, rec); pos = {g: i for i, g in enumerate(tr)}
                fit_p = np.array([pos[g] for g in fit_g]); iv_p = np.array([pos[g] for g in iv_g]); R["e1_d"] = d1.astype(np.float32)
            R.update(tr=tr, te=te, fit_p=fit_p, iv_p=iv_p, S2=S2m, tier=tier, z1_oof=z1o.astype(np.float32), tau1_oof=tau1_oof)
            fit_g, iv_g = tr[fit_p], tr[iv_p]
            h1 = fit_head(X1c[fit_g], y[fit_g], np.ones(len(fit_g)), seed, X1c[iv_g], y[iv_g], metric="f1", ncls=K)
            z1_iv = h1.logits(X1c[iv_g]); tau1 = thr_rate(pmargin(softmax(z1_iv)), 0.30)
            R.update(tau1=tau1, z1_iv=z1_iv.astype(np.float32), e1_epochs=h1.epochs_run)
            for cell in CELLS: R[f"z1|{cell}"] = h1.logits(np.asarray(mm("E1", cell)[te], np.float32)).astype(np.float32)
            if bed == "urban":      # R1: the retrained E1 equals the pilot's, clean test logits
                R["r1_maxdiff"] = float(np.abs(R["z1|clean"] - P["z1_te"]).max())
            if bed == "tau": R["r1_maxdiff"] = float(np.abs(R["z1|clean"] - P["z1_te"]).max())
            sel_p = iv_p[S2m[iv_p]]; fp = fit_p
            for e in EX2:
                h = fit_head(XC[e][tr[fp]], ytr[fp], np.ones(len(fp)), seed, XC[e][tr[sel_p]], ytr[sel_p], metric="acc", ncls=K)
                for cell in CELLS: R[f"UA_{e}|{cell}"] = h.logits(np.asarray(mm(e, cell)[te], np.float32)).astype(np.float32)
                R[f"UA_{e}_epochs"] = h.epochs_run
            if bed == "urban":      # R2: T1b's U-A heads on E2a, E2b, E2c are reproduced
                Tb = np.load(T1B / f"urban/raw_strength/s{seed}_f{fold}.npz")
                R["r2_maxdiff"] = max(float(np.abs(R[f"UA_{e}|clean"] - Tb[f"{e}_A_U_te"]).max()) for e in EX2)
        else:
            S = np.load(strength_dir / f"s{seed}_f{fold}.npz"); assert (S["tr"] == tr).all() and (S["te"] == te).all()
            fit_p, iv_p, S2m, tier, z1o = S["fit_p"], S["iv_p"], S["S2"], S["tier"], S["z1_oof"].astype(np.float64)
            R["te"] = te
            if a.phase == "oof":
                R["tr"] = tr; R["E1_oof"] = S["z1_oof"]
                for e in EX2:
                    X = XC[e]; oof = np.zeros((len(tr), K), np.float32); sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                    for fit_pos, ho_pos in sgkf.split(np.zeros(len(tr)), ytr, rec[tr]):
                        _, tj = fit_head(X[tr[fit_pos]], ytr[fit_pos], np.ones(len(fit_pos)), seed, fixed_epochs=a.epochs_cf, Xrec=X[tr[ho_pos]], ncls=K)
                        oof[ho_pos] = tj[-1]
                    R[f"{e}_oof"] = oof; print(f"{bed} s{seed} f{fold} {e}: oof acc {(oof.argmax(1) == ytr).mean():.4f}", flush=True)
            else:
                m1o = pmargin(softmax(z1o)); sel_p = iv_p[S2m[iv_p]]; X2 = XC["E2a"]
                XB_tr = np.hstack([X2[tr], z1o, m1o[:, None]]).astype(np.float32)
                ones = np.ones(len(tr))
                W = {"U": (ones, 0.0), "M0": (np.where(S2m, 1.0, 0.0), 0.0), "R05": (np.where(S2m, 1.0, 0.05), 0.0), "R10": (np.where(S2m, 1.0, 0.10), 0.0),
                     "R20": (np.where(S2m, 1.0, 0.20), 0.0), "F": (ones, 2.0), "T": (np.where(tier >= 1, 1.0, 0.0), 0.0)}
                Xte = {}
                for cell in CELLS:
                    z1c = S[f"z1|{cell}"].astype(np.float64); Xte[cell] = np.hstack([np.asarray(mm("E2a", cell)[te], np.float32), z1c, pmargin(softmax(z1c))[:, None]]).astype(np.float32)
                def run(name, Xtr_, Xtest, w, gm):
                    keep = w > 0; fp = fit_p[keep[fit_p]]
                    h = fit_head(Xtr_[fp], ytr[fp], w[fp], seed, Xtr_[sel_p], ytr[sel_p], metric="acc", gamma=gm, ncls=K)
                    for cell in CELLS: R[f"{name}|{cell}"] = h.logits(Xtest[cell]).astype(np.float32)
                    R[f"{name}_epochs"] = h.epochs_run
                for nm, (w, gm) in W.items(): run(nm, XB_tr, Xte, w, gm)
                XA_tr = X2[tr]; XAte = {cell: np.asarray(mm("E2a", cell)[te], np.float32) for cell in CELLS}
                run("M0A", XA_tr, XAte, np.where(S2m, 1.0, 0.0), 0.0); run("R10A", XA_tr, XAte, np.where(S2m, 1.0, 0.10), 0.0)
        np.savez_compressed(f_out, **R)
        print(f"{bed} {a.phase} s{seed} f{fold}: done {time.time()-t0:.0f}s" + (f" r1 {R.get('r1_maxdiff')} r2 {R.get('r2_maxdiff')}" if a.phase == "strength" else ""), flush=True)
