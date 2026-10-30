"""T1b jobs (ANALYSIS_PLAN_T1B.md sections 4 and 5). One (seed, fold) per file. E1, difficulty labels, S_2, flagged clips come from the pilot's raw files.
    python t1b_jobs.py --ds urban --phase strength --out DIR --pilot PILOTDIR --seeds 0 1 2
    python t1b_jobs.py --ds urban --phase arms     --out DIR --pilot PILOTDIR --seeds 0 1 2"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common import Q, I5, OPTB, EMB, load_table, pmargin, softmax
from heads import fit_head

ap = argparse.ArgumentParser()
ap.add_argument("--ds", choices=["urban", "tau"], required=True); ap.add_argument("--phase", choices=["strength", "arms"], required=True)
ap.add_argument("--out", required=True, type=Path); ap.add_argument("--pilot", required=True, type=Path)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--folds", type=int, nargs="+", default=None)
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args(); torch.set_num_threads(a.threads)
S2D = Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
if a.ds == "urban":
    d = load_table(); y = d.label.values.astype(np.int64)
    X1 = np.load(EMB["mn10_w8a8"]).astype(np.float32)
    E2 = {"E2a": I5 / "emb/urban8k/fp32__clean.npy", "E2b": I5 / "emb/urban8k/w4g128__clean.npy", "E2c": S2D / "emb_ast_short/urban8k/w4g128__clean.npy"}
    praw = a.pilot / "t1/raw"; folds = a.folds or list(range(1, 11))
else:
    d = pd.read_csv(Path.home() / "Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv", usecols=["folder", "name", "class_id"]); y = d.class_id.values.astype(np.int64)
    X1 = np.load(OPTB / "emb_cnn/tau/mn10_w8a8__clean.npy").astype(np.float32)
    E2 = {"E2a": OPTB / "emb_ast/tau/fp32__clean.npy", "E2b": OPTB / "emb_ast/tau/w4g128__clean.npy"}
    praw = a.pilot / "tau/raw"; folds = a.folds or list(range(1, 6))
XA = {k: np.load(p).astype(np.float32) for k, p in E2.items()}
HC = [1024, 512]
# arm table: (E2, input, arm) ; weights by arm name
ARMS = []
for inp in ("C", "B"):
    ARMS += [("E2a", inp, n) for n in ("U", "M0", "R05", "R10", "R20", "F", "T")]
ARMS += [("E2a", "A", "M0"), ("E2a", "A", "R10")]
for e in [k for k in E2 if k != "E2a"]:
    ARMS += [(e, "C", "U"), (e, "C", "M0"), (e, "C", "R10"), (e, "B", "U"), (e, "B", "R10")]
out_dir = a.out / ("raw_strength" if a.phase == "strength" else "raw_arms"); out_dir.mkdir(parents=True, exist_ok=True)

for seed in a.seeds:
    for fold in folds:
        f_out = out_dir / f"s{seed}_f{fold}.npz"
        if f_out.exists(): continue
        t0 = time.time(); P = np.load(praw / f"s{seed}_f{fold}.npz")
        tr, te, S2, tier, fit_p, iv_p = P["tr"], P["te"], P["S2"], P["tier"], P["fit_p"], P["iv_p"]
        ytr = y[tr]; z1o = P["z1_oof"].astype(np.float64); z1_te = P["z1_te"].astype(np.float64)
        m1o, m1_te = pmargin(softmax(z1o)), pmargin(softmax(z1_te))
        sel_p = iv_p[S2[iv_p]]
        R = dict(te=te, flag=P["flag"], z1_te=P["z1_te"], seed=seed, fold=fold)
        if a.ds == "tau": R["cond"] = P["cond"]
        def feats(e, inp):
            ex_tr = np.hstack([z1o, m1o[:, None]]); ex_te = np.hstack([z1_te, m1_te[:, None]])
            if inp == "A": return XA[e][tr], XA[e][te]
            if inp == "B": return np.hstack([XA[e][tr], ex_tr]).astype(np.float32), np.hstack([XA[e][te], ex_te]).astype(np.float32)
            return np.hstack([X1[tr], XA[e][tr], ex_tr]).astype(np.float32), np.hstack([X1[te], XA[e][te], ex_te]).astype(np.float32)
        ones = np.ones(len(tr))
        W = {"U": (ones, 0.0), "M0": (np.where(S2, 1.0, 0.0), 0.0), "R05": (np.where(S2, 1.0, 0.05), 0.0), "R10": (np.where(S2, 1.0, 0.10), 0.0),
             "R20": (np.where(S2, 1.0, 0.20), 0.0), "F": (ones, 2.0), "T": (np.where(tier >= 1, 1.0, 0.0), 0.0)}
        def run(e, inp, arm):
            Xtr_, Xte_ = feats(e, inp); w, gm = W[arm]
            fp = fit_p[(w > 0)[fit_p]]
            h = fit_head(Xtr_[fp], ytr[fp], w[fp], seed, Xtr_[sel_p], ytr[sel_p], metric="acc", gamma=gm, hidden=HC if inp == "C" else None)
            nm = f"{e}_{inp}_{arm}"; R[nm + "_te"] = h.logits(Xte_).astype(np.float32); R[nm + "_epochs"] = h.epochs_run
        if a.phase == "strength":
            for e in E2: run(e, "A", "U")
        else:
            for e, inp, arm in ARMS: run(e, inp, arm)
        np.savez_compressed(f_out, **R)
        print(f"{a.ds} {a.phase} s{seed} f{fold}: done {time.time()-t0:.0f}s flagged {P['flag'].mean():.3f}", flush=True)
