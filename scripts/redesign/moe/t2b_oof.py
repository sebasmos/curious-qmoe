"""T2b step 1: out-of-fold logits of E2a, E2b, E2c on the training clips of every outer fold (training clips only), and the self-contained
raw file the gate scripts read. E1 and the inner split come from the pilot, E2 test logits from T1b (copied, not scored).
    python t2b_oof.py --out DIR --pilot PILOTDIR --t1b T1BDIR --seeds 0 1 2 --folds 1 2 ..."""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.model_selection import StratifiedGroupKFold
from common import Q, I5, load_table
from heads import fit_head, NCLS

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path); ap.add_argument("--pilot", required=True, type=Path); ap.add_argument("--t1b", required=True, type=Path)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--epochs_cf", type=int, default=30)
a = ap.parse_args(); torch.set_num_threads(a.threads)
S2D = Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
E2 = {"E2a": I5 / "emb/urban8k/fp32__clean.npy", "E2b": I5 / "emb/urban8k/w4g128__clean.npy", "E2c": S2D / "emb_ast_short/urban8k/w4g128__clean.npy"}
d = load_table(); y = d.label.values.astype(np.int64); rec = d.recording.values
XA = {k: np.load(p).astype(np.float32) for k, p in E2.items()}
(a.out / "raw").mkdir(parents=True, exist_ok=True)
for seed in a.seeds:
    for fold in a.folds:
        f_out = a.out / "raw" / f"s{seed}_f{fold}.npz"
        if f_out.exists(): continue
        t0 = time.time(); P = np.load(a.pilot / f"t1/raw/s{seed}_f{fold}.npz"); T = np.load(a.t1b / f"urban/raw_strength/s{seed}_f{fold}.npz")
        tr, te = P["tr"], P["te"]; assert (T["te"] == te).all() and (T["flag"] == P["flag"]).all()
        R = dict(tr=tr, te=te, fit_p=P["fit_p"], iv_p=P["iv_p"], flag=P["flag"], z1_oof=P["z1_oof"], z1_te=P["z1_te"], seed=seed, fold=fold)
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)   # same splits as the pilot's E1 cross-fit
        for e, X in XA.items():
            oof = np.zeros((len(tr), NCLS), np.float32)
            for fit_pos, ho_pos in sgkf.split(np.zeros(len(tr)), y[tr], rec[tr]):
                _, tj = fit_head(X[tr[fit_pos]], y[tr[fit_pos]], np.ones(len(fit_pos)), seed, fixed_epochs=a.epochs_cf, Xrec=X[tr[ho_pos]])
                oof[ho_pos] = tj[-1]
            R[f"{e}_oof"] = oof; R[f"{e}_te"] = T[f"{e}_A_U_te"]
            print(f"s{seed} f{fold} {e}: oof acc {(oof.argmax(1) == y[tr]).mean():.4f}", flush=True)
        np.savez_compressed(f_out, **R); print(f"s{seed} f{fold}: done {time.time()-t0:.0f}s", flush=True)
