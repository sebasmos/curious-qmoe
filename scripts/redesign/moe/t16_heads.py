"""T16 step 1 (CPU): heads and logits. Per (seed, fold): refit E1, E2c, E2b, E2a (T1c recipe, checked against the stored T1c test logits), fit S (FP32 AST at the task's frame count, uniform head,
T1c 'U-A' recipe), cross-fit S on the training clips (5 grouped folds, 30 epochs, the T1c/T2b recipe), and write the logits of every model on the items of every evaluation condition for the test rows
and for the held-out rows (inner validation under the validation field assignment; TAU: the shifted rows of the training folds). Reads only y of training and held-out rows; no test label.
    python t16_heads.py --bed urban|esc50|tau [--seeds ..] [--folds ..] --out DIR"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.model_selection import StratifiedGroupKFold
from t16_common import *
from heads import fit_head

ap = argparse.ArgumentParser(); ap.add_argument("--bed", required=True); ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS); ap.add_argument("--folds", type=int, nargs="+", default=None); ap.add_argument("--conds", nargs="*", default=None); ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args(); torch.set_num_threads(a.threads)
bed = a.bed; K = BEDS[bed]["ncls"]; D = load_bed(bed); y, rec, fold_of, N = D["y"], D["rec"], D["fold"], D["N"]
EX2 = EXPERTS2[bed]; MODELS = ["E1"] + EX2 + ["S"]; CONDS = a.conds or list(EVAL[bed])
folds = a.folds or list(range(1, BEDS[bed]["nf"] + 1)); (a.out / bed).mkdir(parents=True, exist_ok=True)
assign_val = None if bed == "tau" else np.array(field_assignment(N, FIELD_SEEDS["val"]))
X1c = np.load(emb_path(bed, "E1", "clean")).astype(np.float32); XC = {e: np.load(emb_path(bed, e, "clean")).astype(np.float32) for e in EX2}
XSc = np.load(emb16_path(bed, "clean")).astype(np.float32)


def emb_fn(m):
    if m == "S": return lambda cell: emb16_path(bed, cell)
    return lambda cell, m=m: emb_path(bed, m, cell)


def logits_items(h, m, rows, cells):
    X = take(emb_fn(m), rows, cells); ok = np.isfinite(X).all(1); Z = np.full((len(rows), K), np.nan, np.float32)
    if ok.any(): Z[ok] = h.logits(X[ok].astype(np.float32)).astype(np.float32)
    return Z


for seed in a.seeds:
    for fold in folds:
        fo = a.out / bed / f"s{seed}_f{fold}.npz"
        if fo.exists(): continue
        t0 = time.time(); St = np.load(T1C / bed / f"raw_strength/s{seed}_f{fold}.npz"); tr, te, fp, ip, S2m = St["tr"], St["te"], St["fit_p"], St["iv_p"], St["S2"]
        ytr = y[tr]; cond_te = St["cond"] if bed == "tau" else None; R = dict(tr=tr, te=te, fit_p=fp, iv_p=ip)
        fit_g, iv_g = tr[fp], tr[ip]; sel_p = ip[S2m[ip]]
        H = {"E1": fit_head(X1c[fit_g], y[fit_g], np.ones(len(fit_g)), seed, X1c[iv_g], y[iv_g], metric="f1", ncls=K)}
        for m in EX2 + ["S"]:
            X = XSc if m == "S" else XC[m]
            H[m] = fit_head(X[tr[fp]], ytr[fp], np.ones(len(fp)), seed, X[tr[sel_p]], ytr[sel_p], metric="acc", ncls=K)
        # S out-of-fold logits on the training clips (E1's are the stored z1_oof)
        oof = np.zeros((len(tr), K), np.float32); sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fit_pos, ho_pos in sgkf.split(np.zeros(len(tr)), ytr, rec[tr]):
            _, tj = fit_head(XSc[tr[fit_pos]], ytr[fit_pos], np.ones(len(fit_pos)), seed, fixed_epochs=30, Xrec=XSc[tr[ho_pos]], ncls=K); oof[ho_pos] = tj[-1]
        R["S_oof"] = oof; R["E1_oof"] = St["z1_oof"]
        # held-out rows: inner validation (all beds), TAU additionally the shifted rows of the training folds
        iv_rows = tr[ip]
        if bed == "tau":
            sh_rows = np.flatnonzero((fold_of != fold) & (~D["is_a"])); R["sh_rows"] = sh_rows
        maxdiff = {}
        for cn in CONDS:
            kind = EVAL[bed][cn][0]
            tr_, tc_ = items(bed, cn, te, D["assign"], cond_te); R[f"{cn}|te_rows"], R[f"{cn}|te_cells"] = tr_, tc_
            if bed == "tau": ivr, ivc = (items(bed, cn, sh_rows, None, np.ones(len(sh_rows), int)) if cn == "T-shift" else items(bed, cn, iv_rows, None, np.zeros(len(iv_rows), int)))
            else: ivr, ivc = items(bed, cn, iv_rows, assign_val)
            R[f"{cn}|iv_rows"], R[f"{cn}|iv_cells"] = ivr, ivc
            for m in MODELS:
                R[f"L|{m}|{cn}|te"] = logits_items(H[m], m, tr_, tc_); R[f"L|{m}|{cn}|iv"] = logits_items(H[m], m, ivr, ivc)
        # checks against the stored T1c logits (test rows, clean and bg 0 dB / shift)
        for m in ["E1"] + EX2:
            key = (lambda c: f"z1|{c}") if m == "E1" else (lambda c, m=m: f"UA_{m}|{c}"); d = 0.0
            for cell in (["clean"] if bed == "tau" else ["clean", "bg__0db"]):
                mine = logits_items(H[m], m, te, np.array([cell] * len(te))); d = max(d, float(np.nanmax(np.abs(mine - St[key(cell)]))))
            maxdiff[m] = d
        for m, d in maxdiff.items(): R[f"maxdiff_{m}"] = d
        np.savez_compressed(fo, **R)
        print(f"{bed} s{seed} f{fold}: {time.time() - t0:.0f}s stored-logit maxdiff " + " ".join(f"{m} {d:.2e}" for m, d in maxdiff.items()) + f" S_oof acc {(oof.argmax(1) == ytr).mean():.4f}", flush=True)
