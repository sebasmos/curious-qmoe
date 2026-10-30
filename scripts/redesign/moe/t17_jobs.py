"""T17 jobs (ANALYSIS_PLAN_T17.md sections 2 to 4). One (seed, fold) per file; no job reads the label of a test clip (test logits are stored, scored later).
    python t17_jobs.py --bed urban|esc50|tau --out DIR [--seeds 0 1 2] [--folds ...] [--threads 1]
Reads the T1c strength and oof files (split, flags, cross-fitted E1 logits, cross-fitted AST FP32 logits)."""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from t17_common import *
from t17_heads import train_group

ap = argparse.ArgumentParser()
ap.add_argument("--bed", choices=list(BEDS), required=True); ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--folds", type=int, nargs="+", default=None)
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--arms", default="all", help="all | a | b (smoke tests)")
a = ap.parse_args(); torch.set_num_threads(a.threads)
bed, K = a.bed, BEDS[a.bed]["ncls"]; folds = a.folds or list(range(1, BEDS[bed]["nf"] + 1))
D = load_bed(bed); y, fold_of = D["y"], D["fold"]; CELLS = cells_of(bed)
out_dir = a.out / bed / "raw"; out_dir.mkdir(parents=True, exist_ok=True)
XC = {ex: np.load(emb17(bed, ex, "clean")).astype(np.float32) for ex in ("E1", "M20")}
MM = {}
def mm(ex, cell):
    if (ex, cell) not in MM: MM[(ex, cell)] = np.load(emb17(bed, ex, cell), mmap_mode="r")
    return MM[(ex, cell)]
LAMS = [0.1, 0.3, 0.5]; TS = [1, 2, 4]; ALPHAS = [0.25, 0.5, 0.75]

for seed in a.seeds:
    for fold in folds:
        f_out = out_dir / f"s{seed}_f{fold}.npz"
        if f_out.exists(): continue
        t0 = time.time()
        S = np.load(T1C / bed / f"raw_strength/s{seed}_f{fold}.npz"); O = np.load(T1C / bed / f"raw_oof/s{seed}_f{fold}.npz")
        tr = np.flatnonzero((fold_of != fold) & D["is_a"]) if bed == "tau" else np.flatnonzero(fold_of != fold); te = np.flatnonzero(fold_of == fold)
        assert (S["tr"] == tr).all() and (S["te"] == te).all() and (O["te"] == te).all()
        fit_p, iv_p, S2m = S["fit_p"], S["iv_p"], S["S2"]; fit_g, iv_g = tr[fit_p], tr[iv_p]; ytr = y[tr]
        z1o = S["z1_oof"].astype(np.float64); miss = (z1o.argmax(1) != ytr)
        teach = O["E2a_oof"].astype(np.float32)
        fam = families(bed, D, iv_g, fold)
        # embeddings of the test rows (per cell) and of the inner-validation families, per expert
        TE = {ex: {c: np.asarray(mm(ex, c)[te], np.float32) for c in CELLS} for ex in XC}
        FM = {ex: {f: np.concatenate([np.asarray(mm(ex, c)[r], np.float32) for c, r in segs], 0) for f, segs in fam.items()} for ex in XC}
        R = dict(seed=seed, fold=fold, tr=tr, te=te, fit_p=fit_p, iv_p=iv_p)
        for f, segs in fam.items():
            R[f"famrows|{f}"] = np.concatenate([r for _, r in segs]); R[f"famcell|{f}"] = np.concatenate([[ALLCELLS.index(c) if c in ALLCELLS else 0] * len(r) for c, r in segs]).astype(np.int8)
        EX = ["E1", "M20"]
        def emit(name, G, exs):  # members of G are on the experts exs, in order
            for c in CELLS:
                zs = G.logits([TE[ex][c] for ex in exs])
                for mi, z in enumerate(zs): R[f"te|{name}|{mi}|{c}"] = z.astype(np.float32)
            for f in fam:
                zs = G.logits([FM[ex][f] for ex in exs])
                for mi, z in enumerate(zs): R[f"iv|{name}|{mi}|{f}"] = z.astype(np.float32)
            R[f"epochs|{name}"] = G.epochs_run
        Xf = {ex: XC[ex][fit_g] for ex in XC}; Xs = {ex: XC[ex][iv_g] for ex in XC}; yf, ys = y[fit_g], y[iv_g]; one = np.ones(len(fit_g))
        G = {}
        do_a, do_b = a.arms in ("all", "a"), a.arms in ("all", "b")
        G["p10"] = train_group([Xf["E1"]], yf, [one], seed, [Xs["E1"]], ys, K); emit("p10", G["p10"], ["E1"])
        if do_a:
            G["p10b"] = train_group([Xf["E1"]], yf, [one], seed + 100, [Xs["E1"]], ys, K); emit("p10b", G["p10b"], ["E1"])
            G["p20"] = train_group([Xf["M20"]], yf, [one], seed, [Xs["M20"]], ys, K); emit("p20", G["p20"], ["M20"])
            pair = lambda nm, mode, lam: (G.__setitem__(nm, train_group([Xf["E1"], Xf["M20"]], yf, [one, one], seed, [Xs["E1"], Xs["M20"]], ys, K, mode=mode, lam=lam)), emit(nm, G[nm], EX))
            pair("indep", "plain", 0.0)
            for lam in LAMS: pair(f"ncl{lam}", "ncl", lam)
            for lam in LAMS: pair(f"dec{lam}", "dec", lam)
            wb = (1.0 + 2.0 * miss[fit_p]).astype(np.float64)
            G["b20"] = train_group([Xf["M20"]], yf, [wb], seed, [Xs["M20"]], ys, K); emit("b20", G["b20"], ["M20"])
        if do_b:
            for tag, kd in (("kdA", None), ("kdF", S2m[fit_p])):
                for T in TS:
                    for al in ALPHAS:
                        nm = f"{tag}_T{T}_a{al}"
                        G[nm] = train_group([Xf["E1"]], yf, [one], seed, [Xs["E1"]], ys, K, mode="distill", teacher=teach[fit_p], T=float(T), alpha=al, kd_mask=kd); emit(nm, G[nm], ["E1"])
        np.savez_compressed(f_out, **R)
        print(f"{bed} s{seed} f{fold}: done {time.time()-t0:.0f}s, {len(G)} groups", flush=True)
