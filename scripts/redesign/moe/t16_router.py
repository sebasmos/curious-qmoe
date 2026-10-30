"""T16 step 2 (CPU): the hidden-state routers. Per (seed, fold) and per input kind (d4, d6, d8, d10, d12 = block-b features + E1 logits and margin; out = E1 logits and margin only):
target = E1 wrong and S right on the clean out-of-fold logits of the training clips; MLP (one hidden layer, ReLU, dropout 0.2, AdamW 1e-3, BCE with a positive weight) fitted on fit_p,
standardised on fit_p, configuration and epoch chosen by the AUROC on the clean inner validation iv_p. Scores for every item of every evaluation condition (test rows and held-out rows).
Reads only y of the training clips. python t16_router.py --bed B --heads DIR --out DIR [--seeds ..] [--folds ..]"""
import argparse, copy, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from t16_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--bed", required=True); ap.add_argument("--heads", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS); ap.add_argument("--folds", type=int, nargs="+", default=None); ap.add_argument("--variant", default="fp32"); ap.add_argument("--conds", nargs="*", default=None); ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args(); torch.set_num_threads(a.threads)
bed = a.bed; K = BEDS[bed]["ncls"]; D = load_bed(bed); y = D["y"]; CONDS = a.conds or list(EVAL[bed]); KINDS = [f"d{d}" for d in DEPTHS] + ["out"] + [f"p{d}" for d in DEPTHS]       # p<d>: Amendment 2, hidden features reduced to 64 PCA scores (fit on fit_p)
folds = a.folds or list(range(1, BEDS[bed]["nf"] + 1)); (a.out / bed).mkdir(parents=True, exist_ok=True)
FD = FEAT if a.variant == "fp32" else FEAT.parent / "feat_w4g128"; fpath = lambda c: FD / f"{bed}__{c}.npy"; FC = np.load(fpath("clean"), mmap_mode="r")
STRONG = "S" if a.variant == "fp32" else ("E2b" if bed == "tau" else "E2c")
GRID = [(h, wd, pw) for h in (32, 128) for wd in (1e-4, 1e-2) for pw in ("1", "sqrt", "full")]
MAXEP, PAT = 40, 6


def outs(z): p = softmax(z.astype(np.float64)); return np.hstack([z, pmargin(p)[:, None]]).astype(np.float32)


def train(Xf, yf, Xv, yv, h, wd, pw, seed):
    torch.manual_seed(seed); r = (yf == 0).sum() / max((yf == 1).sum(), 1); w = {"1": 1.0, "sqrt": float(np.sqrt(r)), "full": float(r)}[pw]
    net = nn.Sequential(nn.Linear(Xf.shape[1], h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 1)); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w)); X, Y = torch.tensor(Xf), torch.tensor(yf, dtype=torch.float32); Xvt = torch.tensor(Xv)
    best, bs, bad, gen = -1.0, None, 0, torch.Generator().manual_seed(seed)
    for ep in range(MAXEP):
        net.train(); perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]; l = lossf(net(X[b])[:, 0], Y[b]); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): auc = roc_auc_score(yv, net(Xvt)[:, 0].numpy()) if 0 < yv.sum() < len(yv) else 0.5     # held-out rows with one class only (ESC-50 seed 1 fold 5): constant 0.5, first configuration, 7 epochs
        if auc > best + 1e-5: best, bs, bad = auc, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= PAT: break
    net.load_state_dict(bs); net.eval(); return net, best


for seed in a.seeds:
    for fold in folds:
        fo = a.out / bed / f"s{seed}_f{fold}.npz"
        if fo.exists(): continue
        t0 = time.time(); Hd = np.load(a.heads / bed / f"s{seed}_f{fold}.npz"); tr, fp, ip = Hd["tr"], Hd["fit_p"], Hd["iv_p"]; ytr = y[tr]
        zs_oof = Hd["S_oof"] if a.variant == "fp32" else np.load(T1C / bed / f"raw_oof/s{seed}_f{fold}.npz")[f"{STRONG}_oof"]
        assert (np.load(T1C / bed / f"raw_oof/s{seed}_f{fold}.npz")["tr"] == tr).all() if a.variant != "fp32" else True
        yr = ((Hd["E1_oof"].argmax(1) != ytr) & (zs_oof.argmax(1) == ytr)).astype(np.int64)
        O_tr = outs(Hd["E1_oof"]); Ftr = np.asarray(FC[tr]).reshape(len(tr), len(DEPTHS), -1)
        R = dict(pos_rate_train=yr.mean(), pos_rate_iv=yr[ip].mean())
        # item features
        IT = {}
        for cn in CONDS:
            for part in ("te", "iv"):
                rows, cells = Hd[f"{cn}|{part}_rows"], Hd[f"{cn}|{part}_cells"]; IT[cn, part] = (take(fpath, rows, cells).reshape(len(rows), len(DEPTHS), -1), outs(Hd[f"L|E1|{cn}|{part}"]))
        for kind in KINDS:
            dd = DEPTHS.index(int(kind[1:])) if kind != "out" else None; pca = None
            if kind.startswith("p"):
                hs = StandardScaler().fit(Ftr[fp][:, dd]); pca = (hs, PCA(64, random_state=seed).fit(hs.transform(Ftr[fp][:, dd])))
            def mk(F, O):
                if kind == "out": return O
                if pca is None: return np.hstack([F[:, dd], O])
                return np.hstack([pca[1].transform(pca[0].transform(F[:, dd])), O])
            Xtr = mk(Ftr, O_tr); sc = StandardScaler().fit(Xtr[fp]); Xf, Xv = sc.transform(Xtr[fp]).astype(np.float32), sc.transform(Xtr[ip]).astype(np.float32)
            best = (-1, None, None)
            for g in GRID:
                net, auc = train(Xf, yr[fp], Xv, yr[ip], *g, seed * 1000 + fold)
                if auc > best[0]: best = (auc, net, g)
            net = best[1]; R[f"iv_auc|{kind}"] = best[0]; R[f"cfg|{kind}"] = np.array([best[2][0], best[2][1], best[2][2] == "1" and 0 or (best[2][2] == "sqrt" and 1 or 2)])
            for (cn, part), (F, O) in IT.items():
                X = sc.transform(mk(F, O)).astype(np.float32); ok = np.isfinite(X).all(1); s = np.full(len(X), np.nan, np.float32)
                with torch.no_grad(): s[ok] = net(torch.tensor(X[ok]))[:, 0].numpy()
                R[f"sc|{kind}|{cn}|{part}"] = s
            print(f"  {bed} s{seed} f{fold} {kind}: iv auc {best[0]:.3f} cfg {best[2]}", flush=True)
        np.savez_compressed(fo, **R); print(f"{bed} s{seed} f{fold}: router done {time.time() - t0:.0f}s", flush=True)
