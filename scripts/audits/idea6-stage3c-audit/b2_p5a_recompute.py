"""Finding 2 (a) and (d): UrbanSound8K mn10 padded against exact-length static macro F1 per (seed, fold), clean and
field mixture, from the score files: stage 3b heads on padded embeddings (S3B/scores_cnn), the same heads on the
exact-length embeddings (R/scores_unp), and heads retrained on exact-length FP32 embeddings (R/scores_unpr, FP32 head
only). Field assignment re-implemented (seed 5001 permutation, ceil(n/2) clean, the rest over bg/pink/white x 10/5/0)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
S3B = Path(sys.argv[1]); R = Path(sys.argv[2]); out = Path(sys.argv[3])
FIELD_CELLS = tuple((k, s) for k in ("bg", "pink", "white") for s in (10, 5, 0))

def field_assignment(n, seed=5001):
    order = np.random.default_rng(seed).permutation(n)
    o = ["clean"] * n
    for r, i in enumerate(order[(n + 1) // 2:]):
        k, s = FIELD_CELLS[r % 9]; o[i] = f"{k}__{s}db"
    return np.array(o)

def f1_macro(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float); den = cm.sum(0) + cm.sum(1); ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))

def static_f1(d, cfg, K=10):
    z = np.load(d / f"{cfg}__clean.npz"); y, fold = z["y"].astype(int), z["test_fold"].astype(int)
    at = field_assignment(len(y))
    zf = z["test_logits"].copy()
    for c in sorted(set(at) - {"clean"}):
        zc = np.load(d / f"{cfg}__{c}.npz"); assert (zc["y"] == y).all() and (zc["test_fold"] == fold).all()
        mk = at == c; zf[mk] = zc["test_logits"][mk]
    assert np.isfinite(zf).all() and np.isfinite(z["test_logits"]).all()
    res = {}
    for f in np.unique(fold):
        te = fold == f
        res[("clean", int(f))] = f1_macro(y[te], z["test_logits"][te].argmax(1), K)
        res[("field", int(f))] = f1_macro(y[te], zf[te].argmax(1), K)
    return res, y, fold

rows = []
for bb in ("w8a8", "fp32"):
    for k in ("esc", "4", "8", "bitnet"):
        for s in (0, 1, 2):
            pad, y0, f0 = static_f1(S3B / "scores_cnn/urban" / f"mn10_urban_s{s}" / k, f"mn10_{bb}")
            unp, y1, f1_ = static_f1(R / "scores_unp/urban" / f"mn10_urban_s{s}" / k, f"mn10_{bb}")
            assert (y0 == y1).all() and (f0 == f1_).all()
            unr = None
            if k == "esc":
                unr, y2, f2 = static_f1(R / "scores_unpr/urban" / f"mn10unp_urban_s{s}" / k, f"mn10_{bb}")
                assert (y0 == y2).all() and (f0 == f2).all()
            for (c, f), v in pad.items():
                rows.append({"bb": bb, "head": k, "seed": s, "fold": f, "condition": c, "padded": v, "unpadded_same_heads": unp[(c, f)],
                             "unpadded_retrained_heads": unr[(c, f)] if unr else np.nan})
D = pd.DataFrame(rows); D.to_csv(out.with_suffix(".rows.csv"), index=False)
summ = []
for (bb, k, c), g in D.groupby(["bb", "head", "condition"]):
    fm = g.groupby("fold")[["padded", "unpadded_same_heads", "unpadded_retrained_heads"]].mean()
    d1 = fm.unpadded_same_heads - fm.padded
    r = {"bb": bb, "head": k, "condition": c, "padded": fm.padded.mean(), "unpadded_same_heads": fm.unpadded_same_heads.mean(),
         "diff_same_heads_pts": 100 * d1.mean(), "p_t_fold_same": stats.ttest_1samp(d1, 0).pvalue, "fold_min_pts": 100 * d1.min(), "fold_max_pts": 100 * d1.max()}
    if k == "esc":
        d2 = fm.unpadded_retrained_heads - fm.padded
        r.update({"unpadded_retrained_heads": fm.unpadded_retrained_heads.mean(), "diff_retrained_pts": 100 * d2.mean(),
                  "p_t_fold_retrained": stats.ttest_1samp(d2, 0).pvalue, "retrained_fold_min_pts": 100 * d2.min(), "retrained_fold_max_pts": 100 * d2.max()})
    summ.append(r)
S = pd.DataFrame(summ); S.to_csv(out, index=False)
print(S.round(4).to_string())
