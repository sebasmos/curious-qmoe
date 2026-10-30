"""Check (d): score the stage 3b mn10 ESC-50/50 heads (4-bit and FP32 kinds, seeds 0-4) on the corrected real-int8
embeddings (v1: final pool in float; v2: Conv/Gemm only) and on stage 3c's (v0), with three scalers: the FP32-fit
StandardScaler (way a as scored), the same with its std floored at its 10th percentile, and a StandardScaler refit on
the variant's own fold-f clean fit rows (head weights unchanged). Static macro F1 on clean test folds and the field
mixture; FP32 and W8A8 with the FP32 scaler as references; one-sided non-inferiority (H0: int8 - W8A8 <= -1 point)
over the 5 fold means. Embedding error of each variant against torch FP32 and in the FP32 scaler's z space."""
import json, sys, contextlib, io, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
W, S3B, R, A = [Path(p) for p in sys.argv[1:5]]
sys.path.insert(0, str(W)); sys.path.insert(0, str(W / "scripts/redesign/idea6/stage2"))
from scripts.analysis.cascade import load_model  # noqa
torch.set_num_threads(2)
while not (A / "outputs/d1_int8_fixed_build_mn10.json").exists(): time.sleep(30)
FIELD_CELLS = tuple((k, s) for k in ("bg", "pink", "white") for s in (10, 5, 0))
def field_assignment(n, seed=5001):
    order = np.random.default_rng(seed).permutation(n); o = ["clean"] * n
    for r, i in enumerate(order[(n + 1) // 2:]):
        k, s = FIELD_CELLS[r % 9]; o[i] = f"{k}__{s}db"
    return np.array(o)
m, ds, task = "mn10", "esc50", "esc50"
E = S3B / "emb_cnn" / ds; ED = A / "emb_int8_fixed"
fp = np.load(E / f"{m}_fp32__clean.npy"); n = len(fp); at = field_assignment(n); conds = sorted(set(at) - {"clean"})
def field(load):
    X = load("clean").copy()
    for c in conds:
        mk = at == c; X[mk] = load(c)[mk]
    return X
emb = {"fp32": (fp, field(lambda c: np.load(E / f"{m}_fp32__{c}.npy"))), "w8a8": (np.load(E / f"{m}_w8a8__clean.npy"), field(lambda c: np.load(E / f"{m}_w8a8__{c}.npy")))}
for v in ("v0_stage3c", "v1_poolfp32", "v2_convonly"):
    emb[v] = (np.load(ED / f"{m}_{v}__clean.npy"), field(lambda c, v=v: np.load(ED / f"{m}_{v}__{c}.npy")))
# embedding errors
err = {}
mu, sd = fp.mean(0), fp.std(0)
for v, (X, _) in emb.items():
    if v == "fp32": continue
    err[v] = {"rel_l2_mean": float(np.mean(np.linalg.norm(X - fp, axis=1) / np.linalg.norm(fp, axis=1))),
              "cos_mean": float(np.mean((X * fp).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(fp, axis=1)))),
              "z_shift_max_sd": float((np.abs(X.mean(0) - mu) / (sd + 1e-12)).max()), "n_dims_shift_gt_1sd": int(((np.abs(X.mean(0) - mu) / (sd + 1e-12)) > 1).sum()),
              "dim197_n_unique": int(len(np.unique(X[:, 197]))), "dim197_rmse_over_fp32_std": float(np.sqrt(np.mean((X[:, 197] - fp[:, 197]) ** 2)) / sd[197]),
              "n_unique_values_total": int(len(np.unique(X)))}
print(json.dumps(err, indent=1), flush=True)
rows = []
for kind in ("4", "esc"):
    for s in range(5):
        run = S3B / "outputs" / f"{m}_{task}_s{s}" / kind
        cfg = OmegaConf.load(run / "config.yaml"); df = pd.read_csv(cfg.experiment.datasets.esc.csv)
        y = df[cfg.experiment.datasets.esc.get("label_col", "class_id")].to_numpy(); Xc = df[[str(i) for i in range(fp.shape[1])]].to_numpy(np.float32)
        n_cls = len(np.unique(y)); summary = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
        for fold in sorted(summary):
            split = json.loads((run / f"fold_{fold}" / "split.json").read_text()); te, iv = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
            fit = np.setdiff1d(np.arange(n), np.concatenate([te, iv]))
            with contextlib.redirect_stdout(io.StringIO()): model, _ = load_model(run, fold, fp.shape[1], n_cls)
            sc32 = StandardScaler().fit(Xc[fit]); sd32 = sc32.scale_; floor = np.percentile(sd32, 10)
            def f1(Xs):
                with torch.no_grad(): p = model(torch.tensor(Xs, dtype=torch.float32)).numpy().argmax(1)
                return f1_score(y[te], p, average="macro", zero_division=0)
            assert abs(f1((Xc[te] - sc32.mean_) / sd32) - summary[fold]["f1_macro"]) < 1e-6
            for v, (Xclean, Xfield) in emb.items():
                scr = None
                if v.startswith("v"):
                    scr = StandardScaler().fit(np.load(ED / f"{m}_{v}_f{fold}__clean.npy")[fit])
                for cond, X in (("clean", Xclean), ("field", Xfield)):
                    r = {"head": kind, "seed": s, "fold": fold, "variant": v, "condition": cond, "sc32": f1((X[te] - sc32.mean_) / sd32),
                         "sc32_floor_p10": f1((X[te] - sc32.mean_) / np.maximum(sd32, floor))}
                    if scr is not None: r["sc_refit_own"] = f1((X[te] - scr.mean_) / scr.scale_)
                    rows.append(r)
        print(kind, s, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv(A / "outputs/d2_score_fixed.csv", index=False)
T = D.groupby(["head", "condition", "variant"])[["sc32", "sc32_floor_p10", "sc_refit_own"]].mean().round(4)
print(T.to_string())
# non-inferiority within 1 point against W8A8 (sc32), one-sided fold t-test over the 5 fold means
ni = []
for (kind, cond), g in D.groupby(["head", "condition"]):
    ref = g[g.variant == "w8a8"].groupby("fold")["sc32"].mean()
    for v in ("v0_stage3c", "v1_poolfp32", "v2_convonly"):
        for scn in ("sc32", "sc32_floor_p10", "sc_refit_own"):
            x = g[g.variant == v].groupby("fold")[scn].mean(); d = (x - ref).to_numpy()
            t = (d.mean() + 0.01) / (d.std(ddof=1) / np.sqrt(len(d))); p = float(stats.t.sf(t, len(d) - 1))
            ni.append({"head": kind, "condition": cond, "variant": v, "scaler": scn, "int8_f1": float(x.mean()), "w8a8_f1": float(ref.mean()), "diff_pts": 100 * d.mean(),
                       "fold_min_pts": 100 * d.min(), "fold_max_pts": 100 * d.max(), "ni_p_one_sided": p, "holds_1pt": bool(p < 0.05)})
N = pd.DataFrame(ni); N.to_csv(A / "outputs/d2_noninferiority_fixed.csv", index=False); print(N.round(4).to_string())
json.dump(err, open(A / "outputs/d2_embedding_error_fixed.json", "w"), indent=1)
