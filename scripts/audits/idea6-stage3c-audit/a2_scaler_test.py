"""Check (a), second part: is the way-a collapse a scaler problem? Stage 3b head weights unchanged, scored on the
stage 3c real-int8 embeddings with (i) the FP32-fit StandardScaler as in score_int8.py, (ii) a StandardScaler refit on
the fold's int8 clean fit rows (training rows minus test and inner validation), (iii) the FP32 scaler with its std
floored at the 10th percentile of the FP32 stds, (iv) the FP32 scaler with dims of FP32 std < 1e-2 frozen at their
mean. W8A8 and FP32 with the FP32 scaler as references. Static macro F1 on clean test folds and on the field mixture
(test rows take the seed-5001 assignment's condition). Field assignment re-implemented here and asserted equal."""
import json, sys, contextlib, io
from pathlib import Path
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
W = Path(sys.argv[1]); S3B = Path(sys.argv[2]); R = Path(sys.argv[3]); out = Path(sys.argv[4])
sys.path.insert(0, str(W)); sys.path.insert(0, str(W / "scripts/redesign/idea6/stage2"))
from scripts.analysis.cascade import load_model  # noqa
import audio_shift  # noqa
torch.set_num_threads(2)
FIELD_CELLS = tuple((k, s) for k in ("bg", "pink", "white") for s in (10, 5, 0))

def field_assignment(n, seed=5001):
    order = np.random.default_rng(seed).permutation(n)
    o = ["clean"] * n
    for r, i in enumerate(order[(n + 1) // 2:]):
        k, s = FIELD_CELLS[r % 9]; o[i] = f"{k}__{s}db"
    return np.array(o)

JOBS = [("mn10", "esc50", "esc50", "4", [0, 1, 2, 3, 4]), ("mn20", "esc50", "esc50", "4", [0, 1, 2, 3, 4]),
        ("mn10", "urban", "urban8k", "esc", [0, 1, 2])]
rows = []
for m, task, ds, kind, seeds in JOBS:
    E = S3B if m in ("mn04", "mn10") else R
    fp = np.load(E / "emb_cnn" / ds / f"{m}_fp32__clean.npy"); w8 = np.load(E / "emb_cnn" / ds / f"{m}_w8a8__clean.npy")
    i8 = np.load(R / "emb_int8" / ds / f"{m}_int8__clean.npy")
    n = len(fp); at = field_assignment(n); assert (at == np.array(audio_shift.field_assignment(n, 5001))).all()
    conds = sorted(set(at) - {"clean"})
    fpc = {c: np.load(E / "emb_cnn" / ds / f"{m}_fp32__{c}.npy") for c in conds}
    w8c = {c: np.load(E / "emb_cnn" / ds / f"{m}_w8a8__{c}.npy") for c in conds}
    i8c = {c: np.load(R / "emb_int8" / ds / f"{m}_int8__{c}.npy") for c in conds}
    def field(X_clean, Xc):
        Xf = X_clean.copy()
        for c in conds:
            mk = at == c; Xf[mk] = Xc[c][mk]
        return Xf
    fp_f, w8_f, i8_f = field(fp, fpc), field(w8, w8c), field(i8, i8c)
    for s in seeds:
        run = (S3B if m in ("mn04", "mn10") else R) / "outputs" / f"{m}_{task}_s{s}" / kind
        cfg = OmegaConf.load(run / "config.yaml")
        df = pd.read_csv(cfg.experiment.datasets.esc.csv); y = df[cfg.experiment.datasets.esc.get("label_col", "class_id")].to_numpy()
        Xc = df[[str(i) for i in range(fp.shape[1])]].to_numpy(np.float32)
        assert np.allclose(Xc, fp, atol=1e-5), "train CSV is not the clean FP32 embedding"
        n_cls = len(np.unique(y))
        summary = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
        for fold in sorted(summary):
            split = json.loads((run / f"fold_{fold}" / "split.json").read_text())
            te, iv = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
            fit = np.setdiff1d(np.arange(n), np.concatenate([te, iv]))
            with contextlib.redirect_stdout(io.StringIO()):
                model, _ = load_model(run, fold, fp.shape[1], n_cls)
            sc32 = StandardScaler().fit(Xc[fit])
            i8f = np.load(R / "emb_int8" / ds / f"{m}_int8_f{fold}__clean.npy")
            sc8 = StandardScaler().fit(i8f[fit])
            sd = np.sqrt(sc32.var_); floor = np.percentile(sd, 10)
            low = sd < 1e-2
            def f1(model, Xs):
                with torch.no_grad():
                    p = model(torch.tensor(Xs, dtype=torch.float32)).numpy().argmax(1)
                return f1_score(y[te], p, average="macro", zero_division=0)
            def z32(X): return (X[te] - sc32.mean_) / sd
            def zfloor(X): return (X[te] - sc32.mean_) / np.maximum(sd, floor)
            def zmask(X):
                z = z32(X); z[:, low] = 0.0; return z
            def z8(X): return (X[te] - sc8.mean_) / sc8.scale_
            f1_ref = f1(model, z32(fp))
            assert abs(f1_ref - summary[fold]["f1_macro"]) < 1e-6, (run, fold, f1_ref, summary[fold]["f1_macro"])
            for cond, (F, W8, I8) in {"clean": (fp, w8, i8), "field": (fp_f, w8_f, i8_f)}.items():
                rows.append({"model": m, "task": task, "head": kind, "seed": s, "fold": fold, "condition": cond,
                             "fp32_sc32": f1(model, z32(F)), "w8a8_sc32": f1(model, z32(W8)),
                             "int8_sc32": f1(model, z32(I8)), "int8_sc_refit_int8": f1(model, z8(I8)),
                             "int8_sc32_floor_p10": f1(model, zfloor(I8)), "int8_sc32_mask_lowstd": f1(model, zmask(I8)),
                             "w8a8_sc32_floor_p10": f1(model, zfloor(W8)), "n_low_std_dims": int(low.sum()), "floor": float(floor)})
            print(m, task, s, fold, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
D = pd.DataFrame(rows); D.to_csv(out, index=False)
cols = [c for c in D.columns if c.endswith(("sc32", "int8", "p10", "lowstd"))]
print(D.groupby(["model", "task", "head", "condition"])[cols].mean().round(4).to_string())
