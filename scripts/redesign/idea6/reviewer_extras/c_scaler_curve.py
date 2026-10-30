"""Reviewer extras, task C: how many clean training clips does the input-scaler refit of the real-INT8 MobileNets need?

Adapted from the stage 3c audit's a2_scaler_test.py (same heads, same embeddings, same field assignment). The head weights
(Stage 3b / 3c, trained on clean FP32 embeddings) stay unchanged. The StandardScaler is refit on n clean training clips of the
fold, embedded by the fold's real-INT8 model (emb_int8/<ds>/<model>_int8_f<fold>__clean.npy), n in 8, 16, 32, 64, 128, 256,
20 random draws without replacement from the fold's fit rows (training rows minus test and inner validation), generator
seed 20261024 + 1000 * seed + fold, one stream per (job, seed, fold). References: scaler fit on all fit rows of the
int8 embeddings ("full refit"), the FP32 scaler applied to int8 embeddings (the transplant), FP32 backbone + FP32 scaler.
Macro F1 on the fold's test clips, clean and field mixture (test rows take the seed-5001 assignment's condition).

    python c_scaler_curve.py S3B S3C OUT_CSV
"""
import contextlib, io, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

S3B, S3C, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2"))
from scripts.analysis.cascade import load_model  # noqa
import audio_shift  # noqa
torch.set_num_threads(2)
FIELD_CELLS = tuple((k, s) for k in ("bg", "pink", "white") for s in (10, 5, 0))
NS, DRAWS = [8, 16, 32, 64, 128, 256], 20

def field_assignment(n, seed=5001):
    order = np.random.default_rng(seed).permutation(n)
    o = ["clean"] * n
    for r, i in enumerate(order[(n + 1) // 2:]):
        k, s = FIELD_CELLS[r % 9]; o[i] = f"{k}__{s}db"
    return np.array(o)

JOBS = [("mn10", "esc50", "esc50", "4", [0, 1, 2, 3, 4]), ("mn20", "esc50", "esc50", "4", [0, 1, 2, 3, 4]),
        ("mn10", "urban", "urban8k", "esc", [0, 1, 2])]
avail = {(m, ds): (S3C / "emb_int8" / ds / f"{m}_int8__clean.npy").exists() for m, _, ds, _, _ in JOBS}
print("int8 embeddings present:", avail, "mn20 urban8k:", (S3C / "emb_int8/urban8k/mn20_int8__clean.npy").exists(), flush=True)
rows = []
for m, task, ds, kind, seeds in JOBS:
    E = S3B if m in ("mn04", "mn10") else S3C
    fp = np.load(E / "emb_cnn" / ds / f"{m}_fp32__clean.npy")
    i8 = np.load(S3C / "emb_int8" / ds / f"{m}_int8__clean.npy")
    n = len(fp); at = field_assignment(n); assert (at == np.array(audio_shift.field_assignment(n, 5001))).all()
    conds = sorted(set(at) - {"clean"})
    fpc = {c: np.load(E / "emb_cnn" / ds / f"{m}_fp32__{c}.npy") for c in conds}
    i8c = {c: np.load(S3C / "emb_int8" / ds / f"{m}_int8__{c}.npy") for c in conds}
    def field(X_clean, Xc):
        Xf = X_clean.copy()
        for c in conds:
            mk = at == c; Xf[mk] = Xc[c][mk]
        return Xf
    I8 = {"clean": i8, "field": field(i8, i8c)}
    FP = {"clean": fp, "field": field(fp, fpc)}
    for s in seeds:
        run = E / "outputs" / f"{m}_{task}_s{s}" / kind
        cfg = OmegaConf.load(run / "config.yaml")
        df = pd.read_csv(cfg.experiment.datasets.esc.csv); y = df[cfg.experiment.datasets.esc.get("label_col", "class_id")].to_numpy()
        Xc = df[[str(i) for i in range(fp.shape[1])]].to_numpy(np.float32)
        assert np.allclose(Xc, fp, atol=1e-5)
        n_cls = len(np.unique(y))
        summary = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
        for fold in sorted(summary):
            split = json.loads((run / f"fold_{fold}" / "split.json").read_text())
            te, iv = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
            fit = np.setdiff1d(np.arange(n), np.concatenate([te, iv]))
            with contextlib.redirect_stdout(io.StringIO()):
                model, _ = load_model(run, fold, fp.shape[1], n_cls)
            i8f = np.load(S3C / "emb_int8" / ds / f"{m}_int8_f{fold}__clean.npy")
            sc32 = StandardScaler().fit(Xc[fit]); scfull = StandardScaler().fit(i8f[fit])
            def f1(sc, X):
                with torch.no_grad():
                    p = model(torch.tensor((X[te] - sc.mean_) / sc.scale_, dtype=torch.float32)).numpy().argmax(1)
                return f1_score(y[te], p, average="macro", zero_division=0)
            assert abs(f1(sc32, FP["clean"]) - summary[fold]["f1_macro"]) < 1e-6
            rng = np.random.default_rng(20261024 + 1000 * s + fold)
            for cond in ("clean", "field"):
                base = dict(model=m, task=task, head=kind, seed=s, fold=fold, condition=cond)
                rows.append({**base, "variant": "fp32_backbone_fp32_scaler", "n": 0, "draw": 0, "f1": f1(sc32, FP[cond])})
                rows.append({**base, "variant": "int8_fp32_scaler", "n": 0, "draw": 0, "f1": f1(sc32, I8[cond])})
                rows.append({**base, "variant": "int8_full_refit", "n": len(fit), "draw": 0, "f1": f1(scfull, I8[cond])})
            for nn in NS:
                for d in range(DRAWS):
                    sub = rng.choice(fit, nn, replace=False)
                    sc = StandardScaler().fit(i8f[sub])
                    for cond in ("clean", "field"):
                        rows.append(dict(model=m, task=task, head=kind, seed=s, fold=fold, condition=cond,
                                         variant="int8_subset_refit", n=nn, draw=d, f1=f1(sc, I8[cond])))
        print(m, task, "seed", s, "done", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
pd.DataFrame(rows).to_csv(out, index=False)
