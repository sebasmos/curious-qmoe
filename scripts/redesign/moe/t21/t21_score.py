"""T21 scores: logits of every FP32 head on every deployed tier and condition, under both scalers (ANALYSIS_PLAN.md C1).

Heads: <R>/outputs/<d>_<design>_<head>_s<seed>/esc (benchmark.py runs on the clean FP32 CSVs). Before scoring, every
fold's head is re-run on the FP32 CSV's test rows with the FP32-fit scaler and must reproduce summary.json.
Scalers: fp32fit = StandardScaler on the FP32 clean fit rows (score_eff.fold_scaler); refit = QWave.deploy.refit_scaler on
the deployed tier's clean fit rows (for mn10_int8 the fold's own int8 model).
Field mix: test rows read the cell of the test assignment (seed 5001), inner-validation rows that of the validation
assignment (seed 5002), both over the pool rows.

Output <R>/scores/<d>/<design>_<tier>_s<seed>.npz: y, fold (outer fold of each row, -1 if never test), for each
condition c and scaler k: test__<c>__<k> [N, C]; iv_idx_<f>; iv__<f>__<c>__<k> [len(iv), C].

    python t21_score.py --r R --d d1 [--designs pool official] [--tiers mn10_w8a8 mn10_int8 ast_w4g128]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT, ROOT / "scripts/redesign/idea6/stage2"):
    sys.path.insert(0, str(p))
from audio_shift import FIELD_SEEDS, field_assignment  # noqa: E402
from score_eff import fold_scaler, logits_of  # noqa: E402
from scripts.analysis.cascade import load_model  # noqa: E402
from QWave.deploy import refit_scaler  # noqa: E402

NOISE = ["bg__10db", "bg__0db", "pink__10db", "pink__0db", "white__10db", "white__0db"]
CELLS = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]
TIERS = {"mn10_w8a8": ("mn10", "mn_pool", "mn_extra", "mn10_w8a8__{}.npy"),
         "mn10_int8": ("mn10", "mn_int8", None, "mn10_int8__{}.npy"),
         "ast_w4g128": ("ast", "ast_pool", "ast_extra", "w4g128__{}.npy")}


def pool_emb(r, d, sub, pat, cond):
    f = r / "emb" / d / sub / pat.format(cond)
    return np.load(f) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--d", default="d1")
    ap.add_argument("--designs", nargs="+", default=["pool", "official"])
    ap.add_argument("--tiers", nargs="+", default=list(TIERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    torch.set_num_threads(2)
    R, D = a.r, a.d
    pool = pd.read_csv(R / "csv" / D / "pool.csv")
    npool = len(pool)
    at, av = (np.array(field_assignment(npool, FIELD_SEEDS[k])) for k in ("test", "val"))
    out_dir = R / "scores" / D
    out_dir.mkdir(parents=True, exist_ok=True)
    for tier in a.tiers:
        head, psub, esub, pat = TIERS[tier]
        P = {c: pool_emb(R, D, psub, pat, c) for c in ["clean", *CELLS]}
        missing = [c for c, v in P.items() if v is None]
        if missing:
            raise SystemExit(f"{tier}: missing {missing}")
        # field composites over pool rows
        ft = np.full_like(P["clean"], np.nan)
        fv = np.full_like(P["clean"], np.nan)
        for c in ["clean", *CELLS]:
            ft[at == c] = P[c][at == c]
            fv[av == c] = P[c][av == c]
        conds = {"clean": P["clean"], **{c: P[c] for c in NOISE}, "field": ft}
        for design in a.designs:
            if design == "official" and tier == "mn10_int8":
                continue
            if design == "official":
                Xe = np.load(R / "emb" / D / esub / pat.format("clean"))
                ne = len(Xe)

                def ext(X, clean=False):
                    return np.vstack([X, Xe if clean else np.full((ne, X.shape[1]), np.nan, np.float32)])
            else:
                def ext(X, clean=False):
                    return X
            for s in a.seeds:
                run = R / "outputs" / f"{D}_{design}_{head}_s{s}" / "esc"
                of = out_dir / f"{design}_{tier}_s{s}.npz"
                if of.exists():
                    continue
                cfg = OmegaConf.load(run / "config.yaml")
                tr = pd.read_csv(cfg.experiment.datasets.esc.csv)
                dim = sum(c.isdigit() for c in tr.columns)
                Xc = tr[[str(i) for i in range(dim)]].to_numpy(np.float32)
                y = tr["class_id"].to_numpy()
                n_cls = len(np.unique(y))
                summ = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
                res = {"y": y, "fold": np.full(len(y), -1, np.int16)}
                for fold in sorted(summ):
                    sp = json.loads((run / f"fold_{fold}" / "split.json").read_text())
                    te, iv = np.asarray(sp["test_idx"]), np.asarray(sp["inner_val_idx"])
                    model, _ = load_model(run, fold, dim, n_cls)
                    sc32 = fold_scaler(Xc, y, sp, "standard")
                    f1 = f1_score(y[te], logits_of(model, sc32, Xc[te], n_cls).argmax(1), average="macro", zero_division=0)
                    if abs(f1 - summ[fold]["f1_macro"]) > 1e-6:
                        raise SystemExit(f"{run} fold {fold}: rescored {f1} != {summ[fold]['f1_macro']}")
                    fit = np.setdiff1d(np.arange(len(y)), np.concatenate([te, iv]))
                    if tier == "mn10_int8":  # the fold's own int8 model on every clean row
                        Xfit_dep = np.load(R / "emb" / D / "mn_int8" / f"mn10_int8_f{fold}__clean.npy")
                    else:
                        Xfit_dep = ext(conds["clean"], clean=True)
                    if not np.isfinite(Xfit_dep[fit]).all():
                        raise SystemExit(f"{tier} fold {fold}: deployed fit rows missing")
                    with contextlib.redirect_stdout(io.StringIO()):
                        scr, _ = refit_scaler(Xfit_dep[fit], model)
                    res["fold"][te] = fold
                    res[f"iv_idx_{fold}"] = iv
                    for k, sc in (("fp32fit", sc32), ("refit", scr)):
                        for c, X in conds.items():
                            Xd = ext(X, clean=(c == "clean"))
                            key = f"test__{c}__{k}"
                            if key not in res:
                                res[key] = np.full((len(y), n_cls), np.nan, np.float32)
                            res[key][te] = logits_of(model, sc, Xd[te], n_cls)
                        res[f"iv__{fold}__clean__{k}"] = logits_of(model, sc, Xfit_dep[iv], n_cls)
                        if tier != "mn10_int8":
                            res[f"iv__{fold}__field__{k}"] = logits_of(model, sc, ext(fv)[iv], n_cls)
                np.savez_compressed(of, **res)
                print(f"{of.name}: scored, F1 equal to summary.json", flush=True)


if __name__ == "__main__":
    main()
