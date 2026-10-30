"""T24 scores (ANALYSIS_PLAN.md C1, C6): logits of every FP32 head on every deployed tier and condition, under both
scalers, for the fixed-test design (five fold models, one official test set).

Changes from t21_score.py: (1) every fold model scores the same test rows, so test logits are stored per fold;
(2) the fit rows of fold k are split.json's `fit_idx` (benchmark.py fixed_test), which leaves out fold k's noisy rows,
for both the FP32-fit scaler and the refit scaler; (3) clean embeddings come from the all-clips extraction and the
noise cells from the eval-rows extraction (verified train rows and test rows, eval.csv order).
Field mix: over eval.csv rows, test rows read the cell of the test assignment (seed 5001), inner-validation rows that
of the validation assignment (seed 5002), as T21.

Output <R>/scores/<d>/<tier>_s<seed>.npz: y [N_all], test_idx [n_test] (all.csv rows), for each fold f: iv_idx_<f>,
n_fit_<f>, test__<f>__<c>__<k> [n_test, C], iv__<f>__clean__<k>, iv__<f>__field__<k>.

    python t24_score.py --r R --d d3 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import zlib
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
from score_eff import logits_of  # noqa: E402
from scripts.analysis.cascade import load_model  # noqa: E402
from QWave.datasets import EmbeddingAdaptDataset  # noqa: E402
from QWave.deploy import refit_scaler  # noqa: E402

NOISE = ["bg__10db", "bg__0db", "pink__10db", "pink__0db", "white__10db", "white__0db"]
CELLS = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]
TIERS = {"mn10_w8a8": ("mn10", "mn_all", "mn_eval", "mn10_w8a8__{}.npy"),
         "ast_w4g128": ("ast", "ast_all", "ast_eval", "w4g128__{}.npy")}


def load_emb(f, meta):
    X = np.load(f)
    side = json.loads(f.with_suffix(".json").read_text())
    crc = zlib.crc32("\n".join(meta["name"].map(lambda s: Path(s).stem)).encode())
    if side["names_crc32"] != crc or len(X) != len(meta):
        raise SystemExit(f"{f}: rows do not match its CSV")
    return X


def fit_scaler(X, y, fit):
    """The scaler benchmark.py fits for the fold: EmbeddingAdaptDataset('standard') on the fit rows."""
    df = pd.DataFrame(X[fit], columns=[str(i) for i in range(X.shape[1])])
    df["class_id"] = y[fit]
    with contextlib.redirect_stdout(io.StringIO()):
        return EmbeddingAdaptDataset(df, normalization_type="standard").get_scaler()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--d", required=True)
    ap.add_argument("--tiers", nargs="+", default=list(TIERS))
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    R, D = a.r, a.d
    al = pd.read_csv(R / "csv" / D / "all.csv")
    ev = pd.read_csv(R / "csv" / D / "eval.csv")
    pos = {n: i for i, n in enumerate(al.name)}
    ev2all = np.array([pos[n] for n in ev.name])
    all2ev = np.full(len(al), -1)
    all2ev[ev2all] = np.arange(len(ev))
    test_all = np.flatnonzero(al.folder.to_numpy() == 0)
    if (all2ev[test_all] < 0).any():
        raise SystemExit("a test row is missing from eval.csv")
    n_ev = len(ev)
    at, av = (np.array(field_assignment(n_ev, FIELD_SEEDS[k])) for k in ("test", "val"))
    out_dir = R / "scores" / D
    out_dir.mkdir(parents=True, exist_ok=True)
    for tier in a.tiers:
        head, asub, esub, pat = TIERS[tier]
        Xall = load_emb(R / "emb" / D / asub / pat.format("clean"), al)
        if not np.isfinite(Xall).all():
            raise SystemExit(f"{tier}: non-finite clean rows")
        P = {"clean": Xall[ev2all]}
        for c in CELLS:
            P[c] = load_emb(R / "emb" / D / esub / pat.format(c), ev)
        ft = np.full_like(P["clean"], np.nan)
        fv = np.full_like(P["clean"], np.nan)
        for c in ["clean", *CELLS]:
            ft[at == c] = P[c][at == c]
            fv[av == c] = P[c][av == c]
        conds = {"clean": P["clean"], **{c: P[c] for c in NOISE}, "field": ft}
        te_ev = all2ev[test_all]
        for s in a.seeds:
            run = R / "outputs" / f"{D}_{head}_s{s}" / "esc"
            of = out_dir / f"{tier}_s{s}.npz"
            if of.exists():
                continue
            cfg = OmegaConf.load(run / "config.yaml")
            tr = pd.read_csv(cfg.experiment.datasets.esc.csv)
            if not (tr.name.to_numpy() == al.name.to_numpy()).all():
                raise SystemExit(f"{run}: head CSV rows differ from all.csv")
            dim = sum(c.isdigit() for c in tr.columns)
            Xc = tr[[str(i) for i in range(dim)]].to_numpy(np.float32)
            y = tr["class_id"].to_numpy()
            n_cls = len(np.unique(y))
            summ = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
            res = {"y": y, "test_idx": test_all}
            for fold in sorted(summ):
                sp = json.loads((run / f"fold_{fold}" / "split.json").read_text())
                te, iv, fit = (np.asarray(sp[k]) for k in ("test_idx", "inner_val_idx", "fit_idx"))
                assert sp.get("fixed_test") and np.array_equal(np.sort(te), test_all), "test rows are not the official test set"
                assert not (set(iv) & set(te)) and not (set(fit) & set(te)) and not (set(fit) & set(iv)), "split overlap"
                assert (al.folder.to_numpy()[iv] == fold).all() and (al.role.to_numpy()[iv] == "v").all(), "iv rows outside fold or noisy"
                assert (al.folder.to_numpy()[fit] != fold).all() and (al.folder.to_numpy()[fit] > 0).all(), "fit rows inside fold or test"
                if (all2ev[iv] < 0).any():
                    raise SystemExit("an inner-validation row is missing from eval.csv")
                te = test_all  # sorted, the order of every stored test matrix
                model, _ = load_model(run, fold, dim, n_cls)
                sc32 = fit_scaler(Xc, y, fit)
                f1 = f1_score(y[te], logits_of(model, sc32, Xc[te], n_cls).argmax(1), average="macro", zero_division=0)
                if abs(f1 - summ[fold]["f1_macro"]) > 1e-6:
                    raise SystemExit(f"{run} fold {fold}: rescored {f1} != {summ[fold]['f1_macro']}")
                with contextlib.redirect_stdout(io.StringIO()):
                    scr, _ = refit_scaler(Xall[fit], model)
                res[f"iv_idx_{fold}"] = iv
                res[f"n_fit_{fold}"] = np.int64(len(fit))
                for k, sc in (("fp32fit", sc32), ("refit", scr)):
                    for c, X in conds.items():
                        res[f"test__{fold}__{c}__{k}"] = logits_of(model, sc, X[te_ev], n_cls)
                    res[f"iv__{fold}__clean__{k}"] = logits_of(model, sc, Xall[iv], n_cls)
                    res[f"iv__{fold}__field__{k}"] = logits_of(model, sc, fv[all2ev[iv]], n_cls)
            np.savez_compressed(of, **res)
            print(f"{of.name}: scored, F1 equal to summary.json", flush=True)


if __name__ == "__main__":
    main()
