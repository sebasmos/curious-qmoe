"""Score the cheap-tier heads on every EfficientNet embedding file (both
backbone configurations, every condition), or with --dim 768 --configs ...
the short-input AST heads on the short AST files.

Adapted from idea 5 Stage 2's score_stage2.py (branch redesign/idea5-stage2):
heads come from scripts/benchmark.py run directories (<outputs>/<tag>/<kind>),
trained on the clean FP32 rebuilt CSV. For each fold the scaler is fitted on
the fold's fit portion (training folds minus the inner validation split) of
that CSV and applied unchanged to every file, as a deployed head sees a
quantized or shifted backbone. Before scoring, every fold's head is re-run on
the training CSV's test rows: its macro F1 must equal the fold's F1 in
summary.json.

Per (run, file <config>__<condition>.npy) this writes
<out>/<tag>/<kind>/<config>__<condition>.npz:
  y, test_fold [N]             labels and the fold that holds each clip out
  test_logits [N, C]           each clip scored by the head of its test fold
  iv_idx_<k>, iv_logits_<k>    fold k's inner validation rows and their logits
  names_crc32                  of the embedding file's sidecar

    python scripts/redesign/idea6/stage2/score_eff.py --runs OUT/outputs/effp_esc50_s0/esc ... \
        --emb-dir EMB/esc50 --out SCORES/esc50 [--only 'fp32__*']
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.analysis.cascade import load_model  # noqa: E402

def train_frame(cfg, dim):
    esc = cfg.experiment.datasets.esc
    df = pd.read_csv(esc.csv)
    y = df[esc.get("label_col", "class_id")].to_numpy()
    return df[[str(i) for i in range(dim)]].to_numpy(np.float32), y


def fold_scaler(X, y, split, norm):
    from QWave.datasets import EmbeddingAdaptDataset

    fit = np.setdiff1d(np.arange(len(X)), np.concatenate([split["test_idx"], split["inner_val_idx"]]))
    df = pd.DataFrame(X[fit], columns=[str(i) for i in range(X.shape[1])])
    df["class_id"] = y[fit]
    with contextlib.redirect_stdout(io.StringIO()):
        return EmbeddingAdaptDataset(df, normalization_type=norm).get_scaler()


@torch.no_grad()
def logits_of(model, scaler, X, n_cls):
    out = np.full((len(X), n_cls), np.nan, np.float32)
    ok = np.isfinite(X).all(1)
    if ok.any():
        out[ok] = model(torch.tensor(scaler.transform(X[ok]), dtype=torch.float32)).numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--emb-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=["*"], help="glob(s) over embedding file stems")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--configs", nargs="+", default=["fp32", "w8a8"],
                    help="configuration prefixes of the files ('*': every <config>__<condition>.npy)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    files = sorted(p for p in args.emb_dir.glob("*.npy")
                   if any(fnmatch.fnmatch(p.stem, g) for g in args.only)
                   and "__" in p.stem and not p.stem.startswith(("_", "audiostat", "lengths"))
                   and ("*" in args.configs or p.stem.split("__")[0] in args.configs))
    for run in args.runs:
        run = run.resolve()
        cfg = OmegaConf.load(run / "config.yaml")
        Xc, y = train_frame(cfg, args.dim)
        n_cls = len(np.unique(y))
        norm = cfg.experiment.datasets.esc.normalization_type
        out_dir = args.out / run.parent.name / run.name
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [p for p in files if not (out_dir / f"{p.stem}.npz").exists()]
        if not todo:
            continue
        summary = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
        heads = {}
        for fold in sorted(summary):
            split = json.loads((run / f"fold_{fold}" / "split.json").read_text())
            model, _ = load_model(run, fold, Xc.shape[1], n_cls)
            te, iv = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
            scaler = fold_scaler(Xc, y, split, norm)
            f1 = f1_score(y[te], logits_of(model, scaler, Xc[te], n_cls).argmax(1), average="macro", zero_division=0)
            if abs(f1 - summary[fold]["f1_macro"]) > 1e-6:
                raise SystemExit(f"{run} fold {fold}: rescored F1 {f1:.6f} != summary {summary[fold]['f1_macro']:.6f}")
            heads[fold] = (te, iv, scaler, model)
        for p in todo:
            X = np.load(p)
            if X.shape != Xc.shape:
                raise SystemExit(f"{p}: shape {X.shape} != {Xc.shape}")
            crc = json.loads(p.with_suffix(".json").read_text())["names_crc32"]
            res = {"y": y, "test_fold": np.full(len(y), -1, np.int16),
                   "test_logits": np.full((len(y), n_cls), np.nan, np.float32), "names_crc32": np.int64(crc)}
            for fold, (te, iv, scaler, model) in heads.items():
                res["test_logits"][te] = logits_of(model, scaler, X[te], n_cls)
                res["test_fold"][te] = fold
                res[f"iv_idx_{fold}"], res[f"iv_logits_{fold}"] = iv, logits_of(model, scaler, X[iv], n_cls)
            np.savez_compressed(out_dir / f"{p.stem}.npz", **res)
        print(f"{run.parent.name}/{run.name}: scored {len(todo)} file(s), F1 equal to summary.json", flush=True)


if __name__ == "__main__":
    main()
