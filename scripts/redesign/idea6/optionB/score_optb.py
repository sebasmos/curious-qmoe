"""Score trained heads (scripts/benchmark.py run directories, trained on the device-A rows) on the universe
embeddings of TAU 2020 Mobile (idea 6 option B): every row of the universe CSV is scored by the head of its
location's fold, device-A rows and the other devices' rows alike, plus each fold's inner-validation logits.

For each run <outputs>/<tag>/<kind> and each --configs name this writes <out>/<tag>/<kind>/<config>__all.npz:
  y, fold, device [N]        class id, fold (folder column) and device of every universe row
  test_logits [N, C]         each row scored by the head of its fold
  iv_idx_<k>, iv_logits_<k>  fold k's inner-validation rows (device A) and their logits
  n_a, names_crc32           size of the device-A block; zlib.crc32 of the universe stems joined by newlines
Before writing, every fold's macro F1 on the training CSV's test rows (device A, scored from the training CSV's
own features) must equal the fold's F1 in summary.json, and the universe array's device-A block must equal the
training CSV's features for the FP32 configuration (relative 1e-6).

    python scripts/redesign/idea6/optionB/score_optb.py --runs R/outputs/mn10_tau_s0/esc ... --emb-dir R/emb_cnn/tau \
        --configs mn10_fp32 mn10_w8a8 --meta ~/Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv --out R/scores
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6" / "stage2"))

from score_eff import fold_scaler, logits_of, train_frame  # noqa: E402
from scripts.analysis.cascade import load_model  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--emb-dir", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    meta = pd.read_csv(a.meta)
    crc = zlib.crc32("\n".join(meta["name"].map(lambda s: Path(s).stem)).encode())
    y_u = meta.class_id.to_numpy(np.int64)
    fold_u = meta.folder.to_numpy(np.int16)
    dev_u = meta.device.to_numpy().astype(str)
    n_a = int((meta.device == "a").sum())
    U = {}
    for c in a.configs:
        side = json.loads((a.emb_dir / f"{c}__clean.json").read_text())
        if side["names_crc32"] != crc or side["n_clips"] != len(meta):
            raise SystemExit(f"{c}: sidecar rows differ from {a.meta}")
        U[c] = np.load(a.emb_dir / f"{c}__clean.npy")
        if len(U[c]) != len(meta) or not np.isfinite(U[c]).all():
            raise SystemExit(f"{c}: bad array")
    for run in a.runs:
        run = run.resolve()
        dest_dir = a.out / run.parent.name / run.name
        if all((dest_dir / f"{c}__all.npz").exists() for c in a.configs):
            continue
        cfg = OmegaConf.load(run / "config.yaml")
        Xc, y = train_frame(cfg, next(iter(U.values())).shape[1])
        if len(Xc) != n_a or not np.array_equal(y, y_u[:n_a]):
            raise SystemExit(f"{run}: training CSV is not the device-A block")
        fp32 = [c for c in a.configs if c.endswith("fp32") or c == "fp32"]
        for c in fp32:
            if np.abs(U[c][:n_a] - Xc).max() > 1e-6 * np.abs(Xc).max():
                raise SystemExit(f"{run}: {c} device-A block differs from the training CSV features")
        n_cls = len(np.unique(y))
        norm = cfg.experiment.datasets.esc.normalization_type
        summary = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
        heads = {}
        for fold in sorted(summary):
            split = json.loads((run / f"fold_{fold}" / "split.json").read_text())
            te, iv = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
            if not np.array_equal(te, np.flatnonzero(fold_u[:n_a] == fold)):
                raise SystemExit(f"{run} fold {fold}: test rows are not the fold's device-A rows")
            model, _ = load_model(run, fold, Xc.shape[1], n_cls)
            scaler = fold_scaler(Xc, y, split, norm)
            f1 = f1_score(y[te], logits_of(model, scaler, Xc[te], n_cls).argmax(1), average="macro", zero_division=0)
            if abs(f1 - summary[fold]["f1_macro"]) > 1e-6:
                raise SystemExit(f"{run} fold {fold}: rescored F1 {f1:.6f} != summary {summary[fold]['f1_macro']:.6f}")
            heads[fold] = (iv, scaler, model)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for c in a.configs:
            res = {"y": y_u, "fold": fold_u, "device": dev_u, "n_a": np.int64(n_a), "names_crc32": np.int64(crc),
                   "test_logits": np.full((len(meta), n_cls), np.nan, np.float32)}
            for fold, (iv, scaler, model) in heads.items():
                rows = np.flatnonzero(fold_u == fold)
                res["test_logits"][rows] = logits_of(model, scaler, U[c][rows], n_cls)
                res[f"iv_idx_{fold}"], res[f"iv_logits_{fold}"] = iv, logits_of(model, scaler, U[c][iv], n_cls)
            if not np.isfinite(res["test_logits"]).all():
                raise SystemExit(f"{run} {c}: rows left unscored")
            np.savez_compressed(dest_dir / f"{c}__all.npz", **res)
        print(f"{run.parent.name}/{run.name}: {len(heads)} folds, {len(a.configs)} configs, F1 equal to summary.json", flush=True)


if __name__ == "__main__":
    main()
