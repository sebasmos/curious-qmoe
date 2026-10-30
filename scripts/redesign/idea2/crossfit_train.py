#!/usr/bin/env python3
"""Cross-fitted single models for the idea-2 router pilot.

One job trains one single model (bitnet, 1, 4, esc, ...) on every outer fold
except the held-out folds, with the protocol-v2 recipe of the config it reads
(inner validation split for early stopping and checkpoint selection, scaler
fitted on the fit portion, per-job seeding), and writes the softmax of the
selected checkpoint on each held-out fold. A model that never saw folds j and
k gives honest correctness labels on fold j for the router of outer fold k.
ESC-50 and Quinn (5 folds) hold out every pair of folds (10 models per model
kind, each trained on 3 folds). UrbanSound8K groups its 10 folds into 5
blocks of two ({1,2}, {3,4}, ...) and holds out every block (5 models, 8
training folds, for the partner fold) and every pair of blocks (10 models, 6
training folds, for the other blocks): 15 models per kind instead of 45.

--reproduce k instead trains on every fold except k, the exact protocol-v2
fold, and checks the test macro F1 against the run's metrics.json. The steps
before training (seed, inner split, model init, the two FLOP traces that run
train-mode forwards and draw dropout masks) follow scripts/benchmark.py in
order, so the result should match it bit for bit.

Must run with the code that trained the singles on sys.path (--code).

    python crossfit_train.py --code <worktree 5db7e75> --config <run>/<model>/config.yaml \\
        --model 4 --holdout 3 4 7 8 --out <results>/crossfit/urban
"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--config", required=True, help="config.yaml of any single model of the run (same recipe)")
ap.add_argument("--model", required=True)
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--holdout", type=int, nargs="+")
g.add_argument("--reproduce", type=int)
ap.add_argument("--metrics", help="metrics.json to compare against with --reproduce")
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, args.code)
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from fvcore.nn import FlopCountAnalysis  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from QWave.datasets import EmbeddingAdaptDataset, inner_split, make_outer_folds  # noqa: E402
from QWave.train_utils import train_pytorch_local  # noqa: E402
from QWave.utils import set_seed  # noqa: E402
from scripts.benchmark import build_model, prepare_eval_model  # noqa: E402

cfg = OmegaConf.load(args.config)
cfg.experiment.device = "cpu"
torch.set_num_threads(int(cfg.experiment.num_threads))
seed = int(cfg.experiment.seed)
esc = cfg.experiment.datasets.esc
df_full = pd.read_csv(str(esc.csv))
label_col = esc.get("label_col", "class_id")
if label_col != "class_id":
    df_full["class_id"] = df_full[label_col]
labels = df_full["class_id"].values
df_feat = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")
folds = {f: te for f, _, te in make_outer_folds(df_full, labels, cfg.experiment.cross_validation)}
held = [args.reproduce] if args.reproduce else list(args.holdout)
assert all(h in folds for h in held) and len(set(held)) == len(held), held
tr_idx = np.sort(np.concatenate([te for f, te in folds.items() if f not in held]))
ho_idx = {h: folds[h] for h in held}
norm = esc.normalization_type
in_dim, n_cls = df_feat.shape[1] - 1, len(np.unique(labels))
bs = cfg.experiment.train.batch_size

# same order of RNG draws as scripts/benchmark.py run_cv, one fold
loader_gen = set_seed(seed)
fit_idx, iv_idx = inner_split(tr_idx, labels, float(cfg.experiment.cross_validation.get("validation_split_ratio", 0.1)), seed)
with contextlib.redirect_stdout(io.StringIO()):
    tr_ds = EmbeddingAdaptDataset(df_feat.iloc[fit_idx].reset_index(drop=True), normalization_type=norm)
    iv_ds = EmbeddingAdaptDataset(df_feat.iloc[iv_idx].reset_index(drop=True), normalization_type=norm,
                                  scaler=tr_ds.get_scaler())
    ho_ds = {h: EmbeddingAdaptDataset(df_feat.iloc[ix].reset_index(drop=True), normalization_type=norm,
                                      scaler=tr_ds.get_scaler()) for h, ix in ho_idx.items()}
tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, generator=loader_gen)
iv_ld = DataLoader(iv_ds, batch_size=bs, shuffle=False)
with open_dict(cfg):
    cfg.experiment.current_fold = held[0]
model = build_model(args.model, in_dim, n_cls, cfg)
pre = {k: v.clone() for k, v in model.state_dict().items()}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    FlopCountAnalysis(model, tr_ds.features[:bs]).total()
    FlopCountAnalysis(model, ho_ds[held[0]].features[:bs]).total()
model.load_state_dict(pre)
class_weights = torch.tensor(1 / np.bincount(tr_ds.labels.numpy()), dtype=torch.float32)

t0 = time.perf_counter()
with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
    model, _, _, iv_f1, *_ = train_pytorch_local(cfg.experiment, model, tr_ld, iv_ld, class_weights, in_dim,
                                                 "cpu", tmp, False, None)
dur = time.perf_counter() - t0
model = prepare_eval_model(args.model, model, cfg)


@torch.no_grad()
def softmax(X):  # batches of 64 in order, as the benchmark's test loader
    return torch.cat([F.softmax(model(X[i:i + bs]), dim=1) for i in range(0, len(X), bs)]).numpy()


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
arrs, meta = {}, {"model": args.model, "held_out": held, "seed": seed, "n_fit": int(len(fit_idx)),
                  "n_inner_val": int(len(iv_idx)), "inner_val_f1_macro": float(iv_f1),
                  "train_seconds": dur, "config": str(args.config)}
for h in held:
    p = softmax(ho_ds[h].features)
    y = ho_ds[h].labels.numpy()
    arrs[f"idx_{h}"], arrs[f"p_{h}"], arrs[f"y_{h}"] = ho_idx[h], p, y
    meta[f"f1_fold_{h}"] = float(f1_score(y, p.argmax(1), average="macro", zero_division=0))
if args.reproduce:
    ref = json.loads(Path(args.metrics).read_text())["f1_macro"]
    meta["reference_f1_macro"] = ref
    meta["abs_diff"] = abs(ref - meta[f"f1_fold_{held[0]}"])
    name = f"reproduce_{args.model}_fold{held[0]}"
else:
    name = "ho_" + "_".join(map(str, sorted(held)))
    out = out / args.model
    out.mkdir(exist_ok=True)
np.savez_compressed(out / f"{name}.npz", **arrs)
(out / f"{name}.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))
