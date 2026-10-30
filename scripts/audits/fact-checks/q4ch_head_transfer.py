"""AST-Q4ch (per-channel 4-bit weights) on ESC-50 (50 classes): head transferred from FP32, head with a refit input scaler,
and a head trained on its own embeddings. CPU only. Reads the idea 5 stage 2 folder, writes only here.

Variants (all scored on the same test folds, macro F1 per fold, then mean over folds, then over seeds):
  T     FP32-trained head and FP32 scaler (what the paper's Tables 2, 3 and 10 use). Must reproduce analysis/esc50/static.csv.
  Rfull FP32 head weights, input scaler refit on the backbone's own clean embeddings of the fold's fit rows (4 of 5 folds minus inner val).
  R128  same, scaler refit on 128 random clean fit rows (the paper's int8 recipe uses 128 clips).
  O     head trained on the backbone's own clean embeddings (stage 2 csv/esc-50_ast_w4.csv), its own scaler. w4 only.
"""
import sys, json, contextlib, io
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
from omegaconf import OmegaConf

W = Path(__file__).resolve().parents[3]  # archive root
sys.path.insert(0, str(W)); sys.path.insert(0, str(W / "scripts/redesign/idea5"))
from scripts.analysis.cascade import load_model
from score_stage2 import train_frame, fold_scaler, logits_of
torch.set_num_threads(2)

R = Path("/Users/cajas.sebastian/Documents/qmoe-results/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38")
HERE = Path(__file__).resolve().parent
CONF = ["clean"] + [f"{n}__{s}db" for n in ("bg", "pink", "white") for s in (20, 10, 5, 0)]
CFGS = ["fp32", "w4", "w4g128", "w3g64"]
own_runs = {0: R / "outputs/astesc50headw4_s0/esc"}
for s in range(1, 5):
    p = HERE / f"heads_w4/astesc50headw4_s{s}/esc"
    if (p / "summary.json").exists():
        own_runs[s] = p
rows = []
for seed in range(5):
    run = R / f"outputs/astesc50_s{seed}/esc"
    cfg = OmegaConf.load(run / "config.yaml")
    Xc, y = train_frame(cfg)            # FP32 embeddings the head was trained on
    n_cls = len(np.unique(y)); norm = cfg.experiment.datasets.esc.normalization_type
    summ = json.loads((run / "summary.json").read_text())["folds"]
    folds = sorted(f["fold"] for f in summ)
    emb = {(c, k): np.load(R / f"emb/esc50/{c}__{k}.npy") for c in CFGS for k in CONF}
    if seed in own_runs:
        orun = own_runs[seed]; ocfg = OmegaConf.load(orun / "config.yaml"); Xo, yo = train_frame(ocfg)
    for fold in folds:
        split = json.loads((run / f"fold_{fold}" / "split.json").read_text())
        te = np.asarray(split["test_idx"])
        model, _ = load_model(run, fold, Xc.shape[1], n_cls)
        sc_fp = fold_scaler(Xc, y, split, norm)
        fit = np.setdiff1d(np.arange(len(Xc)), np.concatenate([split["test_idx"], split["inner_val_idx"]]))
        rng = np.random.default_rng(1000 * seed + fold)
        sub = rng.choice(fit, 128, replace=False)
        for c in CFGS:
            Xclean = emb[(c, "clean")]
            sc_full = fold_scaler(Xclean, y, split, norm)
            # 128-row refit: pass a split whose fit rows are exactly `sub`
            rest = np.setdiff1d(np.arange(len(Xc)), sub)
            sc_128 = fold_scaler(Xclean, y, {"test_idx": rest[: len(rest) // 2], "inner_val_idx": rest[len(rest) // 2:]}, norm)
            if c == "w4" and seed in own_runs:
                osplit = json.loads((orun / f"fold_{fold}" / "split.json").read_text())
                omodel, _ = load_model(orun, fold, Xo.shape[1], n_cls)
                sc_own = fold_scaler(Xo, yo, osplit, norm)
            for k in CONF:
                X = emb[(c, k)]
                f = lambda m, s: f1_score(y[te], logits_of(m, s, X[te], n_cls).argmax(1), average="macro", zero_division=0)
                rows += [dict(seed=seed, fold=fold, cfg=c, cond=k, variant="T", f1=f(model, sc_fp)),
                         dict(seed=seed, fold=fold, cfg=c, cond=k, variant="Rfull", f1=f(model, sc_full)),
                         dict(seed=seed, fold=fold, cfg=c, cond=k, variant="R128", f1=f(model, sc_128))]
                if c == "w4" and seed in own_runs:
                    rows.append(dict(seed=seed, fold=fold, cfg=c, cond=k, variant="O", f1=f(omodel, sc_own)))
    print("seed", seed, "done", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "q4ch_head_transfer_raw.csv", index=False)
