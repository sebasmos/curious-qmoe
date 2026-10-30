"""Check 4b: how much do the W8A8 embeddings depend on the device they were computed on? The stored ones
came from MPS; here 96 clean ESC-50 test clips of fold 1 are re-embedded on the CPU with the same code
and scored by the fold-1 heads of seed 0 (H1 cheap heads of ESC-50/50 and of the five groups). Compares
top-1, margin and the H1 escalation decision with the stored logits. Writes outputs/check4b_device.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import OUT, R, margin, sm, threshold  # noqa: E402

W = Path("/Users/cajas.sebastian/Desktop/repositories/curious-qmoe-redesign-idea6-stage2")
sys.path.insert(0, str(W))
sys.path.insert(0, str(W / "scripts/redesign/idea6/stage2"))
from QWave.effnet_quant import MelImageSpec, build_backbone, efficientnet_b3_pooled, mel_images  # noqa: E402
from audio_shift import index_wavs, load_clip  # noqa: E402
from score_eff import fold_scaler, train_frame  # noqa: E402
from scripts.analysis.cascade import load_model  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

torch.set_num_threads(2)
E = Path.home() / "Documents/DATASETS"
stems = pd.read_csv(E / "ESC-50/ast_768/esc-50.csv", usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
index = index_wavs(E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio")
z0 = np.load(R / "scores/esc50/effp_esc50_s0/esc/w8a8__clean.npz")
rows = np.flatnonzero(z0["test_fold"] == 1)
rows = np.sort(np.random.default_rng(5).choice(rows, 96, replace=False))
base = efficientnet_b3_pooled()
w8 = build_backbone("w8a8", base, "cpu")
X = []
for b in range(0, len(rows), 16):
    imgs = mel_images([load_clip(index[stems[i]]) for i in rows[b:b + 16]], MelImageSpec())
    with torch.no_grad():
        X.append(w8(imgs).numpy())
X = np.concatenate(X)
stored = np.load(R / "emb/esc50/w8a8__clean.npy", mmap_mode="r")[rows]
out = {"n_clips": len(rows), "emb_rel_diff_median": float(np.median(np.linalg.norm(X - stored, axis=1) / np.linalg.norm(stored, axis=1)))}
for task, kind, budget in (("esc50", "esc", 0.5), ("esc5", "4", 0.5)):
    run = R / f"outputs/effp_{task}_s0/{kind}"
    cfg = OmegaConf.load(run / "config.yaml")
    Xc, y = train_frame(cfg, 1536)
    split = json.loads((run / "fold_1/split.json").read_text())
    n_cls = len(np.unique(y))
    model, _ = load_model(run, 1, 1536, n_cls)
    sc = fold_scaler(Xc, y, split, cfg.experiment.datasets.esc.normalization_type)
    with torch.no_grad():
        lg = model(torch.tensor(sc.transform(X), dtype=torch.float32)).numpy()
    z = np.load(R / f"scores/{task}/effp_{task}_s0/{kind}/w8a8__clean.npz")
    ls = z["test_logits"][rows]
    p_new, p_old = sm(lg), sm(ls)
    t = threshold(margin(sm(z["iv_logits_1"])), budget)
    e_new, e_old = margin(p_new) < t, margin(p_old) < t
    yy = z["y"][rows]
    out[task] = dict(top1_agree=float((p_new.argmax(1) == p_old.argmax(1)).mean()),
                     acc_cpu=float((p_new.argmax(1) == yy).mean()), acc_stored=float((p_old.argmax(1) == yy).mean()),
                     margin_abs_diff_median=float(np.median(np.abs(margin(p_new) - margin(p_old)))),
                     escalation_agree=float((e_new == e_old).mean()), rate_cpu=float(e_new.mean()),
                     rate_stored=float(e_old.mean()))
(OUT / "check4b_device.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
