"""Head training for the difficulty mixture-of-experts pilot. The protocol 3 recipe of config/urban8k.yaml with per-sample weights
(see ANALYSIS_PLAN.md B2). Architecture and recipe come from the library (BitNetExpert num_bits=32, build_training_recipe)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from QWave.qmoe_layers import BitNetExpert  # noqa: E402

CFG = OmegaConf.load(ROOT / "config/urban8k.yaml").experiment
RC = CFG.train
HIDDEN = list(CFG.model.hidden_sizes)
DROP = float(CFG.model.dropout_prob)
NCLS = 10


def macro_f1(y, p, k=NCLS):  # k: number of classes
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))


class Head:
    def __init__(self, model, scaler):
        self.model, self.scaler = model, scaler

    def logits(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(self.scaler.transform(X), dtype=torch.float32)).numpy().astype(np.float64)


def fit_head(Xfit, yfit, wfit, seed, Xsel=None, ysel=None, metric="f1", gamma=0.0, fixed_epochs=None, Xrec=None, hidden=None, ncls=None):
    """Train one head. wfit: per-sample weights (all > 0; remove zero-weight clips before calling).
    Selection: early stopping, plateau schedule and best checkpoint on `metric` ('f1' macro F1, 'acc' accuracy) of (Xsel, ysel).
    fixed_epochs: train exactly that many epochs at constant lr, no selection; Xrec: if given, returns the logits on Xrec after
    every epoch as [epochs, n, C]. ncls: number of classes (default NCLS = 10, so callers that omit it are unchanged)."""
    K = ncls or NCLS
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xfit)
    X = torch.tensor(sc.transform(Xfit), dtype=torch.float32)
    y = torch.tensor(yfit, dtype=torch.long)
    w = torch.tensor(wfit, dtype=torch.float32)
    cnt = np.bincount(yfit, minlength=K).astype(float)
    cw = torch.tensor(1.0 / np.maximum(cnt, 1.0), dtype=torch.float32)
    model = BitNetExpert(X.shape[1], K, hidden or HIDDEN, DROP, num_bits=32)
    opt = torch.optim.AdamW(model.parameters(), lr=RC.lr, weight_decay=RC.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=RC.lr_factor, patience=RC.lr_patience)
    Xs = None if Xsel is None else torch.tensor(sc.transform(Xsel), dtype=torch.float32)
    Xr = None if Xrec is None else torch.tensor(sc.transform(Xrec), dtype=torch.float32)
    rec, best, best_state, bad = [], -1.0, None, 0
    n, bs = len(X), int(RC.batch_size)
    for ep in range(fixed_epochs or int(RC.max_epochs)):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            out = model(X[b])
            ce = F.cross_entropy(out, y[b], reduction="none", label_smoothing=float(RC.label_smoothing))
            den = (w[b] * cw[y[b]]).sum()
            if gamma > 0:
                pt = torch.softmax(out.detach(), 1).gather(1, y[b][:, None])[:, 0]
                ce = ce * (1 - pt) ** gamma
            loss = (w[b] * cw[y[b]] * ce).sum() / den
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(RC.grad_clip_norm))
            opt.step()
        if Xr is not None:
            model.eval()
            with torch.no_grad():
                rec.append(model(Xr).numpy().astype(np.float32))
        if fixed_epochs:
            continue
        model.eval()
        with torch.no_grad():
            pr = model(Xs).argmax(1).numpy()
        m = macro_f1(ysel, pr, K) if metric == "f1" else float((pr == ysel).mean())
        sched.step(m)
        if m > best:
            best, bad = m, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= int(RC.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    h = Head(model, sc)
    h.epochs_run = ep + 1
    h.best_sel = best
    return (h, np.stack(rec)) if Xrec is not None else h
