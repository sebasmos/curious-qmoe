"""T17 head training (ANALYSIS_PLAN_T17.md section 3). One loop for every arm: a group of M members (M = 1 or 2) trained in lock step on shared minibatches.
Recipe = heads.fit_head (AdamW, lr/wd/label smoothing/class-weighted loss/plateau on inner-validation accuracy/early stopping/best checkpoint of config/urban8k.yaml).
With one member, mode 'plain' and bits 32 this reproduces heads.fit_head(metric='acc') bit for bit (checked in t17_check.py).
Modes: plain (cross-entropy only; a pair with lam 0 is two independent heads trained together), ncl (negative correlation learning, minus lam/2 times the mean squared
distance of the two probability vectors), dec (error decorrelation: plus lam times the batch cosine of the two residual vectors p_i - onehot(y)), distill (member 0 only:
per-sample alpha_i * CE_smoothed + (1 - alpha_i) * T^2 * KL(teacher_T || student_T))."""
from __future__ import annotations
import copy, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads import RC, HIDDEN, DROP, NCLS  # noqa: E402
from QWave.qmoe_layers import BitNetExpert  # noqa: E402


class Group:
    def __init__(self, models, scalers): self.models, self.scalers = models, scalers

    def logits(self, Xs):
        out = []
        for m, sc, X in zip(self.models, self.scalers, Xs):
            m.eval()
            with torch.no_grad(): out.append(m(torch.tensor(sc.transform(X), dtype=torch.float32)).numpy().astype(np.float64))
        return out


def train_group(Xfit, yfit, wfit, seed, Xsel, ysel, K, mode="plain", lam=0.0, bits=32, teacher=None, T=1.0, alpha=1.0, kd_mask=None):
    M = len(Xfit); g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    scs = [StandardScaler().fit(X) for X in Xfit]
    X = [torch.tensor(sc.transform(x), dtype=torch.float32) for sc, x in zip(scs, Xfit)]
    y = torch.tensor(yfit, dtype=torch.long); w = [torch.tensor(v, dtype=torch.float32) for v in wfit]
    cw = torch.tensor(1.0 / np.maximum(np.bincount(yfit, minlength=K).astype(float), 1.0), dtype=torch.float32)
    models = [BitNetExpert(x.shape[1], K, HIDDEN, DROP, num_bits=bits) for x in X]
    params = [p for m in models for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=RC.lr, weight_decay=RC.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=RC.lr_factor, patience=RC.lr_patience)
    Xs = [torch.tensor(sc.transform(x), dtype=torch.float32) for sc, x in zip(scs, Xsel)]
    if mode == "distill":
        assert M == 1 and teacher is not None
        Tt = torch.tensor(teacher, dtype=torch.float32); km = torch.tensor(np.ones(len(yfit), bool) if kd_mask is None else kd_mask)
        a_i = torch.where(km, torch.tensor(float(alpha)), torch.tensor(1.0))
    best, best_state, bad = -1.0, None, 0; n, bs = len(y), int(RC.batch_size); ls = float(RC.label_smoothing)
    for ep in range(int(RC.max_epochs)):
        for m in models: m.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; outs = [m(x[b]) for m, x in zip(models, X)]; loss = 0.0
            for k, out in enumerate(outs):
                ce = F.cross_entropy(out, y[b], reduction="none", label_smoothing=ls)
                if mode == "distill":
                    ls_ = F.log_softmax(out / T, 1); pt = F.softmax(Tt[b] / T, 1)
                    kl = (pt * (torch.log(pt.clamp_min(1e-12)) - ls_)).sum(1) * T * T
                    ce = a_i[b] * ce + (1 - a_i[b]) * kl
                loss = loss + (w[k][b] * cw[y[b]] * ce).sum() / (w[k][b] * cw[y[b]]).sum()
            if M == 2 and lam > 0 and mode in ("ncl", "dec"):
                p1, p2 = F.softmax(outs[0], 1), F.softmax(outs[1], 1)
                if mode == "ncl": loss = loss - 0.5 * lam * ((p1 - p2) ** 2).sum(1).mean()
                else:
                    oh = F.one_hot(y[b], K).float(); e1, e2 = p1 - oh, p2 - oh
                    loss = loss + lam * (e1 * e2).sum() / ((e1 * e1).sum().sqrt() * (e2 * e2).sum().sqrt() + 1e-12)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, float(RC.grad_clip_norm)); opt.step()
        for m in models: m.eval()
        with torch.no_grad(): pr = torch.stack([F.softmax(m(x), 1) for m, x in zip(models, Xs)]).mean(0).argmax(1).numpy()
        met = float((pr == ysel).mean()); sched.step(met)
        if met > best:
            best, bad = met, 0; best_state = [copy.deepcopy(m.state_dict()) for m in models]
        else:
            bad += 1
            if bad >= int(RC.patience): break
    for m, s in zip(models, best_state): m.load_state_dict(s)
    G = Group(models, scs); G.epochs_run = ep + 1; G.best_sel = best
    return G
