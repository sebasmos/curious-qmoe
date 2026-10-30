"""T2b step 2: fit every gate arm on the training folds, choose hyper-parameters on grouped inner validation, write test probabilities, costs and decisions.
Never loads the label of a test clip (only y[tr]). python t2b_fit.py --run DIR [--seeds ..] [--folds ..]   (reads DIR/urban/raw, writes DIR/urban/fit)"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import S3C, EMB, load_table, softmax, pmargin

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
ap.add_argument("--threads", type=int, default=1); a = ap.parse_args(); torch.set_num_threads(a.threads)
EXP = ["E1", "E2c", "E2b", "E2a"]
CJ = json.load(open(S3C / "analysis/urban/costs.json"))
COST = np.array([CJ["cheap"]["cnn:mn10:w8a8:esc"], CJ["ast"]["w4g128@400+esc"], CJ["ast"]["w4g128+esc"], CJ["ast"]["fp32+esc"]])
CT = COST / COST[3]                      # normalized costs, E2a = 1
THETA, LAMS, BETAS, RATES, CS = 0.05, [0, .02, .05, .1, .2, .4], [0.0, 0.1], [*np.arange(.05, .951, .05), .99], [.01, .1, 1, 10]
y_all = load_table().label.values.astype(np.int64); X1 = np.load(EMB["mn10_w8a8"]).astype(np.float32)
ent = lambda p: -(p * np.log(p + 1e-12)).sum(1)
def js(p, q):
    m = .5 * (p + q); kl = lambda u, v: (u * (np.log(u + 1e-12) - np.log(v + 1e-12))).sum(1); return .5 * kl(p, m) + .5 * kl(q, m)

class Gate(nn.Module):
    def __init__(s, d, k): super().__init__(); s.n = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, k))
    def forward(s, x): return s.n(x)

def train_gate(Xf, Xv, lossfn, seed):
    """lossfn(logits, idx_set, 'fit'|'iv') -> scalar. Early stopping on the iv loss."""
    torch.manual_seed(seed); g = Gate(Xf.shape[1], 4); opt = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bs, bad = 1e9, None, 0; n = len(Xf); gen = torch.Generator().manual_seed(seed)
    for ep in range(60):
        g.train(); perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 256):
            b = perm[i:i + 256]; l = lossfn(g(Xf[b]), b, "fit"); opt.zero_grad(); l.backward(); opt.step()
        g.eval()
        with torch.no_grad(): lv = float(lossfn(g(Xv), slice(None), "iv"))
        if lv < best - 1e-5: best, bs, bad = lv, copy.deepcopy(g.state_dict()), 0
        else:
            bad += 1
            if bad >= 10: break
    g.load_state_dict(bs); g.eval(); return g

def soft_out(w, Pk, theta=THETA, trunc=True):
    """w [n,4], Pk [n,4,C]; returns probs and cost (E1 always run, others run when w >= theta)."""
    if not trunc: return np.einsum("nk,nkc->nc", w, Pk), np.full(len(w), COST.sum())
    run = w >= theta; run[:, 0] = True; wr = w * run; wr = wr / wr.sum(1, keepdims=True)
    return np.einsum("nk,nkc->nc", wr, Pk), (run * COST[None]).sum(1)

(a.run / "urban/fit").mkdir(parents=True, exist_ok=True)
for seed in a.seeds:
    for fold in a.folds:
        fo = a.run / f"urban/fit/s{seed}_f{fold}.npz"
        if fo.exists(): continue
        t0 = time.time(); R = np.load(a.run / f"urban/raw/s{seed}_f{fold}.npz"); tr, te, fp, ip = R["tr"], R["te"], R["fit_p"], R["iv_p"]; ytr = y_all[tr]
        Zo = np.stack([R[k].astype(np.float64) for k in ["z1_oof", "E2c_oof", "E2b_oof", "E2a_oof"]], 1)       # [ntr,4,C]
        Zt = np.stack([R[k].astype(np.float64) for k in ["z1_te", "E2c_te", "E2b_te", "E2a_te"]], 1)
        Po = np.stack([softmax(Zo[:, k]) for k in range(4)], 1); Pt = np.stack([softmax(Zt[:, k]) for k in range(4)], 1)
        corr = (Po.argmax(2) == ytr[:, None]); err = 1.0 - corr
        oof_acc = corr.mean(0); kb = int(oof_acc.argmax()); B = COST[kb]
        pca = PCA(32, random_state=seed).fit(X1[tr[fp]]); Ho, Ht = pca.transform(X1[tr]), pca.transform(X1[te])
        def feats(P, H, full):
            f = [np.log(P[:, 0] + 1e-12), pmargin(P[:, 0])[:, None], P[:, 0].max(1)[:, None], ent(P[:, 0])[:, None], H]
            if full:
                for k in (1, 2, 3): f += [np.log(P[:, k] + 1e-12), pmargin(P[:, k])[:, None], ent(P[:, k])[:, None]]
                f += [js(P[:, i], P[:, j])[:, None] for i in range(4) for j in range(i + 1, 4)]
            return np.hstack(f)
        out = dict(kb=kb, B=B, oof_acc=oof_acc, cost=COST); grids = {}
        def select(rows, budget=True):
            """rows: list of (acc, cost, lam, beta, tag). Highest iv acc with cost <= B; ties lower cost, lam, beta; none feasible: cheapest."""
            ok = [r for r in rows if r[1] <= B * (1 + 1e-9)] if budget else rows
            if not ok: return min(rows, key=lambda r: (r[1], r[2], r[3]))
            return min(ok, key=lambda r: (-r[0], r[1], r[2], r[3]))
        yi = ytr[ip]
        # A4 always-average, avg2 (extra reference)
        out["A4_p"] = Pt.mean(1).astype(np.float32); out["A4_cost"] = np.full(len(te), COST.sum()); out["AVG2_p"] = (.5 * (Pt[:, 0] + Pt[:, 3])).astype(np.float32)
        # A1 margin threshold E1 -> E2a
        mo, mt = pmargin(Po[:, 0]), pmargin(Pt[:, 0]); rows = []
        for r in RATES:
            tau = np.quantile(mo[ip], r); fl = mo[ip] < tau; pr = np.where(fl[:, None], Po[ip, 3], Po[ip, 0]).argmax(1)
            rows.append(((pr == yi).mean(), COST[0] + fl.mean() * COST[3], r, 0.0, tau))
        s = select(rows); out["A1_sel"] = np.array(s[:4]); fl = mt < s[4]
        out["A1_p"] = np.where(fl[:, None], Pt[:, 3], Pt[:, 0]).astype(np.float32); out["A1_cost"] = COST[0] + fl * COST[3]; grids["A1"] = np.array([r[:4] for r in rows])
        for full in (False, True):
            Fo, Ft = feats(Po, Ho, full), feats(Pt, Ht, full); sc = StandardScaler().fit(Fo[fp])
            Xf, Xv, Xt = [torch.tensor(sc.transform(v[ix] if ix is not None else v), dtype=torch.float32) for v, ix in ((Fo, fp), (Fo, ip), (Ft, None))]
            ErrT, PoT, yT = torch.tensor(err, dtype=torch.float32), torch.tensor(Po, dtype=torch.float32), torch.tensor(ytr)
            cT = torch.tensor(CT, dtype=torch.float32); idx = {"fit": torch.tensor(fp), "iv": torch.tensor(ip)}
            # hard router
            rows, store = [], {}
            for lam in ([0.0] if full else LAMS):
                def lh(lg, b, which, lam=lam):
                    ii = idx[which][b] if which == "fit" else idx[which]; pi = F.softmax(lg, 1); return (pi * (ErrT[ii] + lam * cT[None])).sum(1).mean()
                g = train_gate(Xf, Xv, lh, seed)
                with torch.no_grad(): av, at = g(Xv).argmax(1).numpy(), g(Xt).argmax(1).numpy()
                accv = (Po[ip][np.arange(len(ip)), av].argmax(1) == yi).mean()
                cv = COST.sum() if full else (COST[0] + np.where(av > 0, COST[av], 0)).mean()
                rows.append((accv, cv, lam, 0.0, lam)); store[lam] = at
            s = select(rows, budget=not full); at = store[s[4]]; nm = "A2f" if full else "A2h"
            out[f"{nm}_sel"] = np.array(s[:4]); out[f"{nm}_action"] = at; out[f"{nm}_p"] = Pt[np.arange(len(te)), at].astype(np.float32)
            out[f"{nm}_cost"] = np.full(len(te), COST.sum()) if full else COST[0] + np.where(at > 0, COST[at], 0); grids[nm] = np.array([r[:4] for r in rows])
            # soft mixture
            rows, store = [], {}
            for lam in ([0.0] if full else LAMS):
                for beta in BETAS:
                    def ls(lg, b, which, lam=lam, beta=beta):
                        ii = idx[which][b] if which == "fit" else idx[which]; w = F.softmax(lg, 1)
                        p = (w[:, :, None] * PoT[ii]).sum(1); ce = -torch.log(p.gather(1, yT[ii][:, None])[:, 0] + 1e-8).mean()
                        return ce + lam * (w * cT[None]).sum(1).mean() + beta * (-(w * torch.log(w + 1e-12)).sum(1)).mean()
                    g = train_gate(Xf, Xv, ls, seed)
                    with torch.no_grad(): wv, wt = F.softmax(g(Xv), 1).numpy(), F.softmax(g(Xt), 1).numpy()
                    pv, cv = soft_out(wv, Po[ip], trunc=not full)
                    rows.append(((pv.argmax(1) == yi).mean(), cv.mean(), lam, beta, (lam, beta))); store[(lam, beta)] = wt
            s = select(rows, budget=not full); wt = store[s[4]]; nm = "A3f" if full else "A3h"
            pt, ct = soft_out(wt, Pt, trunc=not full); out[f"{nm}_sel"] = np.array(s[:4]); out[f"{nm}_w"] = wt.astype(np.float32)
            out[f"{nm}_p"] = pt.astype(np.float32); out[f"{nm}_cost"] = ct; grids[nm] = np.array([r[:4] for r in rows])
            if not full: out["A3h_cost_all"] = np.full(len(te), COST.sum()); out["A3h_p_all"] = np.einsum("nk,nkc->nc", wt, Pt).astype(np.float32)
        # A5 stacking
        Lo, Lt = Zo.reshape(len(tr), -1), Zt.reshape(len(te), -1); sc = StandardScaler().fit(Lo[fp]); rows, store = [], {}
        for c in CS:
            m = LogisticRegression(C=c, max_iter=500).fit(sc.transform(Lo[fp]), ytr[fp]); rows.append((m.score(sc.transform(Lo[ip]), yi), COST.sum(), c, 0.0, c)); store[c] = m
        s = select(rows, budget=False); out["A5_sel"] = np.array(s[:4]); out["A5_p"] = store[s[4]].predict_proba(sc.transform(Lt)).astype(np.float32); out["A5_cost"] = np.full(len(te), COST.sum()); grids["A5"] = np.array([r[:4] for r in rows])
        for k, v in grids.items(): out[f"grid_{k}"] = v
        np.savez_compressed(fo, **out)
        print(f"s{seed} f{fold}: {time.time()-t0:.0f}s best single {EXP[kb]} oof acc {oof_acc.max():.4f}; iv sel " + " ".join(f"{n}={out[n+'_sel'][0]:.3f}" for n in ["A1", "A2h", "A2f", "A3h", "A3f", "A5"]), flush=True)
