"""T1c gated-mixture arms (ANALYSIS_PLAN_T1C.md section 7 = ANALYSIS_PLAN_T2B.md sections 3 to 6, generalised to G experts and several test conditions).
Fits every gate on CLEAN out-of-fold expert outputs of the training clips, chooses hyper-parameters on clean grouped inner validation, then applies the fitted gate to the rows of each
requested condition. Reads no test label (only y[tr]). python t1c_t2b_fit.py --run DIR --bed B --conditions C1 C2 ... [--seeds ..] [--folds ..]"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from t1c_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); ap.add_argument("--bed", required=True); ap.add_argument("--conditions", nargs="+", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--folds", type=int, nargs="+", default=None); ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args(); torch.set_num_threads(a.threads)
bed = a.bed; D = load_bed(bed); y_all = D["y"]; K = BEDS[bed]["ncls"]; EX2 = EXPERTS2[bed]; EXP = ["E1"] + EX2; G = len(EXP)
CT0 = cost_table(bed); COST = np.array([CT0[e] for e in EXP]); CT = COST / COST[-1]            # normalised, E2a = 1
THETA, LAMS, BETAS, RATES, CS = 0.05, [0, .02, .05, .1, .2, .4], [0.0, 0.1], [*np.arange(.05, .951, .05), .99], [.01, .1, 1, 10]
folds = a.folds or list(range(1, BEDS[bed]["nf"] + 1))
X1c = np.load(emb_path(bed, "E1", "clean")).astype(np.float32); mm1 = lambda cell: np.load(emb_path(bed, "E1", cell), mmap_mode="r")
ent = lambda p: -(p * np.log(p + 1e-12)).sum(1)
def js(p, q):
    m = .5 * (p + q); kl = lambda u, v: (u * (np.log(u + 1e-12) - np.log(v + 1e-12))).sum(1); return .5 * kl(p, m) + .5 * kl(q, m)

class Gate(nn.Module):
    def __init__(s, d, k): super().__init__(); s.n = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, k))
    def forward(s, x): return s.n(x)

def train_gate(Xf, Xv, lossfn, seed):
    torch.manual_seed(seed); g = Gate(Xf.shape[1], G); opt = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-4)
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

def soft_out(w, Pk, trunc=True):
    if not trunc: return np.einsum("nk,nkc->nc", w, Pk), np.full(len(w), COST.sum())
    run = w >= THETA; run[:, 0] = True; wr = w * run; wr = wr / wr.sum(1, keepdims=True)
    return np.einsum("nk,nkc->nc", wr, Pk), (run * COST[None]).sum(1)

def feats(P, H, full):
    f = [np.log(P[:, 0] + 1e-12), pmargin(P[:, 0])[:, None], P[:, 0].max(1)[:, None], ent(P[:, 0])[:, None], H]
    if full:
        for k in range(1, G): f += [np.log(P[:, k] + 1e-12), pmargin(P[:, k])[:, None], ent(P[:, k])[:, None]]
        f += [js(P[:, i], P[:, j])[:, None] for i in range(G) for j in range(i + 1, G)]
    return np.hstack(f)

(a.run / bed / "t2b_fit").mkdir(parents=True, exist_ok=True)
for seed in a.seeds:
    for fold in folds:
        fo = a.run / bed / f"t2b_fit/s{seed}_f{fold}.npz"
        if fo.exists(): continue
        t0 = time.time(); S = np.load(a.run / bed / f"raw_strength/s{seed}_f{fold}.npz"); O = np.load(a.run / bed / f"raw_oof/s{seed}_f{fold}.npz")
        tr, te, fp, ip = S["tr"], S["te"], S["fit_p"], S["iv_p"]; ytr = y_all[tr]; cond = S["cond"] if bed == "tau" else None
        Zo = np.stack([O[k].astype(np.float64) for k in ["E1_oof"] + [f"{e}_oof" for e in EX2]], 1)                          # [ntr, G, K]
        Po = np.stack([softmax(Zo[:, k]) for k in range(G)], 1); corr = (Po.argmax(2) == ytr[:, None]); err = 1.0 - corr
        oof_acc = corr.mean(0); kb = int(oof_acc.argmax()); B = COST[kb]
        pca = PCA(32, random_state=seed).fit(X1c[tr[fp]]); Ho = pca.transform(X1c[tr])
        # test rows per condition
        TST = {}
        for cn in a.conditions:
            g = groups(bed, cn, te, D["assign"], cond)
            Zt = np.stack([gather({c: S[f"z1|{c}"] for c, _ in g}, g).astype(np.float64)] + [gather({c: S[f"UA_{e}|{c}"] for c, _ in g}, g).astype(np.float64) for e in EX2], 1)
            Xr = np.concatenate([np.asarray(mm1(c)[te], np.float32)[i] for c, i in g], 0)
            Pt = np.stack([softmax(Zt[:, k]) for k in range(G)], 1); TST[cn] = dict(Zt=Zt, Pt=Pt, Ht=pca.transform(Xr), n=len(Zt))
        out = dict(kb=kb, B=B, oof_acc=oof_acc, cost=COST); grids = {}
        def select(rows, budget=True):
            ok = [r for r in rows if r[1] <= B * (1 + 1e-9)] if budget else rows
            if not ok: return min(rows, key=lambda r: (r[1], r[2], r[3]))
            return min(ok, key=lambda r: (-r[0], r[1], r[2], r[3]))
        yi = ytr[ip]
        for cn, T in TST.items():
            Pt = T["Pt"]; out[f"{cn}|A4_p"] = Pt.mean(1).astype(np.float32); out[f"{cn}|A4_cost"] = np.full(T["n"], COST.sum()); out[f"{cn}|AVG2_p"] = (.5 * (Pt[:, 0] + Pt[:, -1])).astype(np.float32)
        # A1 margin threshold E1 -> E2a (last expert)
        mo = pmargin(Po[:, 0]); rows = []
        for r in RATES:
            tau = np.quantile(mo[ip], r); fl = mo[ip] < tau; pr = np.where(fl[:, None], Po[ip, -1], Po[ip, 0]).argmax(1)
            rows.append(((pr == yi).mean(), COST[0] + fl.mean() * COST[-1], r, 0.0, tau))
        s = select(rows); out["A1_sel"] = np.array(s[:4]); grids["A1"] = np.array([r[:4] for r in rows])
        for cn, T in TST.items():
            fl = pmargin(T["Pt"][:, 0]) < s[4]; out[f"{cn}|A1_p"] = np.where(fl[:, None], T["Pt"][:, -1], T["Pt"][:, 0]).astype(np.float32); out[f"{cn}|A1_cost"] = COST[0] + fl * COST[-1]
        for full in (False, True):
            Fo = feats(Po, Ho, full); sc = StandardScaler().fit(Fo[fp])
            Xf, Xv = torch.tensor(sc.transform(Fo[fp]), dtype=torch.float32), torch.tensor(sc.transform(Fo[ip]), dtype=torch.float32)
            Xt = {cn: torch.tensor(sc.transform(feats(T["Pt"], T["Ht"], full)), dtype=torch.float32) for cn, T in TST.items()}
            ErrT, PoT, yT = torch.tensor(err, dtype=torch.float32), torch.tensor(Po, dtype=torch.float32), torch.tensor(ytr)
            cT = torch.tensor(CT, dtype=torch.float32); idx = {"fit": torch.tensor(fp), "iv": torch.tensor(ip)}
            rows, store = [], {}
            for lam in ([0.0] if full else LAMS):
                def lh(lg, b, which, lam=lam):
                    ii = idx[which][b] if which == "fit" else idx[which]; pi = F.softmax(lg, 1); return (pi * (ErrT[ii] + lam * cT[None])).sum(1).mean()
                g = train_gate(Xf, Xv, lh, seed)
                with torch.no_grad(): av = g(Xv).argmax(1).numpy(); at = {cn: g(x).argmax(1).numpy() for cn, x in Xt.items()}
                accv = (Po[ip][np.arange(len(ip)), av].argmax(1) == yi).mean(); cv = COST.sum() if full else (COST[0] + np.where(av > 0, COST[av], 0)).mean()
                rows.append((accv, cv, lam, 0.0, lam)); store[lam] = at
            s = select(rows, budget=not full); nm = "A2f" if full else "A2h"; out[f"{nm}_sel"] = np.array(s[:4]); grids[nm] = np.array([r[:4] for r in rows])
            for cn, T in TST.items():
                at = store[s[4]][cn]; out[f"{cn}|{nm}_action"] = at; out[f"{cn}|{nm}_p"] = T["Pt"][np.arange(T["n"]), at].astype(np.float32)
                out[f"{cn}|{nm}_cost"] = np.full(T["n"], COST.sum()) if full else COST[0] + np.where(at > 0, COST[at], 0)
            rows, store = [], {}
            for lam in ([0.0] if full else LAMS):
                for beta in BETAS:
                    def ls(lg, b, which, lam=lam, beta=beta):
                        ii = idx[which][b] if which == "fit" else idx[which]; w = F.softmax(lg, 1)
                        p = (w[:, :, None] * PoT[ii]).sum(1); ce = -torch.log(p.gather(1, yT[ii][:, None])[:, 0] + 1e-8).mean()
                        return ce + lam * (w * cT[None]).sum(1).mean() + beta * (-(w * torch.log(w + 1e-12)).sum(1)).mean()
                    g = train_gate(Xf, Xv, ls, seed)
                    with torch.no_grad(): wv = F.softmax(g(Xv), 1).numpy(); wt = {cn: F.softmax(g(x), 1).numpy() for cn, x in Xt.items()}
                    pv, cv = soft_out(wv, Po[ip], trunc=not full)
                    rows.append(((pv.argmax(1) == yi).mean(), cv.mean(), lam, beta, (lam, beta))); store[(lam, beta)] = wt
            s = select(rows, budget=not full); nm = "A3f" if full else "A3h"; out[f"{nm}_sel"] = np.array(s[:4]); grids[nm] = np.array([r[:4] for r in rows])
            for cn, T in TST.items():
                wt = store[s[4]][cn]; pt, ct = soft_out(wt, T["Pt"], trunc=not full); out[f"{cn}|{nm}_w"] = wt.astype(np.float32); out[f"{cn}|{nm}_p"] = pt.astype(np.float32); out[f"{cn}|{nm}_cost"] = ct
                if not full: out[f"{cn}|A3h_cost_all"] = np.full(T["n"], COST.sum()); out[f"{cn}|A3h_p_all"] = np.einsum("nk,nkc->nc", wt, T["Pt"]).astype(np.float32)
        Lo = Zo.reshape(len(tr), -1); sc = StandardScaler().fit(Lo[fp]); rows, store = [], {}
        for c in CS:
            m = LogisticRegression(C=c, max_iter=500).fit(sc.transform(Lo[fp]), ytr[fp]); rows.append((m.score(sc.transform(Lo[ip]), yi), COST.sum(), c, 0.0, c)); store[c] = m
        s = select(rows, budget=False); out["A5_sel"] = np.array(s[:4]); grids["A5"] = np.array([r[:4] for r in rows])
        for cn, T in TST.items():
            out[f"{cn}|A5_p"] = store[s[4]].predict_proba(sc.transform(T["Zt"].reshape(T["n"], -1))).astype(np.float32); out[f"{cn}|A5_cost"] = np.full(T["n"], COST.sum())
        for k, v in grids.items(): out[f"grid_{k}"] = v
        np.savez_compressed(fo, **out)
        print(f"{bed} s{seed} f{fold}: {time.time()-t0:.0f}s best single {EXP[kb]} oof acc {oof_acc.max():.4f}; iv sel " + " ".join(f"{n}={out[n+'_sel'][0]:.3f}" for n in ["A1", "A2h", "A2f", "A3h", "A3f", "A5"]), flush=True)
