"""T7 head jobs (ANALYSIS_PLAN_T7.md section 4). One (seed, fold) per file. Reads no test label. Writes test logits of E1, F and every masked head,
on the clean test rows and on the field-mix test rows (heads are fitted on clean training embeddings only).
    python t7_jobs.py --run RUN --pilot PILOT --t2b T2B --seeds 0 1 2 --folds 1 ... [--refs-only]"""
import argparse, copy, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F_
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.preprocessing import StandardScaler
from common import Q, S2, S3B, EMB, load_table, softmax
from heads import fit_head, macro_f1, NCLS, RC, HIDDEN, DROP
from QWave.qmoe_layers import BitNetExpert
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/redesign/idea6/stage2"))
from audio_shift import FIELD_CELLS, FIELD_SEEDS, cond_name, field_assignment

MASKS = [f"{s}{r}" for r in (25, 50) for s in ("energy", "random", "crop")]
ALPHAS = (0.5, 1.0, 2.0)


def fit_head_kd(Xfit, yfit, wfit, seed, Xsel, ysel, Tfit, alpha, metric="acc"):
    """fit_head (heads.py) with an added alpha * KL(p_teacher || p_student) per sample, same weights; teacher probabilities Tfit [n, C] are detached targets."""
    from heads import Head
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    sc = StandardScaler().fit(Xfit)
    X = torch.tensor(sc.transform(Xfit), dtype=torch.float32); y = torch.tensor(yfit, dtype=torch.long); w = torch.tensor(wfit, dtype=torch.float32)
    Tt = torch.tensor(Tfit, dtype=torch.float32)
    cnt = np.bincount(yfit, minlength=NCLS).astype(float); cw = torch.tensor(1.0 / np.maximum(cnt, 1.0), dtype=torch.float32)
    model = BitNetExpert(X.shape[1], NCLS, HIDDEN, DROP, num_bits=32)
    opt = torch.optim.AdamW(model.parameters(), lr=RC.lr, weight_decay=RC.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=RC.lr_factor, patience=RC.lr_patience)
    Xs = torch.tensor(sc.transform(Xsel), dtype=torch.float32)
    best, best_state, bad, n, bs = -1.0, None, 0, len(X), int(RC.batch_size)
    for ep in range(int(RC.max_epochs)):
        model.train(); perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; out = model(X[b])
            ce = F_.cross_entropy(out, y[b], reduction="none", label_smoothing=float(RC.label_smoothing))
            kl = (Tt[b] * (torch.log(Tt[b].clamp_min(1e-12)) - F_.log_softmax(out, 1))).sum(1)
            den = (w[b] * cw[y[b]]).sum()
            loss = (w[b] * cw[y[b]] * (ce + alpha * kl)).sum() / den
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(RC.grad_clip_norm)); opt.step()
        model.eval()
        with torch.no_grad(): pr = model(Xs).argmax(1).numpy()
        m = macro_f1(ysel, pr) if metric == "f1" else float((pr == ysel).mean())
        sched.step(m)
        if m > best: best, bad, best_state = m, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= int(RC.patience): break
    model.load_state_dict(best_state)
    h = Head(model, sc); h.epochs_run = ep + 1; h.best_sel = best
    return h


ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, type=Path); ap.add_argument("--pilot", required=True, type=Path); ap.add_argument("--t2b", required=True, type=Path)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--refs-only", action="store_true"); ap.add_argument("--no-field", action="store_true")
a = ap.parse_args(); torch.set_num_threads(a.threads)

d = load_table(); y = d.label.values.astype(np.int64); N = len(d)
assign = np.array(field_assignment(N, FIELD_SEEDS["test"]))
EM = a.run / "emb/urban8k"
CELLS = [cond_name(k, s) for k, s in FIELD_CELLS]


def mixed(loader):
    """features of every clip under its field-mix condition (clean for half of the clips)."""
    out = None
    for c in ["clean"] + CELLS:
        rows = np.flatnonzero(assign == c)
        if len(rows) == 0: continue
        X = loader(c)
        if out is None: out = np.zeros((N, X.shape[1]), np.float32)
        out[rows] = X[rows]
    assert np.isfinite(out).all()
    return out


X1 = np.load(EMB["mn10_w8a8"]).astype(np.float32)
XF = np.load(Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/emb_ast_short/urban8k/w4g128__clean.npy").astype(np.float32)
S2D = Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
XM = {} if a.refs_only else {m: np.load(EM / f"w4g128__{m}__clean.npy").astype(np.float32) for m in MASKS}
if a.no_field:
    X1f = XFf = None; XMf = {}
else:
    X1f = mixed(lambda c: np.load(S3B / f"emb_cnn/urban8k/mn10_w8a8__{c}.npy").astype(np.float32))
    XFf = mixed(lambda c: np.load(S2D / f"emb_ast_short/urban8k/w4g128__{c}.npy").astype(np.float32))
    XMf = {} if a.refs_only else {m: mixed(lambda c, m=m: np.load(EM / f"w4g128__{m}__{c}.npy").astype(np.float32)) for m in MASKS}
(a.run / "raw").mkdir(parents=True, exist_ok=True)

for seed in a.seeds:
    for fold in a.folds:
        f_out = a.run / "raw" / f"s{seed}_f{fold}{'_refs' if a.refs_only else ''}.npz"
        if f_out.exists(): continue
        t0 = time.time(); P = np.load(a.pilot / f"t1/raw/s{seed}_f{fold}.npz"); T = np.load(a.t2b / f"urban/raw/s{seed}_f{fold}.npz")
        tr, te, S2m, fit_p, iv_p = P["tr"], P["te"], P["S2"], P["fit_p"], P["iv_p"]
        assert (T["tr"] == tr).all() and (T["te"] == te).all()
        ytr = y[tr]; sel_p = iv_p[S2m[iv_p]]; ones = np.ones(len(tr))
        R = dict(te=te, flag=P["flag"], tau1=P["tau1"], z1_te_pilot=P["z1_te"], seed=seed, fold=fold, cond_te=assign[te])
        # E1: the pilot's head, retrained (macro F1 on the whole inner validation), applied to clean and field-mix test rows
        fit_g, iv_g = tr[fit_p], tr[iv_p]
        h1 = fit_head(X1[fit_g], y[fit_g], np.ones(len(fit_g)), seed, X1[iv_g], y[iv_g], metric="f1")
        R["E1_te"] = h1.logits(X1[te]).astype(np.float32)
        if X1f is not None: R["E1_fld"] = h1.logits(X1f[te]).astype(np.float32)
        def prim(X, Xf, name, metric="acc"):
            """T1b recipe: fit on fit_p, select on the inner-validation clips E1's out-of-fold margin flags (metric acc) or on the whole inner validation (metric f1, LIB)."""
            sp = sel_p if metric == "acc" else iv_p
            h = fit_head(X[tr][fit_p], ytr[fit_p], ones[fit_p], seed, X[tr][sp], ytr[sp], metric=metric)
            R[name + "_te"] = h.logits(X[te]).astype(np.float32); R[name + "_ep"] = h.epochs_run
            if Xf is not None: R[name + "_fld"] = h.logits(Xf[te]).astype(np.float32)
            return h
        prim(XF, XFf, "F")
        R["F_te_t1b"] = np.load(Q / "2026-10-24_2208_moe-t1b_code-4e3e036" / f"urban/raw_strength/s{seed}_f{fold}.npz")["E2c_A_U_te"]
        if not a.refs_only:
            prim(XF, XFf, "LIBF", "f1")
            teach = softmax(T["E2c_oof"].astype(np.float64)).astype(np.float32)          # out-of-fold teacher on the training clips, tr order
            for m in MASKS:
                Xm, Xmf = XM[m], XMf.get(m) if XMf else None
                prim(Xm, Xmf, "CE_" + m); prim(Xm, Xmf, "LIB_" + m, "f1")
                best = None
                for al in ALPHAS:
                    hk = fit_head_kd(Xm[tr][fit_p], ytr[fit_p], ones[fit_p], seed, Xm[tr][sel_p], ytr[sel_p], teach[fit_p], al)
                    if best is None or hk.best_sel > best[0].best_sel: best = (hk, al)
                hk, al = best
                R["KD_" + m + "_te"] = hk.logits(Xm[te]).astype(np.float32); R["KD_" + m + "_alpha"] = al; R["KD_" + m + "_ep"] = hk.epochs_run
                if Xmf is not None: R["KD_" + m + "_fld"] = hk.logits(Xmf[te]).astype(np.float32)
        np.savez_compressed(f_out, **R)
        r1 = float(np.abs(R["F_te"] - R["F_te_t1b"]).max()); r2 = float(np.abs(R["E1_te"] - R["z1_te_pilot"]).max())
        print(f"s{seed} f{fold}: done {time.time()-t0:.0f}s  R1 max|F - T1b E2c| {r1:.2e}  R2 max|E1 - pilot| {r2:.2e}", flush=True)
