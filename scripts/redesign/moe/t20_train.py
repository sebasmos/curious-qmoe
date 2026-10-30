"""T20 trainer (ANALYSIS_PLAN_T20.md 3.3): the uniform arm of T17 part (c) (`t17c_train.py`) for any task, fold and length.
LoRA rank 16 on the six Linears of blocks 10 and 11, final LayerNorm and a new linear head trainable, blocks 0 to 9 frozen and run on the
cached filter bank every step; SpecAugment and mixup on the filter bank; class-weighted CE mixed with KL distillation from the cross-fitted
FP32 AST head (T1c `E2a_oof`). Checkpoint = best macro F1 on all clean inner-validation clips (stored grouped split). Reads training-fold
labels only. Several seeds run one after another in one process.
    python t20_train.py --task urban --fold 1 --seeds 0 1 2 --lr_body 1e-4 --out RUNS_DIR [--tag lr1e-4] [--device mps]"""
import argparse, json, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--steps", type=int, default=300); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--eval_every", type=int, default=50); ap.add_argument("--k", type=int, default=2)
ap.add_argument("--lr_body", type=float, default=1e-4); ap.add_argument("--lr_head", type=float, default=1e-3)
ap.add_argument("--freqm", type=int, default=24); ap.add_argument("--timem", type=int, default=60); ap.add_argument("--mix_p", type=float, default=0.5); ap.add_argument("--mix_beta", type=float, default=0.4)
ap.add_argument("--kd_T", type=float, default=2.0); ap.add_argument("--kd_alpha", type=float, default=0.5)
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--tag", default=""); ap.add_argument("--device", default="mps")
a = ap.parse_args()
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t17c_ast import load_ast, front, Tail, NL
from t20_common import *
torch.set_num_threads(2); dev = torch.device(a.device); B = NL - a.k; T = TASKS[a.task]; L, K = T["L"], T["K"]
D = bed(a.task); y = D["y"]; te = test_rows(a.task, D, a.fold); tr = train_rows(a.task, D, a.fold)
FBr = np.load(CACHE / a.task / "clean_rows.npy"); FB = np.lib.format.open_memmap(CACHE / a.task / "clean_fbank.npy", mode="r"); cpos = {int(r): i for i, r in enumerate(FBr)}
assert FB.shape[1:] == (L, 128)
a.out.mkdir(parents=True, exist_ok=True)
ext, model = load_ast(L, dev)
for seed in a.seeds:
    tag = f"f{a.fold}_s{seed}" + (f"_{a.tag}" if a.tag else "")
    if (a.out / f"{tag}.json").exists(): print("skip", tag, flush=True); continue
    log = open(a.out / f"{tag}.log", "w")
    def say(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    iv = inner_split(a.task, seed, a.fold); assert np.isin(iv, tr).all(), "inner validation outside the training rows"
    fit = tr[~np.isin(tr, iv)]; assert not np.isin(fit, te).any() and not np.isin(iv, te).any(), "test row in training"
    teach_all = teacher(a.task, seed, a.fold, tr); tpos = {int(r): i for i, r in enumerate(tr)}
    teach = torch.tensor(teach_all[[tpos[int(r)] for r in fit]], device=dev)
    fit_c = np.array([cpos[int(r)] for r in fit]); iv_c = np.array([cpos[int(r)] for r in iv])
    torch.manual_seed(seed); tail = Tail(model, B, "lora", ncls=K)
    opt = torch.optim.AdamW(tail.param_groups(a.lr_body, a.lr_head), weight_decay=0.01)
    warm = max(1, int(0.05 * a.steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (a.steps - s) / (a.steps - warm)))
    cnt = np.bincount(y[fit], minlength=K).astype(float); cw = torch.tensor(1.0 / np.maximum(cnt, 1.0), dtype=torch.float32, device=dev)
    w = torch.ones(len(fit), dtype=torch.float32)
    gen = torch.Generator().manual_seed(1000 + seed); rng = np.random.default_rng(2000 + seed); yfit = torch.tensor(y[fit], device=dev)
    R = dict(task=a.task, fold=a.fold, seed=seed, steps=a.steps, k=a.k, L=L, n_fit=len(fit), n_iv=len(iv), n_te=len(te), args={kk: str(v) for kk, v in vars(a).items()})
    def specaug(x):
        for i in range(x.shape[0]):
            f = int(rng.integers(0, a.freqm + 1)); t = int(rng.integers(0, a.timem + 1))
            if f: f0 = int(rng.integers(0, 128 - f + 1)); x[i, :, f0:f0 + f] = 0.0
            if t: t0_ = int(rng.integers(0, L - t + 1)); x[i, t0_:t0_ + t, :] = 0.0
        return x
    @torch.no_grad()
    def logits_iv():
        tail.eval(); out = []
        for s in range(0, len(iv_c), 32):
            c = iv_c[s:s + 32]; o = np.argsort(c); inv = np.argsort(o)
            x = torch.from_numpy(np.asarray(FB[c[o]], np.float32)).to(dev)
            out.append(tail(front(model, x, B)).cpu().numpy()[inv])
        return np.concatenate(out)
    best, best_state, best_step, hist, t0 = -1.0, None, 0, [], time.time()
    for step in range(a.steps):
        tail.train(); ip = torch.multinomial(w, a.bs, replacement=False, generator=gen).numpy(); cp = fit_c[ip]
        srt = np.argsort(cp); inv = np.argsort(srt)
        x = torch.from_numpy(np.asarray(FB[cp[srt]], np.float32)).to(dev)[torch.from_numpy(inv).to(dev)]
        ya, tl = yfit[ip], teach[ip]; yb, tlb, lam = ya, tl, 1.0
        if rng.random() < a.mix_p:
            lam = float(rng.beta(a.mix_beta, a.mix_beta)); perm = torch.randperm(a.bs, generator=gen).to(dev); x = lam * x + (1 - lam) * x[perm]; yb, tlb = ya[perm], tl[perm]
        x = specaug(x)
        logit = tail(front(model, x, B))
        ce = lam * F.cross_entropy(logit, ya, reduction="none") + (1 - lam) * F.cross_entropy(logit, yb, reduction="none"); wt = lam * cw[ya] + (1 - lam) * cw[yb]
        hard = (wt * ce).sum() / wt.sum()
        pt = lam * F.softmax(tl / a.kd_T, 1) + (1 - lam) * F.softmax(tlb / a.kd_T, 1)
        soft = ((pt * (torch.log(pt.clamp_min(1e-12)) - F.log_softmax(logit / a.kd_T, 1))).sum(1) * a.kd_T ** 2).mean()
        loss = (1 - a.kd_alpha) * hard + a.kd_alpha * soft
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step(); sched.step()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            z = logits_iv(); f1 = macro_f1(y[iv], z.argmax(1), K); acc = float((z.argmax(1) == y[iv]).mean())
            hist.append(dict(step=step + 1, loss=float(loss), hard=float(hard), soft=float(soft), iv_f1=f1, iv_acc=acc, seconds=time.time() - t0)); say(hist[-1])
            if f1 > best: best, best_step, best_state, best_z = f1, step + 1, {kk: v.detach().clone() for kk, v in tail.state_dict().items()}, z
    keep = {kk: v.cpu() for kk, v in best_state.items() if kk.endswith(".A") or kk.endswith(".Bm") or kk.startswith("ln.") or kk.startswith("head.")}
    torch.save(keep, a.out / f"{tag}_tail.pt"); np.savez(a.out / f"{tag}_iv.npz", iv=iv, z=best_z)
    R.update(hist=hist, best_iv_f1=best, best_step=best_step, train_seconds=time.time() - t0, n_saved_tensors=len(keep))
    (a.out / f"{tag}.json").write_text(json.dumps(R, indent=1)); say("done", tag, "best", best, "at", best_step, "%.0fs" % R["train_seconds"]); log.close()
    del tail, opt, best_state; torch.mps.empty_cache() if a.device == "mps" else None
