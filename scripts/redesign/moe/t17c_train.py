"""T17(c) trainer: AST adapter expert on ALL UrbanSound8K fold-1 training clips (ANALYSIS_PLAN_T17.md section 8 and its amendment). LoRA rank 16 on the last two blocks (+ final LayerNorm + head),
the first ten blocks frozen, NOT cached so SpecAugment and mixup act on the filter bank. Both arms: SpecAugment, mixup, distillation from the FP32 AST head (cross-fitted `E2a_oof`).
Arms: twin (uniform sampling), spec (sampling weight 1.0 on the clips the cross-fitted cheap model misses, 0.25 on the others). Training and checkpoint selection read training-fold labels only.
    python t17c_train.py --arm twin|spec --seed 0 --steps 600 --fb DIR --cache T12CACHEDIR --out DIR [--probe] [--device mps|cpu]"""
import argparse, json, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["twin", "spec"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--eval_every", type=int, default=100); ap.add_argument("--k", type=int, default=2)
ap.add_argument("--lr_body", type=float, default=1e-4); ap.add_argument("--lr_head", type=float, default=1e-3); ap.add_argument("--w_other", type=float, default=0.25)
ap.add_argument("--freqm", type=int, default=24); ap.add_argument("--timem", type=int, default=60); ap.add_argument("--mix_p", type=float, default=0.5); ap.add_argument("--mix_beta", type=float, default=0.4)
ap.add_argument("--kd_T", type=float, default=2.0); ap.add_argument("--kd_alpha", type=float, default=0.5)
ap.add_argument("--fb", type=Path, required=True); ap.add_argument("--cache", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--probe", action="store_true"); ap.add_argument("--device", default="mps")
a = ap.parse_args()
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t17c_ast import *
from t17_common import T1C, load_bed
torch.set_num_threads(2); dev = torch.device(a.device); B = NL - a.k
S = np.load(T1C / f"urban/raw_strength/s{a.seed}_f1.npz"); O = np.load(T1C / f"urban/raw_oof/s{a.seed}_f1.npz")
tr, te, fit_p, iv_p, S2m = S["tr"], S["te"], S["fit_p"], S["iv_p"], S["S2"]; y = load_bed("urban")["y"]
sel_p = iv_p[S2m[iv_p]]; K = 10; fit_g, iv_g, sel_g = tr[fit_p], tr[iv_p], tr[sel_p]
teach = torch.tensor(O["E2a_oof"].astype(np.float32), device=dev)          # [len(tr), K], cross-fitted FP32 AST head logits, row order = tr
FBr = np.load(a.fb / "train_fold1_rows.npy"); assert (FBr[:len(tr)] == tr[:len(FBr)]).all()
FB = np.lib.format.open_memmap(a.fb / "train_fold1_fbank.npy", mode="r")
a.out.mkdir(parents=True, exist_ok=True); tag = f"{a.arm}_s{a.seed}_n{a.steps}" + ("_probe_lb%g" % a.lr_body if a.probe else ""); log = open(a.out / f"{tag}.log", "w")
def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
torch.manual_seed(a.seed); ext, model = load_ast(400, dev); tail = Tail(model, B, "lora")
opt = torch.optim.AdamW(tail.param_groups(a.lr_body, a.lr_head), weight_decay=0.01)
warm = max(1, int(0.05 * a.steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (a.steps - s) / (a.steps - warm)))
cnt = np.bincount(y[fit_g], minlength=K).astype(float); cw = torch.tensor(1.0 / cnt, dtype=torch.float32, device=dev)
w = torch.tensor(np.where(S2m[fit_p], 1.0, a.w_other) if a.arm == "spec" else np.ones(len(fit_g)), dtype=torch.float32)
gen = torch.Generator().manual_seed(1000 + a.seed); rng = np.random.default_rng(2000 + a.seed); ytall = torch.tensor(y[tr], device=dev)
R = dict(arm=a.arm, seed=a.seed, steps=a.steps, k=a.k, n_fit=len(fit_g), n_fit_flagged=int(S2m[fit_p].sum()), n_sel=len(sel_g), n_te=len(te), args={k: str(v) for k, v in vars(a).items()})
def specaug(x):
    n = x.shape[0]
    for i in range(n):
        f = int(rng.integers(0, a.freqm + 1)); t = int(rng.integers(0, a.timem + 1))
        if f: f0 = int(rng.integers(0, 128 - f + 1)); x[i, :, f0:f0 + f] = 0.0
        if t: t0_ = int(rng.integers(0, 400 - t + 1)); x[i, t0_:t0_ + t, :] = 0.0
    return x
@torch.no_grad()
def logits_fb(pos):          # clean filter banks of training-set positions (positions in tr)
    tail.eval(); out = []
    for s in range(0, len(pos), 32):
        x = torch.from_numpy(np.asarray(FB[np.sort(pos[s:s + 32])], np.float32)).to(dev); o = np.argsort(np.argsort(pos[s:s + 32]))
        out.append(tail(front(model, x, B)).cpu().numpy()[o])
    return np.concatenate(out)
@torch.no_grad()
def logits_tok(Cc, rows):
    tail.eval(); out = []
    for s in range(0, len(rows), 64): out.append(tail(Cc.get(rows[s:s + 64], dev)).cpu().numpy())
    return np.concatenate(out)
best, best_state, hist, t0 = -1.0, None, [], time.time(); sel_pos = sel_p                     # positions in tr of the flagged inner-validation clips
for step in range(a.steps):
    tail.train(); ip = torch.multinomial(w, a.bs, replacement=False, generator=gen).numpy(); pos = fit_p[ip]          # positions in tr
    srt = np.argsort(pos); inv = np.argsort(srt)
    x = torch.from_numpy(np.asarray(FB[pos[srt]], np.float32)).to(dev)[torch.from_numpy(inv).to(dev)]
    ya, tl = ytall[pos], teach[pos]; yb, tlb, lam = ya, tl, 1.0
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
        zs = logits_fb(sel_pos); acc_s = float((zs.argmax(1) == y[tr[sel_pos]]).mean())
        hist.append(dict(step=step + 1, loss=float(loss), hard=float(hard), soft=float(soft), val_flagged_acc=acc_s, seconds=time.time() - t0)); say(hist[-1])
        if acc_s > best: best, best_step, best_state = acc_s, step + 1, {k: v.detach().clone() for k, v in tail.state_dict().items()}
R.update(hist=hist, best_val_flagged_acc=best, best_step=best_step, train_seconds=time.time() - t0)
tail.load_state_dict(best_state); Z = {"val_sel": logits_fb(sel_pos), "val_all": logits_fb(iv_p)}
if not a.probe:
    C = TokCache(a.cache, "clean")
    for cn in ("clean", "bg0", "field"):
        Ct = C if cn == "clean" else TokCache(a.cache, f"fold1_{cn}"); Z[f"test_{cn}"] = logits_tok(Ct, te)
np.savez(a.out / f"{tag}.npz", te=te, **Z); (a.out / f"{tag}.json").write_text(json.dumps(R, indent=1)); say("done", R["train_seconds"])
