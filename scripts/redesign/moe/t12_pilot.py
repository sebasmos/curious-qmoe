"""T12 micro-pilot trainer (ANALYSIS_PLAN_T12.md). UrbanSound8K fold 1 test, seed s. Arms: twin (uniform sampling), spec (flagged clips weight 1, others 0.1), frozen (repo head on frozen FP32@400 embeddings).
Training and checkpoint selection read training-fold labels only (inner split of the pilot, flagged = out-of-fold E1 margin below tau1_oof). With --probe no test cache is touched.
    python t12_pilot.py --arm twin|spec|frozen --seed 0 --steps 444 --cache DIR --out DIR [--probe]"""
import argparse, json, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["twin", "spec", "frozen"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=222); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--eval_every", type=int, default=55); ap.add_argument("--k", type=int, default=2)
ap.add_argument("--lr_body", type=float, default=3e-5); ap.add_argument("--lr_head", type=float, default=1e-3); ap.add_argument("--eps", type=float, default=0.10); ap.add_argument("--mode", default="lora"); ap.add_argument("--keep", type=float, default=1.0)
ap.add_argument("--cache", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--probe", action="store_true"); ap.add_argument("--device", default="mps")
a = ap.parse_args()
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t12_common import *
from t1c_common import PILOT, Q, T1B, load_bed
from heads import fit_head
torch.set_num_threads(2); dev = torch.device(a.device); fold = 1; B = NL - a.k
P = np.load(PILOT / f"t1/raw/s{a.seed}_f{fold}.npz"); tr, te, fit_p, iv_p, S2m = P["tr"], P["te"], P["fit_p"], P["iv_p"], P["S2"]
y = load_bed("urban")["y"]; sel_p = iv_p[S2m[iv_p]]; fit_g, iv_g, sel_g = tr[fit_p], tr[iv_p], tr[sel_p]
a.out.mkdir(parents=True, exist_ok=True); tag = f"{a.arm}_k{a.k}_s{a.seed}_n{a.steps}" + (f"_{a.mode}_lb{a.lr_body}_lh{a.lr_head}_keep{a.keep}" if a.probe else "")
C = TokCache(a.cache, "clean"); log = open(a.out / f"{tag}.log", "w")
def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
R = dict(arm=a.arm, seed=a.seed, steps=a.steps, k=a.k, n_fit=len(fit_g), n_fit_flagged=int(S2m[fit_p].sum()), n_sel=len(sel_g), n_te=len(te))
if a.arm == "frozen":
    Xe = C.emb; ix = {int(r): i for i, r in enumerate(C.rows)}; g = lambda rows: Xe[[ix[int(r)] for r in rows]]
    h = fit_head(g(fit_g), y[fit_g], np.ones(len(fit_g)), a.seed, g(sel_g), y[sel_g], metric="acc", ncls=10)
    R["epochs"] = h.epochs_run; Z = {"val_sel": h.logits(g(sel_g)), "val_all": h.logits(g(iv_g))}
    R["val_sel_acc"] = float((Z["val_sel"].argmax(1) == y[sel_g]).mean())
    if not a.probe:
        for cn in ("clean", "bg0", "field"):
            Ct = C if cn == "clean" else TokCache(a.cache, f"fold1_{cn}"); Z[f"test_{cn}"] = h.logits(Ct.emb[[Ct.pos[int(r)] for r in te]])
    np.savez(a.out / f"{tag}.npz", te=te, **Z); (a.out / f"{tag}.json").write_text(json.dumps(R, indent=1)); say(R); sys.exit(0)

torch.manual_seed(a.seed); tail = Tail(load_ast(400, dev)[1], B, a.mode)
n_tr = sum(p.numel() for p in tail.parameters() if p.requires_grad); R["trainable_params"] = n_tr
opt = torch.optim.AdamW(tail.param_groups(a.lr_body, a.lr_head), weight_decay=0.01)
warm = max(1, int(0.05 * a.steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (a.steps - s) / (a.steps - warm)))
cnt = np.bincount(y[fit_g], minlength=10).astype(float); cw = torch.tensor(1.0 / cnt, dtype=torch.float32, device=dev)
w = torch.tensor(np.where(S2m[fit_p], 1.0, a.eps) if a.arm == "spec" else np.ones(len(fit_g)), dtype=torch.float32)
gen = torch.Generator().manual_seed(1000 + a.seed); yt = torch.tensor(y[fit_g], device=dev)


@torch.no_grad()
def logits_of(Cc, rows):
    tail.eval(); out = []
    for s in range(0, len(rows), 64): out.append(tail(Cc.get(rows[s:s + 64], dev)).cpu().numpy())
    return np.concatenate(out)


best, best_state, hist, t0 = -1.0, None, [], time.time()
for step in range(a.steps):
    tail.train(); idx = torch.multinomial(w, a.bs, replacement=False, generator=gen).numpy()
    x = C.get(fit_g[idx], dev); yb = yt[idx]
    if a.keep < 1.0:      # training-time token dropout (per clip random patch subset, CLS and DIST kept); evaluation uses all tokens
        m = int(round(a.keep * (x.shape[1] - 2))); sel = torch.rand(x.shape[0], x.shape[1] - 2, device=dev).argsort(1)[:, :m] + 2
        x = torch.cat([x[:, :2], torch.gather(x, 1, sel[:, :, None].expand(-1, -1, x.shape[2]))], 1)
    ce = F.cross_entropy(tail(x), yb, reduction="none", label_smoothing=0.0); loss = (cw[yb] * ce).sum() / cw[yb].sum()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step(); sched.step()
    if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
        zs, zv = logits_of(C, sel_g), logits_of(C, iv_g); acc_s, acc_v = float((zs.argmax(1) == y[sel_g]).mean()), float((zv.argmax(1) == y[iv_g]).mean())
        hist.append(dict(step=step + 1, loss=float(loss), val_flagged_acc=acc_s, val_all_acc=acc_v, seconds=time.time() - t0)); say(hist[-1])
        if acc_s > best: best, best_step, best_state = acc_s, step + 1, {k: v.detach().clone() for k, v in tail.state_dict().items()}
R.update(hist=hist, best_val_flagged_acc=best, best_step=best_step, train_seconds=time.time() - t0)
tail.load_state_dict(best_state); Z = {"val_sel": logits_of(C, sel_g), "val_all": logits_of(C, iv_g)}
if not a.probe:
    for cn in ("clean", "bg0", "field"):
        Ct = C if cn == "clean" else TokCache(a.cache, f"fold1_{cn}"); Z[f"test_{cn}"] = logits_of(Ct, te)
    torch.save({k: v.cpu() for k, v in tail.state_dict().items()}, a.out / f"{tag}_tail.pt")
np.savez(a.out / f"{tag}.npz", te=te, **Z); (a.out / f"{tag}.json").write_text(json.dumps(R, indent=1)); say("done", R["train_seconds"])
