"""T22 trainer (ANALYSIS_PLAN.md C2): fine-tune EfficientAT MN10 (AudioSet start) on the clean coarse view for one outer fold and
several seeds, one after another in one process. Reads the clean coarse-view cache and training-fold labels only; checkpoint = best
inner-validation macro F1 (FP32 model) on the stored grouped inner split of that seed.
    python t22_train.py --task esc50 --fold 1 --seeds 0 1 2 --lr 3e-4 --out RUNS [--tag lr3e-4] [--device mps] [--max_epochs N]"""
import argparse, json, math, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--tag", default=""); ap.add_argument("--device", default="mps")
ap.add_argument("--max_epochs", type=int, default=None); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--start_by_min", type=float, default=40.0, help="start a new seed only if the process is younger than this (GPU turn of at most about 80 min)")
a = ap.parse_args()
import os as _os, time as _t; T_START = float(_os.environ.get("T22_TURN_START", _t.time()))  # the GPU turn start, shared by the processes of one turn
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t22_common import *  # noqa
import t22_view as V
from QWave import efficientat_mn as M
torch.set_num_threads(2); dev = torch.device(a.device)
T = TASKS[a.task]; K = T["K"]; E = a.max_epochs or T["epochs"]
B = bed(a.task); y = B["y"]; te = np.flatnonzero(B["fold"] == a.fold); tr = train_rows(a.task, B, a.fold)
d = CACHE / a.task; X = np.load(d / "clean_mel.npy", mmap_mode="r"); crow = np.load(d / "clean_rows.npy"); cpos = {int(r): i for i, r in enumerate(crow)}
FR = X.shape[2]
a.out.mkdir(parents=True, exist_ok=True)


def fetch(rows):
    return np.stack([X[cpos[int(r)]] for r in rows])  # [b, 64, FR]


for seed in a.seeds:
    tag = f"{a.task}_f{a.fold}_s{seed}" + (f"_{a.tag}" if a.tag else "")
    if (a.out / f"{tag}.json").exists():
        print("skip", tag, flush=True); continue
    if (_t.time() - T_START) / 60 > a.start_by_min:
        print("deferred", tag, "(turn time)", flush=True); continue
    iv = inner_split(a.task, seed, a.fold)
    assert np.isin(iv, tr).all(), "inner validation outside the training rows"
    fit = tr[~np.isin(tr, iv)]
    assert not np.isin(fit, te).any() and not np.isin(iv, te).any(), "test row in training"
    torch.manual_seed(seed)
    model = V.Coarse(M.load_mn("mn10", WEIGHTS), K).to(dev)
    opt = torch.optim.AdamW([{"params": model.features.parameters(), "lr": a.lr}, {"params": model.fc.parameters(), "lr": 10 * a.lr}], weight_decay=0.01)
    spe = len(fit) // a.bs; total = spe * E; warm = spe
    base_lrs = [g["lr"] for g in opt.param_groups]
    cnt = np.bincount(y[fit], minlength=K).astype(float); cw = torch.tensor(cnt.sum() / (K * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    rng = np.random.default_rng(2000 + seed); order_rng = np.random.default_rng(1000 + seed)
    Xiv = torch.from_numpy(fetch(iv)).unsqueeze(1)

    @torch.no_grad()
    def iv_f1():
        model.eval(); out = []
        n = len(iv); pad = (-n) % a.bs
        idx = np.concatenate([np.arange(n), np.zeros(pad, int)])
        for s in range(0, len(idx), a.bs):
            out.append(model(Xiv[idx[s:s + a.bs]].to(dev)).float().cpu().numpy())
        lg = np.concatenate(out)[:n]
        return macro_f1(y[iv], lg.argmax(1), K)

    best, best_ep, best_state, hist, t0, step, bad = -1.0, -1, None, [], time.time(), 0, 0
    for ep in range(E):
        model.train(); perm = order_rng.permutation(len(fit)); tl = 0.0
        for b in range(spe):
            rows = fit[perm[b * a.bs:(b + 1) * a.bs]]
            x = fetch(rows)
            for i in range(len(x)):  # SpecAugment: one frequency mask (0..8 of 64 bands), one time mask (0..10% of frames), set to 0
                f = int(rng.integers(0, 9)); t = int(rng.integers(0, int(0.1 * FR) + 1))
                if f: f0 = int(rng.integers(0, V.N_MELS - f + 1)); x[i, f0:f0 + f, :] = 0.0
                if t: t0_ = int(rng.integers(0, FR - t + 1)); x[i, :, t0_:t0_ + t] = 0.0
            xb = torch.from_numpy(x).unsqueeze(1).to(dev); yb = torch.from_numpy(y[rows]).to(dev)
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * ((step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            if rng.random() < 0.5:
                lam = float(rng.beta(0.3, 0.3)); pm = torch.from_numpy(rng.permutation(len(rows))).to(dev)
                lo = model(lam * xb + (1 - lam) * xb[pm])
                loss = lam * F.cross_entropy(lo, yb, weight=cw, label_smoothing=0.1) + (1 - lam) * F.cross_entropy(lo, yb[pm], weight=cw, label_smoothing=0.1)
            else:
                loss = F.cross_entropy(model(xb), yb, weight=cw, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1; tl += float(loss)
        f1 = iv_f1(); hist.append({"epoch": ep, "loss": tl / spe, "iv_f1": f1, "sec": time.time() - t0})
        print(f"{tag} ep {ep} loss {tl / spe:.4f} iv_f1 {f1:.4f} {time.time() - t0:.0f}s", flush=True)
        if f1 > best:
            best, best_ep, bad = f1, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    torch.save(best_state, a.out / f"{tag}.pt")
    (a.out / f"{tag}.json").write_text(json.dumps(dict(task=a.task, fold=a.fold, seed=seed, lr=a.lr, head_lr=10 * a.lr, epochs_max=E, best_epoch=best_ep,
        best_iv_f1_fp32=best, n_fit=len(fit), n_iv=len(iv), n_test=len(te), steps_per_epoch=spe, seconds=time.time() - t0, hist=hist,
        args={k: str(v) for k, v in vars(a).items()}), indent=1))
    if dev.type == "mps":
        torch.mps.empty_cache()
