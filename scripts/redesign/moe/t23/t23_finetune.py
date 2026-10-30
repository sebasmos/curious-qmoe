"""T23 fallback (ANALYSIS_PLAN.md C5), used only if the frozen expert fails P1: fine-tune conv_block6 and fc1 plus a linear classifier from the
AudioSet weights, on cached conv_block5 outputs of the frozen trunk (clean audio), per (fold, seed). Then writes, for that fold and seed, the FP32 clean
embedding of every cached row (fine-tuned tail on the FP32 trunk cache, for the protocol 3 head), the deployed W8A8 tail as int8 codes and
scales (tail_q8.pt, t23_tailq.py; amendment 1) and its training rows (train_rows.npy, indices into the metadata). Training rows: the fold's training clips minus the inner-validation split of the stored MN10-Q8 file of
the same seed (the protocol 3 split, checked equal to ours in P1); early stopping on that split's macro F1. Run with python -I.
    python -I t23_finetune.py --task esc50 --fold 1 --seeds 0 1 2 --out R/ft [--device mps]"""
import argparse, copy, json, os, sys, time, zlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--device", default="mps"); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--max-minutes", type=float, default=75.0); a = ap.parse_args(); T_START = time.time()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = str(a.threads)
for p_ in (ROOT, HERE.parent, HERE, ROOT / "scripts/redesign/idea6/stage2"): sys.path.insert(0, str(p_))
import numpy as np, pandas as pd, torch, torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from QWave import efficientat_mn as M  # noqa: E402
from QWave.panns_wavegram import QuantWavegram, load_wavegram  # noqa: E402
from t23_tailq import save_tail_q8  # noqa: E402
from audio_shift import index_wavs, load_clip  # noqa: E402
from t13_common import sdir, tau_file  # noqa: E402
DS = Path.home() / "Documents/DATASETS"; CKPT = Path.home() / "Documents/qmoe-weights/panns-wavegram/Wavegram_Cnn14_mAP=0.389.pth"
DATA = {"esc50": (DS / "ESC-50/ast_768/esc-50.csv", DS / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio", 80000, 50, "label"),
        "urban": (DS / "Urban8k/ast_768/urbansound8k.csv", DS / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio", 64587, 10, "class_id"),
        "tau": (DS / "TAU2020Mobile/ast_768/tau2020mobile.csv", DS / "TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968/TAU-urban-acoustic-scenes-2020-mobile-development/audio", 160000, 10, "class_id")}
LR_TAIL, LR_CLS, WD, BS, EPOCHS, PATIENCE, MIX_ALPHA = 1e-4, 1e-3, 1e-4, 64, 40, 6, 0.2
torch.set_num_threads(a.threads); dev = torch.device(a.device)
csv, aud, pad, K, ycol = DATA[a.task]; meta = pd.read_csv(csv)
rows = np.flatnonzero((meta.device == "a").to_numpy()) if a.task == "tau" else np.arange(len(meta))   # TAU: heads see device A only
SMOKE = int(os.environ.get("T23_SMOKE", "0"))  # code test only: first SMOKE rows, 2 epochs
if SMOKE: rows = rows[:SMOKE]; EPOCHS = 2
stems = meta["name"].map(lambda s: Path(s).stem).tolist(); y_all = meta[ycol].to_numpy(np.int64); fold_all = meta.folder.to_numpy()
base = load_wavegram(CKPT).eval(); q = QuantWavegram(base, 8, 8, "cpu")
cdir = a.out / a.task; cdir.mkdir(parents=True, exist_ok=True)

def trunk(net, w32):
    x = net.wavegram(w32)
    for i in range(2, 6): x = getattr(net, f"conv_block{i}")(x)
    return x

def build_cache():
    f32, f8 = cdir / "trunk_fp32.npy", cdir / "trunk_w8a8.npy"
    if f32.exists() and f8.exists(): return np.load(f32, mmap_mode="r"), np.load(f8, mmap_mode="r")
    idx = index_wavs(aud); b32, b8 = base.to(dev), q.net.to(dev); out32 = out8 = None; t0 = time.time()
    for s in range(0, len(rows), 16):
        rr = rows[s:s + 16]; w = np.stack([np.pad(x, (0, pad - len(x))) for x in (load_clip(idx[stems[i]]) for i in rr)])
        w32 = M.resample_to_32k(torch.from_numpy(w).float()); nb = len(rr)
        if nb < 16: w32 = w32[torch.arange(16) % nb]
        with torch.no_grad(): o32 = trunk(b32, w32.to(dev)).half().cpu().numpy()[:nb]; o8 = trunk(b8, w32.to(dev)).half().cpu().numpy()[:nb]
        if out32 is None: out32 = np.zeros((len(rows),) + o32.shape[1:], np.float16); out8 = np.zeros_like(out32)
        out32[s:s + nb], out8[s:s + nb] = o32, o8
        if dev.type == "mps": torch.mps.empty_cache()
        if s % 1600 == 0: print(f"cache {s}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    np.save(f32, out32); np.save(f8, out8); base.cpu(); q.net.cpu()
    return out32, out8

class Tail(nn.Module):
    def __init__(self, src):
        super().__init__(); self.conv_block6 = copy.deepcopy(src.conv_block6); self.fc1 = copy.deepcopy(src.fc1); self.cls = nn.Linear(2048, K)
    def embed(self, x, train=False):
        x = F.dropout(x, 0.2, train); x = self.conv_block6(x, (1, 1)); x = x.mean(3); x = x.max(2)[0] + x.mean(2)
        return F.relu(self.fc1(x))
    def forward(self, x, train=False): return self.cls(F.dropout(self.embed(x, train), 0.5, train))

def f1(yt, p):
    cm = np.bincount(yt * K + p, minlength=K * K).reshape(K, K); tp = np.diag(cm).astype(float); den = cm.sum(0) + cm.sum(1); ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))

def batches(X, ix, fn, bs=128):
    out = []
    with torch.no_grad():
        for s in range(0, len(ix), bs): out.append(fn(torch.from_numpy(np.asarray(X[ix[s:s + bs]], np.float32)).to(dev)).float().cpu().numpy())
    return np.concatenate(out)

C32, C8 = build_cache(); pos = {r: i for i, r in enumerate(rows)}
for seed in a.seeds:
    od = cdir / f"f{a.fold}_s{seed}"
    if (od / "train_log.json").exists(): continue
    if (time.time() - T_START) / 60 > a.max_minutes - 25: print("time limit before next seed", flush=True); sys.exit(3)
    od.mkdir(exist_ok=True)
    ref = np.load(tau_file("mn10", seed)) if a.task == "tau" else np.load(sdir(a.task, "mn10", seed)[0] / "mn10_w8a8__clean.npz")
    iv = ref[f"iv_idx_{a.fold}"].astype(int); te = np.flatnonzero(fold_all[rows] == a.fold)
    if SMOKE: iv = iv[iv < len(rows)]
    tr = np.setdiff1d(np.setdiff1d(np.arange(len(rows)), te), iv)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Tail(base).to(dev); m.conv_block6.bn1.eval(); m.conv_block6.bn2.eval()
    opt = torch.optim.AdamW([{"params": list(m.conv_block6.parameters()) + list(m.fc1.parameters()), "lr": LR_TAIL}, {"params": m.cls.parameters(), "lr": LR_CLS}], weight_decay=WD)
    yrows = y_all[rows]; best, best_state, bad, log = -1.0, None, 0, []
    for ep in range(EPOCHS):
        m.train(); m.conv_block6.bn1.eval(); m.conv_block6.bn2.eval(); perm = rng.permutation(tr); t0 = time.time()
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]; x = torch.from_numpy(np.asarray(C32[b], np.float32)).to(dev); yb = F.one_hot(torch.from_numpy(yrows[b]), K).float().to(dev)
            lam = float(rng.beta(MIX_ALPHA, MIX_ALPHA)); j = torch.randperm(len(b), device=dev)
            x = lam * x + (1 - lam) * x[j]; yb = lam * yb + (1 - lam) * yb[j]
            loss = torch.sum(-yb * F.log_softmax(m(x, True), 1), 1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); pv = batches(C32, iv, lambda x: m(x)).argmax(1); v = f1(yrows[iv], pv); log.append(dict(epoch=ep, iv_f1=v, seconds=time.time() - t0))
        print(f"{a.task} f{a.fold} s{seed} epoch {ep} iv F1 {v:.4f}", flush=True)
        if v > best: best, bad, best_state = v, 0, copy.deepcopy(m.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE: break
    # Amendment 1: the deployed W8A8 tail is saved as int8 codes + scales (t23_tailq.py), the FP32 tail is not kept (only its clean embeddings,
    # for the protocol 3 head); the W8A8 logits of every condition are computed later by t23_ft_score.py from the W8A8 trunk.
    m.load_state_dict(best_state); m.eval(); save_tail_q8(m.conv_block6, m.fc1, od / "tail_q8.pt"); np.save(od / "train_rows.npy", rows[tr])
    allr = np.arange(len(rows)); E32 = batches(C32, allr, lambda x: m.embed(x))
    sub = meta.iloc[rows]; crc = zlib.crc32("\n".join(sub["name"].map(lambda s: Path(s).stem)).encode())
    np.save(od / "wg_fp32__clean.npy", E32.astype(np.float32))
    (od / "wg_fp32__clean.json").write_text(json.dumps({"names_crc32": crc, "n_clips": len(rows), "rows": "device A" if a.task == "tau" else "all", "fold": a.fold, "seed": seed}))
    (od / "train_log.json").write_text(json.dumps(dict(best_iv_f1=best, epochs=log, recipe=dict(lr_tail=LR_TAIL, lr_cls=LR_CLS, wd=WD, batch=BS, max_epochs=EPOCHS, patience=PATIENCE, mixup=MIX_ALPHA)), indent=1))
    print(f"{a.task} f{a.fold} s{seed}: best iv F1 {best:.4f}", flush=True)
