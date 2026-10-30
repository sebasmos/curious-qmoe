"""T15 V5: a small 1D waveform CNN trained from scratch (ANALYSIS_PLAN_T15.md section 4). Modes:
  --precond: fold --fold, seed --seed, at most --epochs epochs on the fit rows, clean inner-validation macro-F1 after every epoch (never a test clip); writes <out>/wave_precond_<dataset>.json
  --full: every outer fold and seed of the dataset, epoch cap 40; scores the conditions listed in --conditions / --field (see score_conditions) and writes score npz files
The inner split is the stored one (iv_idx_<fold> of the stored MN10-Q8 score file of the same seed), so the waveform CNN is selected on the same clips as every other expert.
Run under the GPU lock (gpu_run.sh)."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2"))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from QWave.effnet_quant import fake_quant_sym, rtn_per_channel
from audio_shift import index_wavs, load_clip
from t15_extract import DATA

CACHE = Path.home() / "Documents/qmoe-work/t15-cache"
SEEDS = {"urban": [0, 1, 2], "esc50": [0, 1, 2, 3, 4]}
NF = {"urban": 10, "esc50": 5}
K_OF = {"urban": 10, "esc50": 50}


def stored_scores(dataset, seed):
    from t13_common import sdir
    d, cfg = sdir("urban" if dataset == "urban" else "esc50", "mn10", seed)
    return np.load(d / f"{cfg}__clean.npz")


class WaveCNN(nn.Module):
    SPEC = [(32, 48, 1), (48, 64, 2), (64, 96, 4), (96, 128, 8), (128, 192, 16)]

    def __init__(self, k):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 32, 80, stride=8, padding=36)
        self.bn0 = nn.BatchNorm1d(32)
        self.convs = nn.ModuleList([nn.Conv1d(a, b, 3, dilation=d, padding=d) for a, b, d in self.SPEC])
        self.bns = nn.ModuleList([nn.BatchNorm1d(b) for _, b, _ in self.SPEC])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(192, k)

    def forward(self, x, lens):
        """x [B, L] RMS-normalised waveform (zero padded), lens [B] valid samples."""
        h = F.relu(self.bn0(self.conv0(x.unsqueeze(1))))
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = F.relu(bn(conv(h)))
            if i < len(self.convs) - 1:
                h = F.max_pool1d(h, 2)
        t = h.shape[-1]
        tv = torch.clamp(torch.ceil(lens.float() / (x.shape[1] / t)), min=1).long()
        mask = (torch.arange(t, device=h.device)[None, :] < tv[:, None]).unsqueeze(1)
        avg = (h * mask).sum(-1) / mask.sum(-1)
        mx = h.masked_fill(~mask, -1e9).max(-1)[0]
        return self.fc(self.drop(avg + mx))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def macs(m, length):
    """MACs of the Conv1d and Linear layers for one clip of `length` samples."""
    tot = [0.0]

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv1d):
            tot[0] += out.shape[1] * out.shape[2] * mod.in_channels * mod.kernel_size[0]
        else:
            tot[0] += mod.in_features * mod.out_features

    hs = [mm.register_forward_hook(hook) for mm in m.modules() if isinstance(mm, (nn.Conv1d, nn.Linear))]
    with torch.no_grad():
        m.eval()(torch.zeros(1, length), torch.tensor([length]))
    for h in hs:
        h.remove()
    return tot[0]


class QuantWave(nn.Module):
    """W8A8 fake quantization of a trained WaveCNN: per-output-channel 8-bit weights, per-sample 8-bit dynamic inputs on every Conv1d and Linear."""

    def __init__(self, base, device):
        super().__init__()
        m = copy.deepcopy(base).cpu().eval()
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, (nn.Conv1d, nn.Linear)):
                    codes, s = rtn_per_channel(mod.weight.data, 8)
                    mod.weight.data.copy_(codes * s)
                    mod.register_forward_pre_hook(lambda _m, inp: (fake_quant_sym(inp[0], 8, tuple(range(1, inp[0].dim()))),) + tuple(inp[1:]))
        self.net = m.to(device)

    @torch.no_grad()
    def forward(self, x, lens):
        return self.net(x, lens)


def macro_f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))


def normalise(x, lens):
    rms = torch.sqrt((x ** 2).sum(1) / lens.float().clamp(min=1)) + 1e-8
    return x / rms[:, None]


def load_audio(dataset):
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = DATA[dataset]
    f, g = CACHE / f"{dataset}_clean.npy", CACHE / f"{dataset}_lens.npy"
    if f.exists() and g.exists():
        return np.load(f, mmap_mode="r"), np.load(g)
    meta = pd.read_csv(cfg["csv"], usecols=["name"])
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    index = index_wavs(cfg["audio"])
    W = np.zeros((len(stems), cfg["pad"]), np.float32)
    lens = np.zeros(len(stems), np.int64)
    for i, s in enumerate(stems):
        x = load_clip(index[s])
        lens[i] = len(x)
        W[i, :len(x)] = x
    np.save(f, W)
    np.save(g, lens)
    return np.load(f, mmap_mode="r"), lens


def predict(model, W, lens, rows, device, bs=64):
    """logits [len(rows), K] for the waveforms W[rows] (already conditioned) in fixed batches of bs."""
    out = []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(rows), bs):
            r = rows[b:b + bs]
            nb = len(r)
            rr = np.concatenate([r, np.repeat(r[:1], bs - nb)]) if nb < bs else r
            x = torch.from_numpy(np.asarray(W[rr], np.float32)).to(device)
            ln = torch.from_numpy(lens[rr]).to(device)
            out.append(model(normalise(x, ln), ln).float().cpu().numpy()[:nb])
    return np.concatenate(out) if out else np.zeros((0, 0))


def train_one(W, lens, y, fit_idx, iv_idx, k, epochs, seed, device, patience=8, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = WaveCNN(k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.6, patience=3)
    cnt = np.bincount(y[fit_idx], minlength=k).astype(float)
    cw = torch.tensor(1.0 / np.maximum(cnt, 1.0), dtype=torch.float32, device=device)
    cw = cw / cw[torch.as_tensor(y[fit_idx], device=device)].mean()
    bs, hist, best, best_state, bad = 32, [], -1.0, None, 0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = rng.permutation(fit_idx)
        for b in range(0, len(perm) - bs + 1, bs):
            r = perm[b:b + bs]
            x = torch.from_numpy(np.asarray(W[r], np.float32))
            ln = torch.from_numpy(lens[r])
            # random circular shift of the valid part, up to a quarter of the clip
            xs = torch.zeros_like(x)
            for j in range(bs):
                L = int(ln[j])
                s = int(rng.integers(0, max(1, L // 4)))
                xs[j, :L] = torch.roll(x[j, :L], s)
            xs, ln = xs.to(device), ln.to(device)
            yb = torch.from_numpy(y[r]).to(device)
            out = model(normalise(xs, ln), ln)
            loss = (F.cross_entropy(out, yb, reduction="none", label_smoothing=0.1) * cw[yb]).sum() / cw[yb].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        lg = predict(model, W, lens, iv_idx, device)
        f1 = macro_f1(y[iv_idx], lg.argmax(1), k)
        hist.append(f1)
        sched.step(f1)
        log(f"  epoch {ep + 1}: inner-validation macro-F1 {f1:.4f} ({time.time() - t0:.0f}s)")
        if f1 > best:
            best, bad, best_state = f1, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def condition_rows(dataset, rows_test, rows_iv, cell, assign):
    """Rows of one cell to score for a fold: full cells (clean, 0 dB cells) all rows; the other field cells only the rows a field assignment gives them."""
    allr = np.union1d(rows_test, rows_iv)
    if cell == "clean" or cell in FULL_CELLS[dataset]:
        return allr
    return allr[(assign[0][allr] == cell) | (assign[1][allr] == cell)]


def full_run(a, W, lens):
    """Outer folds a.folds of seeds a.seeds: train (cap a.epochs), W8A8, score every cell (clean, 0 dB cells, the nine field cells) for the fold's test and inner-validation
    rows, and store the logits in <out>/wave_parts/<dataset>_s<seed>_f<fold>.npz. assemble() turns the parts into score files."""
    import pandas as pd
    from audio_shift import FIELD_CELLS, FIELD_SEEDS, NoiseMaker, cond_name, field_assignment, mix
    cfg = DATA[a.dataset]
    meta = pd.read_csv(cfg["csv"], usecols=["name"]); stems = meta["name"].map(lambda s: Path(s).stem).tolist(); n = len(stems)
    bg_names = pd.read_csv(cfg["bg_csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist(); bg_index = index_wavs(cfg["bg_audio"])
    nm = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)
    assign = [np.array(field_assignment(n, FIELD_SEEDS[k])) for k in ("test", "val")]
    cells = ["clean"] + [cond_name(k, s) for k, s in FIELD_CELLS]
    k = K_OF[a.dataset]; dev = torch.device(a.device)
    scratch = np.zeros((n, W.shape[1]), np.float32)
    parts = a.out / "wave_parts"; parts.mkdir(exist_ok=True)
    for seed in a.seeds:
        z0 = stored_scores(a.dataset, seed); y = z0["y"].astype(np.int64); fold = z0["test_fold"].astype(int)
        for f in a.folds:
            pf = parts / f"{a.dataset}_s{seed}_f{f}.npz"
            if pf.exists():
                continue
            iv = z0[f"iv_idx_{f}"]; te = np.flatnonzero(fold == f); fit = np.setdiff1d(np.flatnonzero(fold != f), iv)
            t0 = time.time(); print(f"{a.dataset} seed {seed} fold {f}: fit {len(fit)}", flush=True)
            model, hist = train_one(W, lens, y, fit, iv, k, a.epochs, seed, dev, log=lambda s: None)
            print(f"  trained {len(hist)} epochs, best inner-validation F1 {max(hist):.4f} ({time.time() - t0:.0f}s)", flush=True)
            q = QuantWave(model, dev); out = {"te": te, "iv": iv, "hist": np.array(hist)}
            for c in cells:
                rws = condition_rows(a.dataset, te, iv, c, assign)
                if c == "clean":
                    Wc = W
                else:
                    Wc = scratch
                    kind, snr = c.split("__"); snr = float(snr[:-2])
                    for i in rws:
                        x = np.asarray(W[i, :lens[i]], np.float32); x, _ = mix(x, nm(stems[i], kind, len(x)), snr); scratch[i, :len(x)] = x
                lg = predict(q, Wc, lens, rws, dev)
                if c != "clean":
                    scratch[rws] = 0.0
                out[f"{c}__rows"] = rws; out[f"{c}__lg"] = lg
            np.savez_compressed(pf, **out); print(f"  scored all cells ({time.time() - t0:.0f}s)", flush=True)


def assemble(a):
    """Parts -> <out>/scores/<dataset>/v5_<dataset>_s<seed>/esc/v5__<cell>.npz (the score_eff format)."""
    import zlib
    import pandas as pd
    from audio_shift import FIELD_CELLS, cond_name
    cfg = DATA[a.dataset]; stems = pd.read_csv(cfg["csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist(); n = len(stems); crc = zlib.crc32("\n".join(stems).encode())
    cells = ["clean"] + [cond_name(k, s) for k, s in FIELD_CELLS]; k = K_OF[a.dataset]
    for seed in SEEDS[a.dataset]:
        z0 = stored_scores(a.dataset, seed); y = z0["y"].astype(np.int64); fold = z0["test_fold"].astype(np.int16)
        res = {c: {"y": y, "test_fold": fold, "test_logits": np.full((n, k), np.nan, np.float32), "names_crc32": np.int64(crc)} for c in cells}
        for f in range(1, NF[a.dataset] + 1):
            p = np.load(a.out / f"wave_parts/{a.dataset}_s{seed}_f{f}.npz")
            for c in cells:
                rws, lg = p[f"{c}__rows"], p[f"{c}__lg"]; pos = {int(r): j for j, r in enumerate(rws)}
                for r in p["te"]:
                    if int(r) in pos: res[c]["test_logits"][r] = lg[pos[int(r)]]
                ivl = np.full((len(p["iv"]), k), np.nan, np.float32)
                for j, r in enumerate(p["iv"]):
                    if int(r) in pos: ivl[j] = lg[pos[int(r)]]
                res[c][f"iv_idx_{f}"] = p["iv"]; res[c][f"iv_logits_{f}"] = ivl
        d = a.out / f"scores/{a.dataset}/v5_{a.dataset}_s{seed}/esc"; d.mkdir(parents=True, exist_ok=True)
        for c in cells:
            np.savez_compressed(d / f"v5__{c}.npz", **res[c])
        print(f"seed {seed} written to {d}", flush=True)


FULL_CELLS = {"urban": ["bg__0db"], "esc50": ["bg__0db", "pink__0db", "white__0db"]}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=["urban", "esc50"])
    p.add_argument("--precond", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--folds", type=int, nargs="*", default=None)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="smoke test: use the first N fit rows")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    W, lens = load_audio(a.dataset)
    z = stored_scores(a.dataset, a.seed)
    y = z["y"].astype(np.int64)
    fold = z["test_fold"].astype(int)
    k = K_OF[a.dataset]
    if a.assemble:
        assemble(a)
        return
    if not a.precond:
        a.seeds = a.seeds if a.seeds is not None else SEEDS[a.dataset]
        a.folds = a.folds if a.folds is not None else list(range(1, NF[a.dataset] + 1))
        full_run(a, W, lens)
        return
    iv = z[f"iv_idx_{a.fold}"]
    fit = np.setdiff1d(np.flatnonzero(fold != a.fold), iv)
    if a.limit:
        fit = fit[:a.limit]
    assert not np.intersect1d(fit, np.flatnonzero(fold == a.fold)).size
    print(f"{a.dataset} fold {a.fold} seed {a.seed}: fit {len(fit)}, inner validation {len(iv)}, params {n_params(WaveCNN(k))}", flush=True)
    model, hist = train_one(W, lens, y, fit, iv, k, a.epochs, a.seed, torch.device(a.device), log=lambda s: print(s, flush=True))
    q = QuantWave(model, torch.device(a.device))
    f1q = macro_f1(y[iv], predict(q, W, lens, iv, torch.device(a.device)).argmax(1), k)
    ref = macro_f1(y[iv], z[f"iv_logits_{a.fold}"].argmax(1), k)
    res = {"dataset": a.dataset, "fold": a.fold, "seed": a.seed, "n_fit": int(len(fit)), "n_iv": int(len(iv)), "params": n_params(model), "epochs_run": len(hist),
           "iv_f1_per_epoch_fp32": hist, "best_iv_f1_fp32": max(hist), "reached_0_60_within_10": bool(max(hist[:10]) >= 0.60), "best_epoch_iv_f1_w8a8": f1q,
           "mn10_q8_iv_f1_same_clips": ref, "macs_per_clip_at_pad": macs(model.cpu(), DATA[a.dataset]["pad"])}
    (a.out / f"wave_precond_{a.dataset}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
