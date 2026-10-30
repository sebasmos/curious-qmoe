"""Redesign idea 3, stage 1: per-fold deterministic logits of every single model of
one <ds>_singles_s<seed> run, plus stochastic-rounding (SR) samples of the cheap heads.

SR re-quantizes each BitLinear's stored latent FP32 weight onto the SAME grid and
per-output-channel scale its deterministic quantizer uses (QWave/qmoe_layers.py
at commit 5db7e75), rounding up or down at random with probability proportional
to distance, so E[w_sr] = w inside the grid range:
  k >= 2  step = row_max * best_ratio / qmax (the MSE clip search, recomputed and
          checked to reproduce quantize_weight_kbit); w_sr = clamp(floor(w/step + U)) * step
  k = 1   scale a = mean|w_row|, levels {-a, +a}; P(+a) = clamp((w/a + 1) / 2, 0, 1)
  bitnet  Wc = W - rowmean, alpha = mean |Wc| over the deterministic nonzeros,
          levels {-alpha, 0, +alpha}; P(sign(Wc) alpha) = clamp(|Wc| / alpha, 0, 1)
Activations keep their deterministic quantizers (8-bit absmax, ternary for BitNet);
the final FP32 classifier is unchanged. One SR draw = one full weight sample shared
by all clips of the fold; R independent sets of K draws are stored (seeded).
A manual forward with the deterministic weights is checked against the model.

usage: python extract.py --run .../outputs/urban_singles_s0 --out DIR [--code DIR] [--K 8] [--R 3]
"""
import argparse, contextlib, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch import nn

MODELS = ["bitnet", "1", "2", "4", "8", "16", "esc", "qesc"]
CHEAP = ["bitnet", "1", "2", "4"]
NON_FEATURE_COLS = ["folder", "name", "label", "category"]


def load_frame(cfg):
    esc = cfg.experiment.datasets.esc
    df = pd.read_csv(str(esc.csv))
    lc = esc.get("label_col", "class_id")
    if lc != "class_id":
        df["class_id"] = df[lc]
    return df.drop(columns=NON_FEATURE_COLS, errors="ignore")


def fold_rows(df_feat, cfg, split):
    from QWave.datasets import EmbeddingAdaptDataset
    test_idx, iv_idx = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
    fit_idx = np.setdiff1d(np.arange(len(df_feat)), np.concatenate([test_idx, iv_idx]))
    norm = cfg.experiment.datasets.esc.normalization_type
    rows = {}
    with contextlib.redirect_stdout(io.StringIO()):
        fit = EmbeddingAdaptDataset(df_feat.iloc[fit_idx].reset_index(drop=True), normalization_type=norm)
        for name, idx in (("inner_val", iv_idx), ("test", test_idx)):
            d = EmbeddingAdaptDataset(df_feat.iloc[idx].reset_index(drop=True), normalization_type=norm,
                                      scaler=fit.get_scaler())
            rows[name] = (d.features, d.labels.numpy())
    return rows


def load_model(model_dir, fold, in_dim, n_cls):
    from scripts.benchmark import build_model, checkpoint_name, prepare_eval_model
    cfg = OmegaConf.load(model_dir / "config.yaml")
    cfg.experiment.device = "cpu"
    cfg.experiment.current_fold = fold
    kind = model_dir.name
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(kind, in_dim, n_cls, cfg)
        model.load_state_dict(torch.load(model_dir / f"fold_{fold}" / checkpoint_name(kind), map_location="cpu"))
        model = prepare_eval_model(kind, model, cfg)
    return model.eval()


def layer_cost(m, kind):
    """[MACs, bit-ops]: MACs x weight bits x activation bits (ternary = 2), as scripts/analysis/cascade.py."""
    from QWave.qmoe_layers import BitLinear
    if kind == "qesc":
        if type(m).__name__ == "Linear" and "quantized" in type(m).__module__:
            macs = m.in_features * m.out_features
            return np.array([macs, macs * 8 * 8], dtype=float)
        return None
    if isinstance(m, BitLinear):
        nb = m.num_bits
        w, a = (2, 2) if nb == "bitnet" else (nb, 8 if nb < 16 else 32)
    elif isinstance(m, nn.Linear):
        w, a = 32, 32
    else:
        return None
    macs = m.in_features * m.out_features
    return np.array([macs, macs * w * a], dtype=float)


def cost(module, kind):
    return sum((c for c in (layer_cost(m, kind) for m in module.modules()) if c is not None), np.zeros(2))


@torch.no_grad()
def logits(model, X, bs=64):
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)]).numpy()


# ---------- stochastic rounding on the deterministic quantizer's grid ----------

def kbit_step(w, num_bits):
    """Per-row step of quantize_weight_kbit (k >= 2), same clip-ratio search."""
    from QWave.qmoe_layers import KBIT_CLIP_RATIOS
    qmax = 2 ** (num_bits - 1) - 1
    row_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    z = w / row_max
    best_err = best_ratio = None
    buf = torch.empty_like(z)
    for ratio in KBIT_CLIP_RATIOS.tolist():
        torch.mul(z, qmax / ratio, out=buf)
        buf.round_().clamp_(-qmax, qmax).mul_(ratio / qmax).sub_(z)
        err = buf.pow_(2).sum(dim=1, keepdim=True)
        if best_err is None:
            best_err, best_ratio = err, torch.full_like(err, ratio)
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_ratio = torch.where(better, torch.full_like(err, ratio), best_ratio)
    return row_max * best_ratio / qmax, qmax


def bitnet_parts(W):
    Wc = W - W.mean(dim=1, keepdim=True)
    thr_w = 0.15 * Wc.abs().mean(dim=1, keepdim=True)
    code = torch.where(Wc.abs() < thr_w, torch.zeros_like(Wc), torch.sign(Wc))
    nz = (code != 0).float()
    alpha = (Wc.abs() * nz).sum(dim=1, keepdim=True) / nz.sum(dim=1, keepdim=True).clamp(min=1.0)
    return Wc, code, alpha


def det_weight(layer):
    W = layer.weight.detach()
    nb = layer.num_bits
    if nb == "bitnet":
        _, code, alpha = bitnet_parts(W)
        return alpha * code
    if nb == 1:
        return torch.where(W >= 0, 1.0, -1.0) * W.abs().mean(dim=1, keepdim=True)
    step, qmax = kbit_step(W, nb)
    return (W / step).round().clamp(-qmax, qmax) * step


def sr_weight(layer, gen):
    W = layer.weight.detach()
    nb = layer.num_bits
    U = torch.rand(W.shape, generator=gen)
    if nb == "bitnet":
        Wc, _, alpha = bitnet_parts(W)
        r = (Wc.abs() / alpha).clamp(0, 1)
        return alpha * torch.sign(Wc) * (U < r).float()
    if nb == 1:
        a = W.abs().mean(dim=1, keepdim=True)
        p_plus = ((W / a + 1) / 2).clamp(0, 1)
        return torch.where(U < p_plus, 1.0, -1.0) * a
    step, qmax = kbit_step(W, nb)
    return torch.floor(W / step + U).clamp(-qmax, qmax) * step


@torch.no_grad()
def bitlinear_forward(layer, x, wq):
    """BitLinear.forward with the quantized weight supplied (same expressions as 5db7e75)."""
    W = layer.weight.detach()
    x = layer.act_ln(x)
    if layer.num_bits == "bitnet":
        thr_a = 0.05 * x.abs().mean(dim=-1, keepdim=True)
        xa_code = torch.where(x.abs() < thr_a, torch.zeros_like(x), torch.sign(x))
        xa = x + (xa_code - x)
    else:
        x_scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        xq = (x * x_scale).round().clamp(-128, 127) / x_scale
        xa = x + (xq - x)
    return F.linear(xa, W + (wq - W), layer.bias)


@torch.no_grad()
def manual_logits(model, X, weights):
    from QWave.qmoe_layers import BitLinear
    x, i = X, 0
    for m in model.net:
        if isinstance(m, BitLinear):
            x = bitlinear_forward(m, x, weights[i]); i += 1
        else:
            x = m(x)
    return x


def complete_folds(run):
    folds = sorted(int(p.name.split("_")[1]) for p in (run / "esc").glob("fold_*"))
    return [k for k in folds if all((run / m / f"fold_{k}" / "metrics.json").exists() for m in MODELS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--code", type=Path, default=Path("/Users/cajas.sebastian/Desktop/repositories/curious-qmoe-run-5db7e75"))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--R", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, str(a.code))
    torch.set_num_threads(2)
    from QWave.qmoe_layers import BitLinear, quantize_weight_kbit
    a.out.mkdir(parents=True, exist_ok=True)

    cfg4 = OmegaConf.load(a.run / "4" / "config.yaml")
    seed = int(cfg4.experiment.seed)
    df = load_frame(cfg4)
    in_dim, n_cls = df.shape[1] - 1, int(df["class_id"].nunique())
    folds = complete_folds(a.run)
    checks, costs, sr_checks = [], {}, []
    for fold in folds:
        split = json.loads((a.run / "esc" / f"fold_{fold}" / "split.json").read_text())
        for m in MODELS:
            s = json.loads((a.run / m / f"fold_{fold}" / "split.json").read_text())
            assert s["test_idx"] == split["test_idx"] and s["inner_val_idx"] == split["inner_val_idx"], (m, fold)
        rows = fold_rows(df, cfg4, split)
        (X_iv, y_iv), (X_te, y_te) = rows["inner_val"], rows["test"]
        arrs = {"y_iv": y_iv, "y_te": y_te, "test_idx": np.asarray(split["test_idx"]),
                "iv_idx": np.asarray(split["inner_val_idx"])}
        for m in MODELS:
            model = load_model(a.run / m, fold, in_dim, n_cls)
            costs[m] = cost(model, m).tolist()
            arrs[f"l_iv_{m}"] = logits(model, X_iv)
            arrs[f"l_te_{m}"] = logits(model, X_te)
            met = json.loads((a.run / m / f"fold_{fold}" / "metrics.json").read_text())
            f1_te = f1_score(y_te, arrs[f"l_te_{m}"].argmax(1), average="macro", zero_division=0)
            checks.append({"fold": fold, "model": m, "f1_te": f1_te, "metrics_f1": met["f1_macro"],
                           "diff_te": abs(f1_te - met["f1_macro"]), "size_bytes": met["model_size_bytes"]})
            assert checks[-1]["diff_te"] < 1e-9, checks[-1]
            if m not in CHEAP:
                continue
            layers = [x for x in model.net if isinstance(x, BitLinear)]
            det = [det_weight(x) for x in layers]
            # the recomputed deterministic weights equal the layer's own quantizer
            for x, w in zip(layers, det):
                ref = (bitnet_parts(x.weight.detach())[2] * bitnet_parts(x.weight.detach())[1]
                       if x.num_bits == "bitnet" else quantize_weight_kbit(x.weight.detach(), x.num_bits))
                assert torch.equal(ref, w), (m, fold)
            md = (manual_logits(model, X_te, det).numpy() - arrs[f"l_te_{m}"])
            sr_checks.append({"fold": fold, "model": m, "max_abs_logit_diff_manual_vs_model": float(np.abs(md).max())})
            assert np.abs(md).max() < 1e-4, (m, fold, np.abs(md).max())
            gen = torch.Generator().manual_seed(100003 * seed + 1009 * fold + CHEAP.index(m))
            P = {"iv": np.zeros((a.R, a.K, len(y_iv), n_cls), np.float32),
                 "te": np.zeros((a.R, a.K, len(y_te), n_cls), np.float32)}
            for r in range(a.R):
                for k in range(a.K):
                    ws = [sr_weight(x, gen) for x in layers]
                    P["iv"][r, k] = torch.softmax(manual_logits(model, X_iv, ws), 1).numpy()
                    P["te"][r, k] = torch.softmax(manual_logits(model, X_te, ws), 1).numpy()
            arrs[f"sr_iv_{m}"], arrs[f"sr_te_{m}"] = P["iv"], P["te"]
        np.savez_compressed(a.out / f"fold_{fold}.npz", **arrs)
        print(fold, "max |F1 - metrics.json| test", max(c["diff_te"] for c in checks if c["fold"] == fold),
              "| manual-forward max logit diff",
              max(c["max_abs_logit_diff_manual_vs_model"] for c in sr_checks if c["fold"] == fold), flush=True)
    (a.out / "checks.json").write_text(json.dumps(
        {"run": str(a.run), "seed": seed, "folds": folds, "n_classes": n_cls, "K": a.K, "R": a.R,
         "checks": checks, "sr_checks": sr_checks, "costs": costs}, indent=1))


if __name__ == "__main__":
    main()
