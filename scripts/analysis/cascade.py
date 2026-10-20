#!/usr/bin/env python3
"""Confidence and uncertainty cascades built from trained single models.

A post-hoc adaptive-precision baseline. For each (seed, fold) it takes the
single-model checkpoints of that same seed and fold, runs Q4 on every sample,
and escalates a sample to a higher-precision model (Q8, or FP32) when Q4 looks
unsure. Samples that are not escalated keep the deterministic Q4 prediction.

Escalation signals (a sample escalates when its score is above the threshold):
  confidence   1 - max softmax probability of Q4
  mc_variance  predictive variance of Q4 under MC dropout (M passes, dropout
               on, variance of the softmax summed over classes, the router's
               uncertainty measure applied to the classifier)

Thresholds are chosen per fold on that fold's inner validation split only (the
split that selected the checkpoints) and applied once to the held-out test
fold; test labels are read only to score the final predictions. Two operating
points:
  match_moe  inner-validation escalation rate closest to the share of
             inner-validation samples the precision-prior MoE
             (<ds>_pprior_a1.0, same seed and fold) routes to its
             highest-precision expert
  max_f1     highest inner-validation macro F1 (ties: fewer escalations)

Reported per variant: test macro F1, escalation fraction, expected storage
(Q4 plus the escalation model, both stored) and expected compute per sample
relative to one Q4 pass, as MACs and as bit operations (MACs x weight bits x
activation bits; ternary counts as 2 bits). The MoE arms of the same seed and
fold get the same cost measures from their held-out routing shares.

Every <ds>_singles_s<seed> run with Q4, Q8 and FP32 under the outputs
directories is used; results go to <out>/<ds>_s<seed>.json, one record per
fold, so the collector can pair them with the MoE arms on (seed, fold).

    python scripts/analysis/cascade.py --outputs outputs [/other/outputs ...] \\
        --out outputs/cascade
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SINGLES = {"q4": "4", "q8": "8", "fp32": "esc"}
MOE_ARMS = {"uniform": "{ds}_uniform_ctrl", "pprior": "{ds}_pprior_a1.0",
            "st_uniform": "{ds}_sharedtrunk_uniform_ctrl", "st_pprior": "{ds}_sharedtrunk_a1.0"}
SIGNALS, TARGETS, POINTS = ("confidence", "mc_variance"), ("q8", "fp32"), ("match_moe", "max_f1")
NON_FEATURE_COLS = ["folder", "name", "label", "category"]


# ── thresholds (inner validation only) ─────────────────────────────────────

def candidate_thresholds(scores) -> np.ndarray:
    """Escalate iff score > t: -inf escalates every sample, +inf none, and
    one candidate sits between each pair of distinct scores."""
    u = np.unique(scores)
    return np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])


def choose_threshold(scores, y, pred_low, pred_high, point: str, target_rate: float | None = None) -> float:
    """Threshold from one split, the inner validation split: max_f1 maximizes
    macro F1 of the cascade's predictions, match_moe brings the escalation
    rate closest to target_rate; ties go to fewer escalations."""
    best = None
    for t in candidate_thresholds(scores):
        esc = scores > t
        if point == "max_f1":
            key = (-f1_score(y, np.where(esc, pred_high, pred_low), average="macro", zero_division=0), esc.mean())
        else:
            key = (abs(esc.mean() - target_rate), esc.mean())
        if best is None or key < best[0]:
            best = (key, float(t))
    return best[1]


# ── models, data and costs ─────────────────────────────────────────────────

def find_runs(roots) -> dict:
    """{(tag, seed, model kind): run dir} for every finished run."""
    runs = {}
    for root in roots:
        for s in sorted(root.glob("*/*/summary.json")):
            m = re.match(r"(.+)_s(\d+)$", s.parent.parent.name)
            if not m:
                continue
            key = (m.group(1), int(m.group(2)), s.parent.name)
            if key in runs:
                raise ValueError(f"{key} is in two outputs directories: {runs[key]} and {s.parent}")
            runs[key] = s.parent
    return runs


def load_frame(cfg, cache: dict) -> pd.DataFrame:
    """Feature frame with class_id set from label_col, read once per CSV."""
    esc = cfg.experiment.datasets.esc
    key = (str(esc.csv), esc.get("label_col", "class_id"))
    if key not in cache:
        cache.clear()
        df = pd.read_csv(key[0])
        if key[1] != "class_id":
            df["class_id"] = df[key[1]]
        cache[key] = df.drop(columns=NON_FEATURE_COLS, errors="ignore")
    return cache[key]


def fold_rows(df_feat: pd.DataFrame, cfg, split: dict) -> dict:
    """Inner validation and test rows of a fold, scaled with the scaler fitted
    on the fold's fit portion (training folds minus the inner validation)."""
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


def load_model(run_dir: Path, fold: int, in_dim: int, n_cls: int):
    """The fold's selected checkpoint, prepared exactly as it was scored."""
    from QWave.moe import qMoEModelBatched
    from scripts.benchmark import build_model, checkpoint_name, prepare_eval_model

    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.experiment.device = "cpu"
    cfg.experiment.current_fold = fold
    kind, r = run_dir.name, cfg.experiment.router
    with contextlib.redirect_stdout(io.StringIO()):
        model = (qMoEModelBatched(cfg, in_dim, n_cls, r.num_experts, r.top_k) if kind == "moe"
                 else build_model(kind, in_dim, n_cls, cfg))
        model.load_state_dict(torch.load(run_dir / f"fold_{fold}" / checkpoint_name(kind), map_location="cpu"))
        model = prepare_eval_model(kind, model, cfg)
    return model.eval(), cfg


def layer_cost(m) -> np.ndarray | None:
    """[MACs, bit operations] of one linear layer per sample, else None."""
    from QWave.qmoe_layers import BitLinear

    if isinstance(m, BitLinear):
        nb = m.num_bits
        w, a = (2, 2) if nb == "bitnet" else (nb, 8 if nb < 16 else 32)
    elif isinstance(m, nn.Linear):
        w, a = 32, 32
    else:
        return None
    macs = m.in_features * m.out_features
    return np.array([macs, macs * w * a], dtype=float)


def cost(module) -> np.ndarray:
    return sum((c for c in map(layer_cost, module.modules()) if c is not None), np.zeros(2))


def moe_cost(model, shares) -> np.ndarray:
    """Expected per-sample cost of an MoE: every router pass (M MC passes plus
    the logits pass for a Bayesian router), the shared trunk if any, and each
    expert weighted by its routing share."""
    passes = model.router.mc_samples + 1 if model.use_curiosity else 1
    c = passes * cost(model.router)
    if model.shared is not None:
        return c + cost(model.shared.trunk) + sum(s * cost(h) for s, h in zip(shares, model.shared.heads))
    return c + sum(s * cost(e) for s, e in zip(shares, model.experts))


@torch.no_grad()
def softmax(model, X) -> np.ndarray:
    return F.softmax(model(X), dim=1).numpy()


@torch.no_grad()
def mc_variance(model, X, passes: int, seed: int) -> np.ndarray:
    """Variance of the softmax over MC-dropout passes, summed over classes."""
    torch.manual_seed(seed)
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in drops:
        m.train()
    try:
        P = torch.stack([F.softmax(model(X), dim=1) for _ in range(passes)])
    finally:
        for m in drops:
            m.eval()
    return P.var(dim=0, unbiased=False).sum(dim=1).numpy()


def moe_escalation_rate(model, X, seed: int, batch_size: int) -> float:
    """Share of X the MoE routes to its highest-precision expert, MC dropout
    seeded and batched as in the scored test pass."""
    from scripts.benchmark import route_test_fold

    loader = DataLoader(TensorDataset(X, torch.zeros(len(X), dtype=torch.long)), batch_size=batch_size)
    experts, _, _ = route_test_fold(model, loader, "cpu", seed)
    return float(np.mean(experts == int(model.precision_beta.argmax())))


def same_split(a: dict, b: dict) -> bool:
    return a["test_idx"] == b["test_idx"] and a["inner_val_idx"] == b["inner_val_idx"]


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


# ── one dataset and seed ───────────────────────────────────────────────────

def run_seed(ds: str, seed: int, runs: dict, passes: int, cache: dict) -> dict:
    from QWave.memory import storage_bits

    dirs = {name: runs[(f"{ds}_singles", seed, kind)] for name, kind in SINGLES.items()}
    moe_dirs = {arm: runs[(tag.format(ds=ds), seed, "moe")] for arm, tag in MOE_ARMS.items()
                if (tag.format(ds=ds), seed, "moe") in runs}
    summaries = {name: {f["fold"]: f for f in json.loads((d / "summary.json").read_text())["folds"]}
                 for name, d in {**dirs, **moe_dirs}.items()}
    folds = sorted(summaries["q4"])
    q4_cfg = OmegaConf.load(dirs["q4"] / "config.yaml")
    df = load_frame(q4_cfg, cache)
    in_dim, n_cls = df.shape[1] - 1, int(df["class_id"].nunique())
    out = {"dataset": ds, "seed": seed, "mc_passes": passes, "label_col": q4_cfg.experiment.datasets.esc.get(
        "label_col", "class_id"), "in_dim": in_dim, "num_classes": n_cls,
        "run_dirs": {k: str(v) for k, v in {**dirs, **moe_dirs}.items()}, "costs": {}, "folds": []}

    for fold in folds:
        split = json.loads((dirs["q4"] / f"fold_{fold}" / "split.json").read_text())
        for name, d in {**dirs, **moe_dirs}.items():
            if not same_split(split, json.loads((d / f"fold_{fold}" / "split.json").read_text())):
                raise ValueError(f"{ds} seed {seed} fold {fold}: {name} was not trained on the same split")
        rows = fold_rows(df, q4_cfg, split)
        (X_iv, y_iv), (X_te, y_te) = rows["inner_val"], rows["test"]
        models = {name: load_model(d, fold, in_dim, n_cls)[0] for name, d in dirs.items()}
        costs = {name: cost(m) for name, m in models.items()}
        storage = {name: storage_bits(m) / 8 for name, m in models.items()}
        out["costs"] = {name: {"macs": costs[name][0], "bops": costs[name][1], "storage_bytes": storage[name]}
                        for name in models}

        p_iv = {name: softmax(m, X_iv) for name, m in models.items()}
        p_te = {name: softmax(m, X_te) for name, m in models.items()}
        pred_iv = {name: p.argmax(1) for name, p in p_iv.items()}
        pred_te = {name: p.argmax(1) for name, p in p_te.items()}
        scores = {"confidence": (1 - p_iv["q4"].max(1), 1 - p_te["q4"].max(1)),
                  "mc_variance": (mc_variance(models["q4"], X_iv, passes, seed),
                                  mc_variance(models["q4"], X_te, passes, seed))}

        rec = {"seed": seed, "fold": int(fold), "n_inner_val": int(len(y_iv)), "n_test": int(len(y_te)),
               "singles": {name: {"f1_macro": macro_f1(y_te, pred_te[name]),
                                  "summary_f1_macro": summaries[name][fold]["f1_macro"],
                                  "inner_val_f1_macro": macro_f1(y_iv, pred_iv[name])} for name in models},
               "moe_inner_val_escalation": None, "variants": {}, "moe_arms": {}}
        for name, s in rec["singles"].items():
            if abs(s["f1_macro"] - s["summary_f1_macro"]) > 1e-9:
                print(f"[WARN] {ds} s{seed} fold {fold}: {name} rescored at {s['f1_macro']:.4f}, "
                      f"summary has {s['summary_f1_macro']:.4f}")

        if "pprior" in moe_dirs:
            moe, moe_cfg = load_model(moe_dirs["pprior"], fold, in_dim, n_cls)
            rec["moe_inner_val_escalation"] = moe_escalation_rate(moe, X_iv, seed,
                                                                  int(moe_cfg.experiment.train.batch_size))
        for arm, d in moe_dirs.items():
            moe, _ = load_model(d, fold, in_dim, n_cls)
            f = summaries[arm][fold]
            shares = f["routing"]["overall"]
            c = moe_cost(moe, shares) / costs["q4"]
            rec["moe_arms"][arm] = {"f1_macro": f["f1_macro"],
                                    "escalation_fraction": shares[int(moe.precision_beta.argmax())],
                                    "storage_bytes": f["model_size_bytes"],
                                    "compute_rel_q4_macs": c[0], "compute_rel_q4_bops": c[1]}

        for signal in SIGNALS:
            s_iv, s_te = scores[signal]
            q4_passes = 1 + (passes if signal == "mc_variance" else 0)
            for target in TARGETS:
                for point in POINTS:
                    rate = rec["moe_inner_val_escalation"]
                    if point == "match_moe" and rate is None:
                        continue
                    t = choose_threshold(s_iv, y_iv, pred_iv["q4"], pred_iv[target], point, rate)
                    esc_iv, esc_te = s_iv > t, s_te > t
                    pred = np.where(esc_te, pred_te[target], pred_te["q4"])
                    frac = float(esc_te.mean())
                    c = (q4_passes * costs["q4"] + frac * costs[target]) / costs["q4"]
                    rec["variants"][f"{signal}_{target}_{point}"] = {
                        "signal": signal, "target": target, "point": point, "threshold": t,
                        "inner_val_escalation": float(esc_iv.mean()),
                        "inner_val_f1_macro": macro_f1(y_iv, np.where(esc_iv, pred_iv[target], pred_iv["q4"])),
                        "escalation_fraction": frac, "f1_macro": macro_f1(y_te, pred),
                        "accuracy": float(accuracy_score(y_te, pred)),
                        "storage_bytes": storage["q4"] + storage[target],
                        "compute_rel_q4_macs": c[0], "compute_rel_q4_bops": c[1]}
        out["folds"].append(rec)
        best = max(rec["variants"].items(), key=lambda kv: kv[1]["f1_macro"])
        print(f"  {ds} s{seed} fold {fold}: Q4 {rec['singles']['q4']['f1_macro']:.3f}, "
              f"best cascade {best[0]} {best[1]['f1_macro']:.3f} "
              f"(escalated {best[1]['escalation_fraction']:.2f})")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", nargs="+", required=True, help="outputs directories of scripts/benchmark.py")
    ap.add_argument("--out", required=True, help="writes <out>/<ds>_s<seed>.json")
    ap.add_argument("--mc-passes", type=int, default=10)
    ap.add_argument("--datasets", nargs="+", default=None, help="default: every dataset with Q4, Q8 and FP32")
    args = ap.parse_args(argv)

    roots = list(dict.fromkeys(Path(p).resolve() for p in args.outputs))
    runs = find_runs(roots)
    todo = sorted({(tag[: -len("_singles")], seed) for tag, seed, kind in runs
                   if tag.endswith("_singles") and kind == SINGLES["q4"]})
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache, written = {}, []
    for ds, seed in todo:
        if args.datasets and ds not in args.datasets:
            continue
        missing = [k for k in SINGLES.values() if (f"{ds}_singles", seed, k) not in runs]
        if missing:
            print(f"skip {ds} seed {seed}: no single model(s) {missing}")
            continue
        print(f"== {ds} seed {seed}")
        res = run_seed(ds, seed, runs, args.mc_passes, cache)
        path = out_dir / f"{ds}_s{seed}.json"
        path.write_text(json.dumps(res, indent=2))
        written.append(path)
    print(f"\n{len(written)} cascade result(s) in {out_dir}")
    return written


if __name__ == "__main__":
    main()
