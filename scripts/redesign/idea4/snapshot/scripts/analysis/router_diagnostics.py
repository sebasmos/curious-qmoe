#!/usr/bin/env python3
"""Held-out router diagnostics for MoE runs (redesign idea 4).

For every fold of each run directory (outputs/<tag>/moe) the scored test pass
is replayed (same batches through a DataLoader, same seed, so the MC-dropout
draws match) and these are computed on the held-out clips only:

  f1_macro                 replayed macro F1, checked against metrics.json
  shares                   share of clips routed to each expert
  rule_changed_share       share of routing decisions that change when the
                           uncertainty rule is switched off on the same
                           weights (precision prior alpha 0, escalation off)
  f1_macro_rule_off        macro F1 with the rule off
  auroc_<signal>_model_err AUROC of a signal for the MoE's own errors
  auroc_<signal>_cheap_err AUROC for the lowest-precision expert's errors
                           (signals: rule_u = the uncertainty the routing rule
                           reads, mc_variance, error_prob when the router has
                           an error head; references that need the experts'
                           outputs, so no router can use them before dispatch:
                           ref_model_conf = 1 - max softmax of the MoE output,
                           ref_cheap_conf = 1 - max softmax of the cheapest
                           expert)
  hi_share_by_difficulty_decile
                           share routed to the highest-precision expert per
                           decile of difficulty (mean cross-entropy of the true
                           class over all experts, each expert run on every
                           clip; decile 1 = easiest)
  expert_acc               accuracy of each expert on every clip
  bops_per_clip, bops_rel_q4
                           expected bit operations per clip (router passes plus
                           experts weighted by their routing shares, as in
                           cascade.moe_cost) and the ratio to one Q4 pass
  expert_bops_rel_q4       the experts' part alone, relative to one Q4 pass

Test labels are read only to score. Writes one JSON with per-fold rows and
means per run. A run given as name=path#no_escalation is replayed with the
hard escalation rule switched off (the same weights with the precision prior
only), so its F1 is not checked against the scored pass.

    python scripts/analysis/router_diagnostics.py --run A=outputs/esc_A_s0/moe \\
        --run uniform=outputs/esc_uniform_ctrl_s0/moe --out diag.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.cascade import cost, moe_cost  # noqa: E402
from scripts.analysis.run_loader import list_folds, load_trained_fold  # noqa: E402


@torch.no_grad()
def replay(model, X, y, batch_size, seed):
    """Routing, prediction and uncertainty signals of the scored pass."""
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    torch.manual_seed(seed)
    model.eval()
    sel, pred, sig = [], [], {"rule_u": [], "mc_variance": [], "error_prob": [], "ref_model_conf": []}
    for xb, _ in loader:
        out, router_p, _, u = model(xb)
        sel.append(router_p.argmax(1))
        pred.append(out.argmax(1))
        aux = getattr(model.router, "aux", {})
        conf = 1 - F.softmax(out, 1).max(1).values
        for name, v in (("rule_u", u), ("mc_variance", aux.get("mc_variance")), ("error_prob", aux.get("error_prob")),
                        ("ref_model_conf", conf)):
            if v is not None:
                sig[name].append(v)
    sig = {k: torch.cat(v).numpy() for k, v in sig.items() if v}
    return torch.cat(sel).numpy(), torch.cat(pred).numpy(), sig


@contextlib.contextmanager
def rule_off(model):
    """Precision prior and hard escalation switched off on the same weights."""
    alpha = model.curiosity_alpha
    tau = model.escalation_tau.clone() if hasattr(model, "escalation_tau") else None
    model.curiosity_alpha = 0.0
    if tau is not None:
        model.escalation_tau.fill_(float("inf"))
    try:
        yield
    finally:
        model.curiosity_alpha = alpha
        if tau is not None:
            model.escalation_tau.copy_(tau)


def auroc(score, wrong):
    return float(roc_auc_score(wrong, score)) if 0 < wrong.sum() < len(wrong) else None


def deciles(score):
    """Decile index 0..9 of each value by rank (0 = lowest score)."""
    rank = np.argsort(np.argsort(score, kind="stable"), kind="stable")
    return rank * 10 // len(score)


def fold_diagnostics(run_dir: Path, fold: int, csv: str | None, no_escalation: bool = False) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        tf = load_trained_fold(run_dir, fold, csv=csv)
    m, cfg = tf.model, tf.cfg
    if no_escalation:
        m.escalation_tau.fill_(float("inf"))
    seed, bs = int(cfg.experiment.get("seed", 0)), int(cfg.experiment.train.batch_size)
    X, y = tf.X_test, tf.y_test
    yn = y.numpy()
    sel, pred, sig = replay(m, X, y, bs, seed)
    with rule_off(m):
        sel0, pred0, _ = replay(m, X, y, bs, seed)

    with torch.no_grad():
        logits = torch.stack([e(X) for e in m.experts])                     # (E, N, C)
        ce = torch.stack([F.cross_entropy(l, y, reduction="none") for l in logits]).numpy()
    exp_pred = logits.argmax(2).numpy()
    sig["ref_cheap_conf"] = 1 - F.softmax(logits[int(m.precision_beta.argmin())], 1).max(1).values.numpy()
    n_exp = len(m.experts)
    cheap, hi = int(m.precision_beta.argmin()), int(m.precision_beta.argmax())
    wrong_model = (pred != yn).astype(int)
    wrong_cheap = (exp_pred[cheap] != yn).astype(int)
    dec = deciles(ce.mean(0))
    shares = np.bincount(sel, minlength=n_exp) / len(sel)

    quants = [str(q) for q in cfg.experiment.router.expert_quantizations]
    c = moe_cost(m, shares)
    q4 = cost(m.experts[quants.index("4")])[1] if "4" in quants else None

    metrics = json.loads((run_dir / f"fold_{fold}" / "metrics.json").read_text())
    row = {
        "fold": fold, "n": int(len(yn)),
        "f1_macro": float(f1_score(yn, pred, average="macro", zero_division=0)),
        "f1_macro_scored": None if no_escalation else float(metrics["f1_macro"]),
        "f1_macro_rule_off": float(f1_score(yn, pred0, average="macro", zero_division=0)),
        "shares": shares.tolist(),
        "shares_rule_off": (np.bincount(sel0, minlength=n_exp) / len(sel0)).tolist(),
        "rule_changed_share": float(np.mean(sel != sel0)),
        "prediction_changed_share": float(np.mean(pred != pred0)),
        "model_error_rate": float(wrong_model.mean()),
        "cheap_error_rate": float(wrong_cheap.mean()),
        "expert_acc": [float((exp_pred[e] == yn).mean()) for e in range(n_exp)],
        "hi_share_by_difficulty_decile": [float((sel[dec == d] == hi).mean()) for d in range(10)],
        "bops_per_clip": float(c[1]), "macs_per_clip": float(c[0]),
        "bops_rel_q4": float(c[1] / q4) if q4 else None,
        "expert_bops_rel_q4": float(sum(s * cost(e)[1] for s, e in zip(shares, m.experts)) / q4) if q4 else None,
        "escalation": metrics.get("escalation"),
    }
    for name, s in sig.items():
        row[f"auroc_{name}_model_err"] = auroc(s, wrong_model)
        row[f"auroc_{name}_cheap_err"] = auroc(s, wrong_cheap)
        row[f"{name}_frac_zero"] = float(np.mean(s <= 1e-12))
    return row


def mean_rows(rows):
    out = {}
    for k, v in rows[0].items():
        vals = [r[k] for r in rows]
        if k == "fold" or any(x is None for x in vals) or isinstance(v, (dict, str)):
            continue
        out[k] = np.mean(np.asarray(vals, dtype=float), axis=0).tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="name=path/to/outputs/<tag>/moe")
    ap.add_argument("--csv", default=None, help="dataset CSV (default: the run's config)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    result = {}
    for spec in a.run:
        name, path = spec.split("=", 1)
        path, _, view = path.partition("#")
        rows = [fold_diagnostics(Path(path), f, a.csv, no_escalation=view == "no_escalation") for f in list_folds(path)]
        for r in rows:
            if r["f1_macro_scored"] is not None and abs(r["f1_macro"] - r["f1_macro_scored"]) > 1e-9:
                raise RuntimeError(f"{name} fold {r['fold']}: replay F1 {r['f1_macro']} != scored {r['f1_macro_scored']}")
        result[name] = {"run_dir": str(path), "view": view or "as scored", "folds": rows, "mean": mean_rows(rows)}
        mn = result[name]["mean"]
        print(f"{name}: F1 {mn['f1_macro']:.4f} changed {mn['rule_changed_share']:.3f} "
              f"AUROC(rule_u, model err) {mn.get('auroc_rule_u_model_err', float('nan')):.3f} "
              f"shares {np.round(mn['shares'], 3).tolist()}", flush=True)
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
