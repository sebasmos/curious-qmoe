#!/usr/bin/env python3
"""Component figure: (a) full model, (b) epistemic uncertainty, (c) precision prior.

Three panels side by side, drawn at print size for a full text width (6.5 in),
in the visual language of generate_architecture_figure.py (its helpers are
reused). Each panel works in inches: one data unit is one inch.

Two modes, same layout:

--run-dir       Real data. One fold of a trained precision-prior MoE run is
                rebuilt with run_loader.py (held-out test rows only, scaled
                with the scaler fitted on that fold's fit portion). Every test
                sample is routed in eval mode with the checkpoint's running
                uncertainty range, in the test loader's batches and with the
                run's seed, so the routing matches the scored test pass. The
                confident sample is the median-u sample of the lowest u decile,
                the uncertain one the median-u sample of the highest decile.
                Panel (b) shows the uncertain sample's own dropout passes,
                panel (c) both samples' base and prior-weighted routing.
--illustrative  Schematic values, tagged "illustrative" in the figure.

Every plotted number goes to a JSON next to the PDF.

    python scripts/analysis/generate_component_figure.py --illustrative
    python scripts/analysis/generate_component_figure.py \
        --run-dir outputs/esc_pprior_a1.0_s0/moe --csv <esc-50.csv> --fold 1
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

from generate_architecture_figure import (EXPERTS, INK, ROSE_T, ROUTER_FC, SUBINK,
                                          beta, card, fmt_beta, router, wire)

FIG_W, DRAW_H, HEAD_H = 6.5, 3.0, 0.45
FIG_H = DRAW_H + HEAD_H
PANELS = [  # left edge, width (inches), title, subtitle
    (0.00, 2.30, "(a) Full model", "shared trunk, top-1 routing"),
    (2.38, 1.92, "(b) Epistemic uncertainty", "knowing when it doesn't know"),
    (4.36, 2.14, "(c) Precision prior", "deferring to precision when unsure"),
]
FS, SUBFS = 8.0, 7.0   # card title / subtitle sizes in print points
ALPHA_DEMO = 2.0       # illustrative strength for panel (c), not a fitted value
PANEL_FC = "#F6F6F8"
NAMES = [n for n, *_ in EXPERTS]
QUANTS = ["bitnet", "4", "8"]  # expert_quantizations the panels are drawn for


def _argmax(v):
    return max(range(len(v)), key=v.__getitem__)


def illustrative_values():
    """Schematic numbers; (c) applies the real prior to them with ALPHA_DEMO."""
    b = [beta(bits) for _, _, bits, _ in EXPERTS]

    def sample(u, p):
        q = [pi * math.exp(ALPHA_DEMO * u * bi) for pi, bi in zip(p, b)]
        q = [qi / sum(q) for qi in q]
        return {"u": u, "u_raw": None, "p": list(p), "p_ua": q,
                "top1_base": _argmax(p), "top1_prior": _argmax(q)}

    unc = sample(1.0, (0.42, 0.33, 0.25))
    unc["passes"] = [(0.55, 0.30, 0.15), (0.30, 0.45, 0.25), (0.40, 0.20, 0.40)]
    return {"mode": "illustrative", "alpha": ALPHA_DEMO, "beta": b, "mc_samples": None,
            "expert_names": NAMES, "confident": sample(0.0, (0.70, 0.20, 0.10)), "uncertain": unc}


def real_values(run_dir, fold=1, csv=None, seed=None):
    """Route one fold's held-out samples as the scored test pass did and pick the two samples."""
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.run_loader import load_trained_fold

    tf = load_trained_fold(run_dir, fold, csv=csv)
    m, cfg = tf.model, tf.cfg
    if not (m.use_curiosity and m.curiosity_strategy == "precision_prior" and m.gate_threshold == 0):
        sys.exit("needs a precision-prior MoE run without uncertainty gating")
    if [str(q) for q in cfg.experiment.router.expert_quantizations] != QUANTS:
        sys.exit(f"the panels are drawn for experts {QUANTS}")
    seed = int(cfg.experiment.get("seed", 0)) if seed is None else seed
    bs = int(cfg.experiment.train.batch_size)
    batches = [tf.X_test[s:s + bs] for s in range(0, len(tf.X_test), bs)]

    # Replay the router's MC-dropout passes exactly as BayesianRouter.forward
    # runs them, keeping the per-pass distributions, with the batches and seed
    # of the scored test pass (scripts/benchmark.py, route_test_fold).
    torch.manual_seed(seed)
    passes, logits = [], []
    with torch.no_grad():
        for x in batches:
            m.router.train(True)
            passes.append(torch.stack([F.softmax(m.router.net(x), dim=1)
                                       for _ in range(m.router.mc_samples)]))
            m.router.train(False)
            logits.append(m.router.net(x))
        passes, logits = torch.cat(passes, dim=1), torch.cat(logits)
        u_raw = passes.var(dim=0, unbiased=False).sum(dim=1)
        p, p_ua = F.softmax(logits, dim=1), m._apply_curiosity(logits, u_raw)
        u = m._normalized_uncertainty(u_raw)
        if u is None:
            sys.exit("degenerate running uncertainty range in the checkpoint")
        u = u.squeeze(1)
        # The model's own forward, same seed and batches, must route identically.
        torch.manual_seed(seed)
        ref = [m(x) for x in batches]
    assert torch.allclose(torch.cat([r[3] for r in ref]), u_raw)
    assert torch.allclose(torch.cat([r[1] for r in ref]), p_ua)

    order = np.argsort(u_raw.numpy(), kind="stable")
    k = max(1, len(order) // 10)
    picks = {"confident": int(order[:k][(k - 1) // 2]), "uncertain": int(order[-k:][(k - 1) // 2])}
    meta = pd.read_csv(cfg.experiment.datasets.esc.csv, usecols=lambda c: c in ("name", "category"))

    def record(i):
        row = int(tf.test_idx[i])
        return {"test_position": i, "csv_row": row, **{c: str(meta[c].iloc[row]) for c in meta},
                "label": int(tf.y_test[i]), "u_raw": float(u_raw[i]), "u": float(u[i]),
                "p": p[i].tolist(), "p_ua": p_ua[i].tolist(),
                "top1_base": int(p[i].argmax()), "top1_prior": int(p_ua[i].argmax()),
                "passes": passes[:, i].tolist()}

    top_b, top_q = p.argmax(dim=1).numpy(), p_ua.argmax(dim=1).numpy()
    share = lambda t: (np.bincount(t, minlength=len(QUANTS)) / len(t)).tolist()
    metrics = Path(run_dir) / f"fold_{fold}" / "metrics.json"
    scored = json.loads(metrics.read_text()).get("routing", {}).get("overall") if metrics.exists() else None
    return {"mode": "real", "run_dir": str(Path(run_dir).resolve()),
            "csv": str(cfg.experiment.datasets.esc.csv), "fold": fold, "seed": seed, "batch_size": bs,
            "n_test": len(order), "mc_samples": int(m.router.mc_samples),
            "alpha": float(m.curiosity_alpha), "beta": m.precision_beta.tolist(), "expert_names": NAMES,
            "running_u_min": float(m.unc_running_min), "running_u_max": float(m.unc_running_max),
            "selection": (f"held-out samples ranked by u_raw (u is its clamped running min/max "
                          f"rescaling); decile size {k}; confident = lowest-decile sample at sorted "
                          f"position {(k - 1) // 2}, uncertain = highest-decile sample at the same "
                          f"position within its decile"),
            "n_top1_changed_by_prior": int((top_b != top_q).sum()),
            "fraction_top1_changed_by_prior": float((top_b != top_q).mean()),
            "top1_share_base": share(top_b), "top1_share_prior": share(top_q),
            "scored_pass_top1_share": scored,
            **{name: record(i) for name, i in picks.items()}}


def panel(fig, left, width, title, subtitle):
    ax = fig.add_axes([left / FIG_W, 0.0, width / FIG_W, DRAW_H / FIG_H])
    ax.set_xlim(0, width); ax.set_ylim(0, DRAW_H); ax.axis("off")
    x = (left + width / 2) / FIG_W
    fig.text(x, (DRAW_H + 0.32) / FIG_H, title, ha="center", va="center", fontsize=9, color=INK)
    fig.text(x, (DRAW_H + 0.15) / FIG_H, subtitle, ha="center", va="center", fontsize=7.5,
             color=SUBINK)
    return ax


def c(ax, x, y, w, h, text, fc, **kw):
    card(ax, x, y, w, h, text, fc, fs=kw.pop("fs", FS), subfs=kw.pop("subfs", SUBFS),
         pad=0.03, lw=0.8, gap=0.6, **kw)


def w_(ax, p, q, rad=0.0, lw=0.9, **kw):
    wire(ax, p, q, rad=rad, lw=lw, ms=7, **kw)


def note(ax, x, y, s, ha="left", color=SUBINK, **kw):
    ax.text(x, y, s, ha=ha, va="center", fontsize=kw.pop("fs", 7), color=color, **kw)


def well(ax, x, y, w, h):
    """Neutral rounded area that holds the routing bars."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=PANEL_FC, ec="none",
                                zorder=0))


def bars(ax, x0, y0, vals, bw, gap, hmax=0.5, top=None, names=False):
    """Routing distribution: one pastel bar per expert on a thin baseline."""
    span = 3 * bw + 2 * gap
    ax.plot([x0 - 0.03, x0 + span + 0.03], [y0, y0], color=SUBINK, lw=0.6, solid_capstyle="butt")
    for k, ((name, col, _, _), v) in enumerate(zip(EXPERTS, vals)):
        x = x0 + k * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, y0), bw, v * hmax, boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=col, ec=INK if k == top else "none", lw=0.9))
        if names:
            ax.text(x + bw / 2, y0 - 0.04, name, ha="center", va="top", fontsize=7, color=SUBINK)
    return span


def full_model(ax):
    cx = 1.15
    c(ax, 0.30, 2.80, 1.70, 0.16, "Audio embedding  $z$", "#FEF4B5")
    w_(ax, (0.52, 2.76), (0.52, 2.59))
    c(ax, 0.04, 2.25, 0.95, 0.30, "Shared 8-bit Trunk", "#EFEFF1", sub="once per sample")
    ry = 1.94
    a, b = router(ax, cx, ry, fs=7.5)
    w_(ax, (cx, 2.76), (cx, ry + b + 0.02))
    c(ax, 1.63, ry - 0.17, 0.63, 0.34, "Epistemic", "#FBEBE9", sub="uncertainty  $u$",
      dashed=True, tc=ROSE_T, subc=ROSE_T)
    w_(ax, (cx + a + 0.02, ry), (1.58, ry), color=ROSE_T)
    ty = 1.33
    c(ax, cx - 0.48, ty - 0.14, 0.96, 0.28, "Top-1", "#F6F6F8", sub="dispatch to head $i^*$")
    w_(ax, (cx, ry - b - 0.02), (cx, ty + 0.19))
    note(ax, cx + 0.05, (ry - b + ty + 0.17) / 2, "$p_i$", fs=7.5)
    # precision prior: uncertainty feeds back into routing
    w_(ax, (1.94, ry - 0.22), (cx + 0.53, ty), rad=0.42, color=ROSE_T, lw=1.1)
    note(ax, 2.00, 1.20, "precision\nprior", ha="center", color=ROSE_T, linespacing=1.0)
    # trunk features into the dispatch
    w_(ax, (0.52, 2.21), (cx - 0.53, ty), rad=0.35)
    note(ax, 0.52, 1.58, "features")
    ey = 0.60
    for k, (name, col, bits, _) in enumerate(EXPERTS):
        x = 0.04 + 0.775 * k
        c(ax, x, ey, 0.68, 0.30, name, col, sub=f"$\\beta={fmt_beta(bits)}$")
        w_(ax, (cx, ty - 0.18), (x + 0.34, ey + 0.34), rad=(0.16, 0.0, -0.16)[k])
        w_(ax, (x + 0.34, ey - 0.04), (cx, 0.31), rad=(-0.14, 0.0, 0.14)[k])
    c(ax, cx - 0.56, 0.03, 1.12, 0.25, "Output", "#EFEFF1",
      sub=r"head $i^*$ logits $\times\, p^{\mathrm{UA}}_{i^*}$")


def sci(x):
    """Compact mathtext number (no $): 0.0123 stays decimal, 1.2e-05 becomes 1.2 x 10^-5."""
    if x == 0 or 1e-3 <= abs(x) < 1e3:
        return f"{x:.3g}"
    e = math.floor(math.log10(abs(x)))
    return f"{x / 10 ** e:.2g}\\times10^{{{e}}}"


def tag(d, plural=False):
    if d["mode"] == "illustrative":
        return "illustrative"
    return f"held-out sample{'s' if plural else ''}, fold {d['fold']}"


def uncertainty(ax, d):
    W = 1.92; cx = W / 2
    s, M = d["uncertain"], d["mc_samples"]
    c(ax, cx - 0.50, 2.80, 1.00, 0.16, "embedding  $z$", "#FEF4B5")
    # M stochastic passes: ghosted copies of the router behind the front one
    rx, ry = cx - 0.12, 2.30
    a, b = router(ax, rx, ry, fs=7.5, zorder=4)
    for k, alpha in ((2, 0.30), (1, 0.55)):
        ax.add_patch(Ellipse((rx + 0.07 * k, ry + 0.05 * k), 2 * a, 2 * b,
                             fc=ROUTER_FC, ec="white", lw=0.8, alpha=alpha, zorder=4 - k))
    w_(ax, (rx, 2.76), (rx, ry + b + 0.02))
    note(ax, rx + a + 0.20, ry, "$M$ dropout\npasses", linespacing=1.05)
    w_(ax, (rx, ry - b - 0.02), (rx, 1.80))
    # this sample's routing distribution in passes 1, 2 and M
    well(ax, 0.04, 1.14, W - 0.08, 0.60)
    note(ax, 0.06, 1.69, "$p_i^{(m)}$ per pass")
    note(ax, W - 0.06, 1.69, tag(d), ha="right")
    shown = [s["passes"][0], s["passes"][1], s["passes"][-1]]
    labels = ("$m{=}1$", "$m{=}2$", f"$m{{=}}M{{=}}{M}$" if M else "$m{=}M$")
    for x, v, lab in zip((0.14, 0.70, 1.36), shown, labels):
        span = bars(ax, x, 1.26, v, bw=0.11, gap=0.04, hmax=0.30)
        note(ax, x + span / 2, 1.20, lab, ha="center")
    note(ax, 1.235, 1.33, r"$\cdots$", ha="center", fs=8)
    w_(ax, (cx, 1.08), (cx, 0.92))
    value = "" if s["u_raw"] is None else " = " + sci(s["u_raw"])
    c(ax, cx - 0.74, 0.56, 1.48, 0.30, "spread across passes", "#F6F6F8", fs=7.5,
      sub=r"$u_{\mathrm{raw}} = \Sigma_i\, \mathrm{Var}_m\, p_i^{(m)}" + value + "$")
    w_(ax, (cx, 0.51), (cx, 0.36), color=ROSE_T)
    note(ax, cx + 0.06, 0.435, "running min/max", color=ROSE_T)
    c(ax, cx - 0.62, 0.03, 1.24, 0.28, "Epistemic uncertainty", "#FBEBE9",
      sub=r"$u \in [0, 1]$" if s["u_raw"] is None else f"$u = {s['u']:.2f}$",
      dashed=True, tc=ROSE_T, subc=ROSE_T)


def precision_prior(ax, d):
    W = 2.14
    real = d["mode"] == "real"
    fmt = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
    note(ax, W / 2, 2.88, r"$p_i^{\mathrm{UA}} \propto p_i \cdot \exp(\alpha\, u\, \beta_i)$",
         ha="center", color=ROSE_T, fs=8.5)
    terms = [f"{n} $\\beta={fmt(bv)}$" for n, bv in zip(NAMES, d["beta"])]
    note(ax, W / 2, 2.70, "   ".join(([f"$\\alpha={d['alpha']:g}$"] if real else []) + terms), ha="center")
    well(ax, 0.04, 0.50, W - 0.08, 2.03)
    note(ax, W / 2, 2.43, tag(d, plural=True), ha="center")
    if real:
        note(ax, W / 2, 2.31, f"prior changes top-1 for {d['n_top1_changed_by_prior']} of {d['n_test']}"
             f" ({100 * d['fraction_top1_changed_by_prior']:.1f}%)", ha="center")
    bw, gap, hmax = 0.11, 0.15, 0.42
    span = 3 * bw + 2 * gap
    for x, key, word in ((0.56, "confident", "\\approx 0"), (1.62, "uncertain", "\\approx 1")):
        s = d[key]
        x0 = x - span / 2
        uval = f"= {s['u']:.2f}" if real else word
        note(ax, x, 2.17, f"{key}, $u {uval}$", ha="center", color=INK, fs=7.5)
        bars(ax, x0, 1.62, s["p"], bw=bw, gap=gap, hmax=hmax, names=True)
        note(ax, x0 - 0.05, 1.68, "$p_i$", ha="right", fs=7.5)
        w_(ax, (x, 1.44), (x, 1.23), color=ROSE_T, lw=1.1)
        b0, top = s["top1_base"], s["top1_prior"]
        note(ax, x - 0.06, 1.335, "top-1 kept" if b0 == top else f"{NAMES[b0]} $\\rightarrow$ {NAMES[top]}",
             ha="right", color=ROSE_T)
        bars(ax, x0, 0.74, s["p_ua"], bw=bw, gap=gap, hmax=hmax, top=top)
        note(ax, x0 - 0.05, 0.80, r"$p_i^{\mathrm{UA}}$", ha="right", fs=7.5)
        bx = x0 + top * (bw + gap) + bw / 2
        w_(ax, (bx, 0.70), (x, 0.37))
        note(ax, (bx + x) / 2 + (0.05 if bx >= x else -0.05), 0.54, "top-1",
             ha="left" if bx >= x else "right", color=INK)
        name, col, bits, kind = EXPERTS[top]
        c(ax, x - 0.40, 0.04, 0.80, 0.30, name, col, sub=f"{kind}, $\\beta={fmt(d['beta'][top])}$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", help="trained precision-prior MoE run, <outputs>/<tag>/moe")
    mode.add_argument("--illustrative", action="store_true", help="schematic bars instead of real data")
    ap.add_argument("--csv", help="dataset CSV (default: the one in the run's config.yaml)")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, help="MC-dropout seed (default: the run's seed, as scored)")
    ap.add_argument("--out", default="scripts/analysis/outputs-paper/architecture_panels.pdf")
    args = ap.parse_args()

    d = illustrative_values() if args.illustrative else real_values(args.run_dir, args.fold, args.csv, args.seed)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = [panel(fig, *p) for p in PANELS]
    full_model(axes[0]); uncertainty(axes[1], d); precision_prior(axes[2], d)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    out.with_suffix(".json").write_text(json.dumps(d, indent=2) + "\n")
    for key in ("confident", "uncertain"):
        r = d[key]
        print(f"{key}: u={r['u']:.3f} top-1 {NAMES[r['top1_base']]} -> {NAMES[r['top1_prior']]}"
              + (f" (csv row {r['csv_row']})" if "csv_row" in r else ""))
    if d["mode"] == "real":
        print(f"top-1 changed by the prior: {d['fraction_top1_changed_by_prior']:.4f}; "
              f"top-1 share {[round(v, 4) for v in d['top1_share_prior']]}, "
              f"scored pass {d['scored_pass_top1_share']}")
    print(f"saved {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
