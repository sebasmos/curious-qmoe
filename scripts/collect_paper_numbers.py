#!/usr/bin/env python3
"""Collect every number the paper reports from outputs/ into one report.

Reads the summary.json files written by scripts/reproduce_paper.sh and prints
the paper's tables (and a machine-readable JSON alongside), so any claim in the
manuscript can be traced back to a specific run directory.

    python scripts/collect_paper_numbers.py [--outputs outputs] [--json paper_numbers.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def folds(root: Path, tag: str):
    """Per-fold F1 for a run tag, or None when the run is absent."""
    for sub in ("moe", "qmoe"):
        f = root / tag / sub / "summary.json"
        if f.exists():
            return [x["best_f1"] for x in json.loads(f.read_text())["folds"]]
    return None


def agg(vals):
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def fmt(vals, pooled_from=None):
    if not vals:
        return "run missing"
    m, s = agg(vals)
    note = f"  ({pooled_from} runs pooled, {len(vals)} folds)" if pooled_from else ""
    return f"{m:.4f} +/- {s:.4f}{note}"


def var_red(cand, base):
    if not cand or not base:
        return None
    return 1 - np.var(cand) / np.var(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--json", default="paper_numbers.json")
    args = ap.parse_args()
    root = Path(args.outputs)
    report = {}

    print("=" * 72)
    print("PAPER NUMBERS  (source: %s)" % root.resolve())
    print("=" * 72)

    # --- headline comparison, ESC-50 super-categories --------------------
    base = folds(root, "esc_uniform_ctrl") or folds(root, "esc_bitnet_4_8_baseline_eq8fix_ctrl")
    pp = (folds(root, "esc_pprior_a1.0") or folds(root, "esc_bitnet_4_8_pprior_a1.0") or []) + \
         (folds(root, "esc_pprior_a1.0_r2") or folds(root, "esc_bitnet_4_8_pprior_a1.0_r2") or [])
    st = (folds(root, "esc_sharedtrunk_a1.0") or []) + (folds(root, "esc_sharedtrunk_a1.0_r2") or [])

    print("\n-- ESC-50 super-categories, BitNet-Q4/8 --")
    print(f"  uniform routing          {fmt(base)}")
    print(f"  precision prior          {fmt(pp, pooled_from=2)}")
    print(f"  shared trunk (RECOMMENDED) {fmt(st, pooled_from=2)}")
    for name, cand in (("precision prior", pp), ("shared trunk", st)):
        if cand and base:
            vr = var_red(cand, base)
            p = stats.ttest_ind(cand, base).pvalue
            print(f"    {name} vs uniform: dF1={np.mean(cand)-np.mean(base):+.4f}, "
                  f"variance reduction={vr:.1%}, t-test p={p:.4f}")
    if st and pp:
        print(f"    shared trunk vs precision prior: dF1={np.mean(st)-np.mean(pp):+.4f}, "
              f"p={stats.ttest_ind(st, pp).pvalue:.4f}")
    report["esc"] = {"uniform": agg(base), "precision_prior": agg(pp), "shared_trunk": agg(st)}

    # --- cross-dataset ----------------------------------------------------
    print("\n-- cross-dataset (BitNet-Q4/8) --")
    report["cross_dataset"] = {}
    for ds, u, p_, s_ in (
        ("ESC-50", "esc_uniform_ctrl", "esc_pprior_a1.0", "esc_sharedtrunk_a1.0"),
        ("Quinn", "quinn_uniform_ctrl", "quinn_pprior_a1.0", "quinn_sharedtrunk_a1.0"),
        ("Urban8K", "urban_uniform_ctrl", "urban_pprior_a1.0", "urban_sharedtrunk_a1.0"),
    ):
        row = {}
        for label, tag in (("uniform", u), ("precision prior", p_), ("shared trunk", s_)):
            fallback = {"esc_uniform_ctrl": "esc_bitnet_4_8_baseline_eq8fix_ctrl",
                        "quinn_uniform_ctrl": "quinn_bitnet_4_8_baseline_eq8fix_ctrl",
                        "urban_uniform_ctrl": "urban_bitnet_4_8_baseline_eq8fix_ctrl",
                        "esc_pprior_a1.0": "esc_bitnet_4_8_pprior_a1.0",
                        "quinn_pprior_a1.0": "quinn_bitnet_4_8_pprior_a1.0",
                        "urban_pprior_a1.0": "urban_bitnet_4_8_pprior_a1.0"}.get(tag)
            v = folds(root, tag) or (folds(root, fallback) if fallback else None)
            row[label] = agg(v)
            print(f"  {ds:8s} {label:16s} {fmt(v)}")
        report["cross_dataset"][ds] = row

    # --- supplementary ablations -----------------------------------------
    print("\n-- Appendix A: routing strategies (ESC, alpha=0.3 unless noted) --")
    report["strategies"] = {}
    strat = [("uniform", "esc_uniform_ctrl"), ("kl_divergence", "esc_kl_divergence_a0.3"),
             ("entropy_regularization", "esc_entropy_regularization_a0.3"),
             ("precision_sharp", "esc_precision_sharp_a0.3"), ("escalation", "esc_escalation_a0.3"),
             ("soft_escalation", "esc_soft_escalation_a0.3"), ("precision_prior a0.3", "esc_pprior_a0.3")]
    for label, tag in strat:
        alt = {"esc_uniform_ctrl": "esc_bitnet_4_8_baseline_eq8fix_ctrl",
               "esc_kl_divergence_a0.3": "esc_bitnet_4_8_curiosity_eq8fix_a03",
               "esc_precision_sharp_a0.3": "esc_bitnet_4_8_precision_sharp_a0.3",
               "esc_escalation_a0.3": "esc_bitnet_4_8_escalation_a0.3",
               "esc_soft_escalation_a0.3": "esc_bitnet_4_8_soft_escalation_a0.3",
               "esc_pprior_a0.3": "esc_bitnet_4_8_pprior_a0.3"}.get(tag)
        v = folds(root, tag) or (folds(root, alt) if alt else None)
        vr = var_red(v, base)
        print(f"  {label:24s} {fmt(v)}" + (f"   variance reduction {vr:.0%}" if vr is not None else ""))
        report["strategies"][label] = agg(v)

    print("\n-- Appendix A: Monte Carlo budget (precision prior, alpha=1.0) --")
    report["mc_budget"] = {}
    for M, tag in ((1, "esc_pprior_a1.0_mc1"), (5, "esc_pprior_a1.0_mc5"),
                   (10, None), (20, "esc_pprior_a1.0_mc20")):
        if M == 10:
            v = pp
        else:
            v = folds(root, tag) or []
            if M == 5:
                v = v + (folds(root, "esc_pprior_a1.0_mc5_r2") or [])
        print(f"  M={M:<3d} {fmt(v)}")
        report["mc_budget"][M] = agg(v)

    print("\n-- Appendix A: expert sets (precision prior, alpha=1.0) --")
    report["expert_sets"] = {}
    for label, tag in (("BitNet,Q4,Q8", None), ("BitNet,Q4,Q8,Q16", "esc_pprior_a1.0_4experts"),
                       ("BitNet,Q8,Q16,PTQ", "esc_pprior_a1.0_8_16_qesc"),
                       ("BitNet,PTQ", "esc_pprior_a1.0_qesc")):
        v = pp if tag is None else folds(root, tag)
        print(f"  {label:20s} {fmt(v)}")
        report["expert_sets"][label] = agg(v)

    print("\n-- Appendix A: negative ablations --")
    for label, tag in (("uncertainty gating", "esc_gated_a1.0"),
                       ("warm start (per-fold)", "esc_warmstart_perfold")):
        print(f"  {label:24s} {fmt(folds(root, tag))}")

    # --- latency ----------------------------------------------------------
    lat = root.parent / "scripts" / "analysis" / "outputs-paper" / "latency" / "latency_results.json"
    if lat.exists():
        d = json.loads(lat.read_text())
        print("\n-- Section 5.4: latency --")
        print(f"  Q4-Base  {d['q4_base']['mean_ms']:.1f} +/- {d['q4_base']['std_ms']:.2f} ms")
        print(f"  MoE      {d['moe']['mean_ms']:.1f} +/- {d['moe']['std_ms']:.2f} ms")
        print(f"  {d['variance_test']['summary']}")
        report["latency"] = d
    else:
        print("\n-- Section 5.4: latency -- (not yet run)")

    Path(args.json).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nMachine-readable copy written to {args.json}")


if __name__ == "__main__":
    main()
