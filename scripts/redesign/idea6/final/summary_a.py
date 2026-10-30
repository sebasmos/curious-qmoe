"""Tables of part A (E6, E7) next to Stage 2's EfficientNet-tier rows: analysis/summary_a.md.

    python summary_a.py --r R --s2 S2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summary_final import md  # noqa: E402

PICKED = {"esc50": "mn20", "urban8k": "mn10"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", required=True, type=Path)
    ap.add_argument("--s2", required=True, type=Path)
    a = ap.parse_args()
    A = a.r / "analysis"
    L = ["# Part A: real shift on the small CNNs (exploratory; every family named, Holm within each)\n"]

    t = pd.read_csv(A / "e6_tests_rate.csv")
    t["picked"] = [c == PICKED[s] for s, c in zip(t.source, t.cnn)]
    L.append("## E6, escalation rate on the other dataset's shared-class clips against the in-domain test fold, clean thresholds\n")
    L.append("A1 = picked CNN of the source dataset (ESC-50 source: mn20-Q8; UrbanSound8K source: mn10-Q8), A2 = the other CNN. "
             "Fold-level two-sided t-test, seeds averaged within fold; Holm within the family of 10.\n")
    L.append(md(t[["family", "source", "cnn", "budget", "rate_in", "rate_shift", "diff", "n_folds", "p_t_fold", "p_holm", "family_size"]]) + "\n")

    d = pd.read_csv(A / "e6_tests_detect.csv")
    L.append("## E6, detection AUROC: shifted clips against the in-domain test fold\n")
    L.append("The margin gate (score 1 - margin), idea 7's noise_lr gate and the random gate (0.5). A3 = picked CNN, A4 = other CNN; "
             "Holm within the family of 4.\n")
    L.append(md(d[["family", "source", "cnn", "contrast", "auroc_margin", "auroc_noise_lr", "diff", "n_folds", "p_t_fold", "p_holm", "family_size"]]) + "\n")

    r = pd.read_csv(A / "e6_rates.csv")
    L.append("## E6, cheap-tier accuracy and gate AUROC for cheap-wrong clips, in-domain and shifted\n")
    L.append(md(r[r.budget.isin(["b05", "b30"])][["source", "cnn", "budget", "cheap_acc_in", "cheap_acc_cross", "rate_in", "rate_cross",
                                                  "auroc_wrong_in", "auroc_wrong_cross"]]) + "\n")
    c = pd.read_csv(A / "e6_cascade.csv")
    L.append("## E6, cascade accuracy (rule ast: the AST tier's own label on escalated clips)\n")
    L.append(md(c[c.budget.isin(["b05", "b30"])][["source", "cnn", "tier", "budget", "ast_acc_in", "ast_acc_cross", "cascade_acc_in",
                                                  "cascade_acc_cross", "auroc_fixable_in", "auroc_fixable_cross"]]) + "\n")

    # Stage 2's EfficientNet tier, the same quantities
    s2 = pd.read_csv(a.s2 / "analysis/e6_cross_dataset.csv")
    s2 = s2[(s2.cheap_bb == "w8a8") & (s2.tier == "w4g128") & s2.budget.isin(["b05", "b30"])]
    L.append("### Reference: Stage 2, EfficientNet-B3 W8A8 tier, the same checks (AST tier w4g128 at 1024 frames)\n")
    L.append(md(s2[["source", "target", "budget", "cheap_acc_in", "cheap_acc_cross", "rate_in", "rate_cross", "auroc_wrong_in",
                    "auroc_wrong_cross", "cascade_acc_in", "cascade_acc_cross"]]) + "\n")

    o = pd.read_csv(A / "e7_ood.csv")
    L.append("## E7, unseen classes: escalation rate in-domain and on unseen-class clips of the other dataset\n")
    cols = ["task", "cnn", "family", "n_ood"] + [f"rate_{w}_{b}" for b in ("b05", "b10", "b20", "b30", "b50") for w in ("in", "ood")] + \
           ["auroc_margin", "auroc_noise_lr"]
    L.append(md(o[cols]) + "\n")
    tt = pd.read_csv(A / "e7_tests.csv")
    L.append("Tests (fold-level t; A5 = picks' escalation, A6 = ESC-50 mn10, A7 = picks' detection AUROC, A8 = ESC-50 mn10's):\n")
    L.append(md(tt[["family", "test", "task", "cnn", "budget", "value_a", "value_b", "diff", "n_folds", "p_t_fold", "p_holm", "family_size"]]) + "\n")
    e7s2 = pd.read_csv(a.s2 / "analysis/e7_ood.csv")
    L.append("### Reference: Stage 2, EfficientNet tier (H1 cheap system of each task)\n")
    L.append(md(e7s2) + "\n")
    (A / "summary_a.md").write_text("\n".join(L))
    print("\n".join(L)[:3000])


if __name__ == "__main__":
    main()
