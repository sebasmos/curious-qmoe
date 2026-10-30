"""Clean test macro F1 of the heads on the rebuilt EfficientNet embeddings
against Stage 1's heads on the old CSV embeddings, from each run's
summary.json, paired by (seed, fold); fold-level t-test (seeds averaged).

    python scripts/redesign/idea6/stage2/extractor_check.py --results R --stage1 S1 --p3 P3 --out R/analysis/extractor_check_summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OLD = {"esc50": ("S1", "effesc50_s{}"), "esc5": ("P3", "esc_singles_s{}"), "urban": ("P3", "urban_singles_s{}")}
SEEDS = {"esc50": range(5), "esc5": range(5), "urban": range(3)}


def f1s(run: Path):
    d = json.loads((run / "summary.json").read_text())
    return {f["fold"]: f["f1_macro"] for f in d["folds"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--p3", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    roots = {"S1": a.stage1 / "outputs", "P3": a.p3 / "outputs"}
    rows = []
    for task, (root, tag) in OLD.items():
        for k in ("esc", "8", "4", "bitnet"):
            new, old = [], []
            for s in SEEDS[task]:
                rn = a.results / "outputs" / f"effp_{task}_s{s}" / k
                ro = roots[root] / tag.format(s) / k
                if not (rn / "summary.json").exists():
                    continue
                fn, fo = f1s(rn), f1s(ro)
                for f in fn:
                    new.append((s, f, fn[f]))
                    old.append((s, f, fo[f]))
            if not new:
                continue
            n, o = pd.DataFrame(new, columns=["seed", "fold", "f1"]), pd.DataFrame(old, columns=["seed", "fold", "f1"])
            d = (n.f1 - o.f1).groupby(n.fold).mean()
            rows.append({"task": task, "head": k, "seeds": n.seed.nunique(), "old_f1": o.f1.mean(), "new_f1": n.f1.mean(),
                         "diff_pts": 100 * (n.f1 - o.f1).mean(), "fold_min_pts": 100 * d.min(), "fold_max_pts": 100 * d.max(),
                         "p_t_fold": float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 else np.nan})
    D = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(a.out, index=False)
    print(D.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
