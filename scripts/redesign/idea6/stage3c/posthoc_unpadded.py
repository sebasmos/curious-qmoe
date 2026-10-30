"""Post hoc (not in the Stage 3c plan): UrbanSound8K mn10 static F1 with FP32 heads retrained on the exact-length
clean FP32 embeddings (R/scores_unpr/urban), against the padded model with its padded heads (Stage 3b scores) and
the exact-length model with the padded heads (R/scores_unp/urban, P5a). Per (backbone, condition): mean F1, the
difference per (seed, fold) with Stage 2's fold-level tests and a 95% CI. Writes R/analysis/urban/padding_posthoc.csv.

    python scripts/redesign/idea6/stage3c/posthoc_unpadded.py --r R --s3b S3B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "stage2"))
sys.path.insert(0, str(HERE.parents[3]))
import analyze as A  # noqa: E402
from analyze_3c import ci95  # noqa: E402

SEEDS = [0, 1, 2]
CONDS = ["clean"] + [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (20, 10, 5, 0)] + ["field"]
K = 10


def f1_rows(sd: A.ScoreDir, cfg, cond):
    r = sd.get(cfg, cond)
    folds = sorted(r["iv"])
    return {f: A.f1_fast(r["y"][r["fold"] == f], A.softmax(r["z"][r["fold"] == f]).argmax(1), K) for f in folds}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--s3b", type=Path, required=True)
    a = ap.parse_args(argv)
    A.NB_RATIO["task"] = 1.0 / 9.0  # 10 folds of equal size: n_test / n_train
    rows = []
    for bb in ("fp32", "w8a8"):
        cfg = f"mn10_{bb}"
        for cond in CONDS:
            per = {}
            for name, root, tag in (("padded_padded_heads", a.s3b / "scores_cnn" / "urban", "mn10_urban_s{}"),
                                    ("unpadded_padded_heads", a.r / "scores_unp" / "urban", "mn10_urban_s{}"),
                                    ("unpadded_retrained_heads", a.r / "scores_unpr" / "urban", "mn10unp_urban_s{}")):
                vals = {}
                for s in SEEDS:
                    sd = A.ScoreDir(root / tag.format(s) / "esc")
                    for f, v in f1_rows(sd, cfg, cond).items():
                        vals[(s, f)] = v
                per[name] = vals
            keys = sorted(per["padded_padded_heads"])
            row = {"backbone": bb, "condition": cond}
            for name, vals in per.items():
                row[f"f1_{name}"] = float(np.mean([vals[k] for k in keys]))
            for name in ("unpadded_padded_heads", "unpadded_retrained_heads"):
                x = np.array([per[name][k] for k in keys])
                z = np.array([per["padded_padded_heads"][k] for k in keys])
                t = A.tests(x, z, keys)
                fm = np.array(json.loads(t["fold_diffs"]))
                lo, hi = ci95(fm)
                row.update({f"{name}_minus_padded": t["diff"], f"{name}_ci95_lo": lo, f"{name}_ci95_hi": hi,
                            f"{name}_p_t_fold": t["p_t_fold"], f"{name}_p_nb": t["p_nb"]})
            rows.append(row)
    df = pd.DataFrame(rows)
    out = a.r / "analysis" / "urban" / "padding_posthoc.csv"
    df.to_csv(out, index=False)
    print(df[df.condition.isin(["clean", "field"])].to_string())


if __name__ == "__main__":
    main()
