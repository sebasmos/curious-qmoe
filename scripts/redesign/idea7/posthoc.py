"""Idea 7 stage 1, post hoc (not pre-specified, written after the primary
result): why the clean-calibrated `pre` threshold misses non-inferiority.

1. For n3g64 + FP32 head, rule pre: the gap to the cloud on the fold's clean
   inner validation at the chosen threshold, against the gap on the clean test
   fold at the same threshold, and the spread of the realized test rate over
   (seed, fold) pairs.
2. The gap to the cloud when every fold defers the same share of its test
   clips, ranked by margin (clean and field mixture).

    python scripts/redesign/idea7/posthoc.py --stage2 R --out OUT/outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from defer import NI_MARGIN, TASKS, Run, macro_f1, margin_score, softmax, threshold_pre  # noqa: E402

RATES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="n3g64")
    args = ap.parse_args(argv)
    gap, curve = [], []
    for task, T in TASKS.items():
        for tag in T["tags"]:
            run = Run(args.stage2 / "scores" / task / tag / "esc")
            seed = int(tag.rsplit("_s", 1)[1])
            d, c = run.get(args.device, "clean"), run.get("fp32", "clean")
            y = d["y"]
            for f, (idx, zi) in sorted(d["iv"].items()):
                yi, pdi, pci = y[idx], softmax(zi), softmax(c["iv"][f][1])
                si = margin_score(pdi)
                fc = macro_f1(yi, pci.argmax(1))
                t = threshold_pre(si, yi, pdi.argmax(1), pci.argmax(1), fc - NI_MARGIN)
                di = si > t
                m = d["fold"] == f
                pdt, pct, yt = softmax(d["z"][m]), softmax(c["z"][m]), y[m]
                dt = margin_score(pdt) > t
                fct = macro_f1(yt, pct.argmax(1))
                gap.append({"task": task, "seed": seed, "fold": f, "n_iv": len(idx), "rate_iv": di.mean(),
                            "rate_test": dt.mean(),
                            "gap_iv_device": 100 * (macro_f1(yi, pdi.argmax(1)) - fc),
                            "gap_iv_sys": 100 * (macro_f1(yi, np.where(di, pci.argmax(1), pdi.argmax(1))) - fc),
                            "gap_test_device": 100 * (macro_f1(yt, pdt.argmax(1)) - fct),
                            "gap_test_sys": 100 * (macro_f1(yt, np.where(dt, pct.argmax(1), pdt.argmax(1))) - fct)})
            for cond in ("clean", "field"):
                d, c = run.get(args.device, cond), run.get("fp32", cond)
                for f in sorted(d["iv"]):
                    m = d["fold"] == f
                    pdt, pct, yt = softmax(d["z"][m]), softmax(c["z"][m]), d["y"][m]
                    order = np.argsort(-margin_score(pdt), kind="stable")
                    fct = macro_f1(yt, pct.argmax(1))
                    for r in RATES:
                        dd = np.zeros(len(yt), bool)
                        dd[order[:int(round(r * len(yt)))]] = True
                        curve.append({"task": task, "condition": cond, "seed": seed, "fold": f, "rate": r,
                                      "gap": 100 * (macro_f1(yt, np.where(dd, pct.argmax(1), pdt.argmax(1))) - fct)})
    G, Cv = pd.DataFrame(gap), pd.DataFrame(curve)
    G.to_csv(args.out / "posthoc_threshold_gap.csv", index=False)
    Cv.to_csv(args.out / "posthoc_fixed_rate.csv", index=False)
    s = G.groupby("task").agg(n_iv=("n_iv", "mean"), rate_iv=("rate_iv", "mean"), rate_test=("rate_test", "mean"),
                              rate_test_min=("rate_test", "min"), rate_test_max=("rate_test", "max"),
                              gap_iv_device=("gap_iv_device", "mean"), gap_iv_sys=("gap_iv_sys", "mean"),
                              gap_test_device=("gap_test_device", "mean"), gap_test_sys=("gap_test_sys", "mean"))
    print(s.round(3).to_string())
    print(Cv.groupby(["task", "condition", "rate"])["gap"].mean().unstack("rate").round(2).to_string())


if __name__ == "__main__":
    main()
