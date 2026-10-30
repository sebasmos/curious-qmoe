"""One small table for the README: margin-gate AUROC (ALL conditions) and macro F1 (ALL) differences against FP32.

    python compact_cross.py --out OUT
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = [("esc50", "ESC-50 x50"), ("esc5", "ESC-50 x5"), ("urban", "UrbanSound8K"), ("tau", "TAU")]
ROWS = [("AST 1024 frames, FP32 head", [("ast1024", b, "esc") for b in ("Q8", "Q6", "Q5", "Q4", "Q4pc", "Q3")]),
        ("AST real length, FP32 head", [("astReal", b, "esc") for b in ("Q8", "Q4", "Q4pc", "Q3")]),
        ("AST real length, Q3 with its own head", [("astReal", "Q3ownhead", "esc")]),
        ("mn10, FP32 head", [("mn10", "Q8sim", "esc")]),
        ("mn20, FP32 head", [("mn20", "Q8sim", "esc")]),
        ("real INT8, own retrained head (ESC-50x50: Q4 head, TAU: Q8 head)", [("mn10", "INT8", None), ("mn20", "INT8", None)])]
HEADS = [("mn10", "FP32"), ("mn20", "FP32"), ("astReal", "FP32")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    st = pd.read_csv(a.out / "paired_vs_fp32.csv")
    sy = pd.read_csv(a.out / "systems.csv")
    st = st.merge(sy[["sid", "task", "model", "backbone", "head"]], on="sid")

    def get(panel, task, model, bb, head, metric):
        x = st[(st.panel == panel) & (st.task == task) & (st.model == model) & (st.backbone == bb) & (st.cond == "ALL") & (st.metric == metric)]
        if head:
            x = x[x["head"] == head]
        if x.empty:
            return "n/a"
        r = x.iloc[0]
        return f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{'*' if r['decided'] else ''}"

    out = []
    for metric, title in (("auroc_margin", "Margin-gate AUROC for 'model is wrong'"), ("f1", "Macro F1")):
        out.append(f"\n### {title}: difference to FP32, ALL conditions, 95% interval over folds\n")
        out.append("| precision | " + " | ".join(n for _, n in TASKS) + " |")
        out.append("|---|" + "---|" * len(TASKS))
        for label, items in ROWS:
            for model, bb, head in items:
                tag = f"{label}: {bb}" if model.startswith("ast") else f"{model} {bb}"
                if label.startswith("real INT8"):
                    tag = f"{model} INT8"
                out.append(f"| {tag} | " + " | ".join(get("backbone", t, model, bb, head, metric) for t, _ in TASKS) + " |")
        out.append("| BitNet head vs FP32 head, mn10 FP32 | " + " | ".join(get("head", t, "mn10", "FP32", "bitnet", metric) for t, _ in TASKS) + " |")
        out.append("| BitNet head vs FP32 head, mn20 FP32 | " + " | ".join(get("head", t, "mn20", "FP32", "bitnet", metric) for t, _ in TASKS) + " |")
        out.append("| Q4 head vs FP32 head, mn20 FP32 | " + " | ".join(get("head", t, "mn20", "FP32", "4", metric) for t, _ in TASKS) + " |")
    (a.out / "cross_table_readme.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
