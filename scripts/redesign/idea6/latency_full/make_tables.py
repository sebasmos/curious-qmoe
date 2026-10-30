"""Markdown tables from tables/steps.csv and tables/cascade_latency.csv -> tables/tables.md.

    python scripts/make_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import pandas as pd  # noqa: E402

ORDER = ["mel_front_end", "mn_front_end", "fbank_512", "fbank_400",
         "effnet_torch_fp32", "effnet_torch_int8_fx", "effnet_ort_fp32", "effnet_ort_int8_qdq",
         "mn10_torch_fp32", "mn10_ort_fp32", "mn10_ort_int8", "mn20_torch_fp32", "mn20_ort_fp32", "mn20_ort_int8",
         "head_cheap_mn20_esc50_4bit", "head_cheap_mn10_urban_fp32", "head_cheap_esc50_fp32",
         "head_cheap_urban_8bit", "head_ast_esc50_fp32", "head_ast_urban_fp32",
         "ast_512_fp32", "ast_512_n4_kleidiai", "ast_512_int8_dynamic",
         "ast_400_fp32", "ast_400_n4_kleidiai", "ast_400_int8_dynamic"]
LABEL = {"mel_front_end": "EfficientNet mel image front end", "mn_front_end": "EfficientAT front end (resample, log mel)",
         "fbank_512": "AST fbank, 512 frames", "fbank_400": "AST fbank, 400 frames",
         "effnet_torch_fp32": "EfficientNet-B3, PyTorch FP32", "effnet_torch_int8_fx": "EfficientNet-B3, PyTorch int8 (FX, qnnpack)",
         "effnet_ort_fp32": "EfficientNet-B3, ONNX Runtime FP32", "effnet_ort_int8_qdq": "EfficientNet-B3, ONNX Runtime int8 (QDQ)",
         "mn10_torch_fp32": "EfficientAT mn10, PyTorch FP32", "mn10_ort_fp32": "EfficientAT mn10, ONNX Runtime FP32",
         "mn10_ort_int8": "EfficientAT mn10, ONNX Runtime int8 (QDQ, fold 1)",
         "mn20_torch_fp32": "EfficientAT mn20, PyTorch FP32", "mn20_ort_fp32": "EfficientAT mn20, ONNX Runtime FP32",
         "mn20_ort_int8": "EfficientAT mn20, ONNX Runtime int8 (QDQ, fold 1)",
         "head_cheap_mn20_esc50_4bit": "cheap head, mn20 ESC-50/50 4-bit (Stage 3c)",
         "head_cheap_mn10_urban_fp32": "cheap head, mn10 UrbanSound8K FP32 (Stage 3b)",
         "head_cheap_esc50_fp32": "cheap head, EfficientNet ESC-50/50 FP32 (Stage 2)",
         "head_cheap_urban_8bit": "cheap head, EfficientNet UrbanSound8K 8-bit (Stage 2)",
         "head_ast_esc50_fp32": "AST head, ESC-50/50 FP32", "head_ast_urban_fp32": "AST head, UrbanSound8K FP32",
         "ast_512_fp32": "short AST 512, FP32", "ast_512_n4_kleidiai": "short AST 512, n4 KleidiAI",
         "ast_512_int8_dynamic": "short AST 512, int8 dynamic",
         "ast_400_fp32": "short AST 400, FP32", "ast_400_n4_kleidiai": "short AST 400, n4 KleidiAI",
         "ast_400_int8_dynamic": "short AST 400, int8 dynamic"}
SYS = {"torch_fp32": "PyTorch FP32", "ort_fp32": "ONNX Runtime FP32", "ort_int8": "ONNX Runtime int8"}
ASTK = {"fp32": "FP32", "n4_kleidiai": "n4 KleidiAI", "int8_dynamic": "int8 dynamic"}


def fmt(v, d=1):
    return f"{v:.{d}f}"


def main():
    S = pd.read_csv(C.OUT / "tables/steps.csv")
    Cc = pd.read_csv(C.OUT / "tables/cascade_latency.csv")
    threads = sorted(S.threads.unique())
    lines = ["# Tables", "", "## Step latencies, median ms (IQR), per thread count", "",
             "Batch 1; 50 warm-up and 500 timed passes per block; the load before and after each block is in "
             "`raw/*_summary_t<threads>.json` and `run_manifest.json`.", ""]
    for ds in ("esc50", "urban8k"):
        clip = "ESC-50 clip (5 s)" if ds == "esc50" else "UrbanSound8K clip (4 s; 4.04 s padded for EfficientAT)"
        lines += [f"### {clip}", "", "| step | " + " | ".join(f"{t} thread{'s' if t > 1 else ''}" for t in threads) + " |",
                  "|---|" + "---|" * len(threads)]
        for step in ORDER:
            r = S[(S.step == step) & (S.dataset.isin([ds, "both"]))]
            if r.empty:
                continue
            cells = []
            for t in threads:
                x = r[r.threads == t]
                d = 2 if x.median_ms.iloc[0] < 10 else 1
                cells.append(f"{fmt(x.median_ms.iloc[0], d)} ({fmt(x.iqr_ms.iloc[0], d)})" if len(x) else "")
            lab = LABEL[step] + (" (same input for both clips)" if r.dataset.iloc[0] == "both" and ds == "esc50" else
                                 " (same as ESC-50)" if r.dataset.iloc[0] == "both" else "")
            lines.append(f"| {lab} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Cascades against the statics: mean / median / p95 ms per clip", "",
              "Stage 3c picks: ESC-50/50 mn20 -> w4g128@512 (P3, field 17.5%, clean 6.5%); UrbanSound8K mn10 -> "
              "w8g64@400 (S4, field 40.8%, clean 30.0%). The row marked * runs the AST on the real kernel closest to "
              "the pick's fake-quantized tier (n4 KleidiAI for w4g128, int8 dynamic for w8g64). "
              "`tables/cascade_latency.csv` has every combination.", ""]
    for t in threads:
        for task, name in (("esc50", "ESC-50/50"), ("urban", "UrbanSound8K")):
            d = Cc[(Cc.threads == t) & (Cc.task == task)]
            lines += [f"### {name}, {t} thread{'s' if t > 1 else ''}", "",
                      "| system | CNN runtime | AST kernel | clean | field |", "|---|---|---|---|---|"]
            for _, r in d[d.system.str.startswith("static short AST")].iterrows():
                v = f"{fmt(r.mean_ms)} / {fmt(r.median_ms)} / {fmt(r.p95_ms)}"
                lines.append(f"| {r.system} | | {ASTK[r.ast]} | {v} | {v} |")
            for _, r in d[d.system.str.startswith("static CNN")].iterrows():
                v = f"{fmt(r.mean_ms)} / {fmt(r.median_ms)} / {fmt(r.p95_ms)}"
                lines.append(f"| {r.system} | {SYS[r.cnn]} | | {v} | {v} |")
            cas = d[d.system == "cascade"]
            for cn in SYS:
                for an in ASTK:
                    rows = cas[(cas.cnn == cn) & (cas.ast == an)]
                    if rows.empty:
                        continue
                    cells = []
                    for cond in ("clean", "field"):
                        r = rows[rows.condition == cond].iloc[0]
                        cells.append(f"{fmt(r.mean_ms)} / {fmt(r.median_ms)} / {fmt(r.p95_ms)}")
                    star = " *" if bool(rows.ast_closest_to_pick.iloc[0]) else ""
                    lines.append(f"| cascade{star} | {SYS[cn]} | {ASTK[an]} | {cells[0]} | {cells[1]} |")
            lines.append("")
    (C.OUT / "tables/tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
