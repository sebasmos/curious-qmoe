"""Check 1: compare recompute.py points with Stage 2's analysis/<dataset>/front_comparisons.csv
(primary family). Prints max absolute differences and every row whose comparator or
verdict differs.

    python compare_to_stage2.py STAGE2_DIR AUDIT_OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

s2, out = Path(sys.argv[1]), Path(sys.argv[2])
lines = []
for ds in ("esc50", "esc5", "urban"):
    mine = pd.read_csv(out / "bitops" / ds / "points.csv")
    fc = pd.read_csv(s2 / "analysis" / ds / "front_comparisons.csv")
    fc = fc[fc.family == "primary"].copy()
    fc["cheap"] = fc.method.str.split(">").str[0]
    fc["cond"] = fc.condition
    m = mine.merge(fc, left_on=["cond", "cheap", "point"], right_on=["cond", "cheap", "operating_point"],
                   how="outer", indicator=True, suffixes=("", "_s2"))
    only_mine = m[m._merge == "left_only"]
    only_s2 = m[m._merge == "right_only"]
    b = m[m._merge == "both"].copy()
    fam = int(b.family_size.iloc[0])
    b["pt_b"] = (b.p_t * fam).clip(upper=1)
    b["pw_b"] = (b.p_w * fam).clip(upper=1)
    b["mine_verdict"] = np.where((b.diff_pts > 0) & (b.pt_b < 0.05) & (b.pw_b < 0.05), "PASS", "FAIL")
    lines.append(f"## {ds}: {len(b)} primary points matched, {len(only_mine)} only in the audit, "
                 f"{len(only_s2)} only in Stage 2; Stage 2 family size {fam}")
    lines.append(f"max |F1 diff| {np.abs(b.f1 - b.f1_macro).max():.2e}; max |bit-ops diff| "
                 f"{np.abs(b.bitops - b.bitops_per_clip).max():.3e}; max |paired diff| "
                 f"{np.abs(b.diff_pts / 100 - b.mean_diff).max():.2e}; max |p_t diff| "
                 f"{np.abs(b.pt_b - b.p_ttest_bonferroni).max():.2e}; max |p_w diff| "
                 f"{np.abs(b.pw_b - b.p_wilcoxon_bonferroni).max():.2e}")
    cm = b[b.comparator != b.comparator_method]
    vd = b[b.mine_verdict != b.verdict]
    lines.append(f"comparator differs in {len(cm)} rows; verdict differs in {len(vd)} rows")
    for _, r in pd.concat([cm, vd]).drop_duplicates(["cond", "cheap", "point"]).iterrows():
        lines.append(f"  {r.cond} {r.cheap} {r.point}: audit {r.comparator} {r.comp_f1:.4f} diff {r.diff_pts:.3f} "
                     f"{r.mine_verdict}; stage2 {r.comparator_method} {r.comparator_f1_macro:.4f} "
                     f"diff {100 * r.mean_diff:.3f} {r.verdict}")
    if len(only_mine):
        lines.append("  only in the audit: " + ", ".join(sorted(set(only_mine.cond))))
print("\n".join(lines))
