"""T19 step 19.3: the cost table from code (ANALYSIS_PLAN_T19.md B4) -> <results>/tables/costs.csv.
Checks the audit's totals first (W8A8 with 16-bit P 9.45 T, W4A8 6.15 T at 1024 frames, unpriced, no front end or head)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t19_common import CONFIGS, LEGACY_SHORT, LEGACY_TAU, STATICS_NEW, TASKS, backbone_bitops, head_bitops, results_dir  # noqa: E402
from QWave.costs import ast_t19_bitops  # noqa: E402

w8 = ast_t19_bitops(1024, 8, 8, 16, group=None, asym=False)["unpriced"] / 1e12
w4 = ast_t19_bitops(1024, 4, 8, 16, group=None, asym=False)["unpriced"] / 1e12
assert abs(w8 - 9.45) < 0.005 and abs(w4 - 6.15) < 0.005, (w8, w4)
print(f"audit check: W8A8 P16 {w8:.4f} T, W4A8 {w4:.4f} T (1024 frames, unpriced)")
rows = []
for task in ("esc50", "urban", "tau"):
    t = TASKS[task]
    ref = backbone_bitops(task, "w8a8")
    names = STATICS_NEW + ["n4a4sr1"] + ["L:" + c for c in (LEGACY_TAU if task == "tau" else LEGACY_SHORT)]
    for c in names:
        pr, un = backbone_bitops(task, c, True), backbone_bitops(task, c, False)
        rows.append(dict(task=task, frames=t["length"], config=c, backbone_priced_T=pr / 1e12, backbone_unpriced_T=un / 1e12,
                         fbank_T=t["fbank"] / 1e12, head_T=head_bitops(t["K"]) / 1e12,
                         system_priced_T=(pr + t["fbank"] + head_bitops(t["K"])) / 1e12,
                         ratio_to_w8a8_priced=pr / ref, ratio_to_w8a8_unpriced=un / backbone_bitops(task, "w8a8", False)))
    # the plan's section 4.2 table at 1024 frames (no rescale terms): N4A4 and W4A6 against W8A8 with 8-bit P
for name, wb, ab in (("W8A8 P8", 8, 8), ("N4A4 P8", 4, 4), ("W4A6 P8", 4, 6), ("W8A8 P16 (stored rule)", 8, None)):
    c = ast_t19_bitops(1024, wb, ab or 8, 16 if ab is None else 8, group=None, asym=False)
    print(f"4.2 check {name}: linear {c['linear']/1e12:.3f} qk {c['qk']/1e12:.3f} pv {c['pv']/1e12:.3f} patch {c['patch']/1e12:.3f} "
          f"total {c['unpriced']/1e12:.3f} T")
df = pd.DataFrame(rows)
out = results_dir() / "tables/costs.csv"
df.to_csv(out, index=False, float_format="%.6g")
print(df[df.task != "tau"].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(df[df.task == "tau"].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
