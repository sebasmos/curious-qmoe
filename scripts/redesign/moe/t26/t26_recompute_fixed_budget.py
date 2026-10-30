"""T26 check written after the d3 verdict (not part of the plan): the own recompute (t26_recompute.py, unchanged) with
S1's budget forced to the analysis's choice (20%), because the recompute's linear inner-validation hull picked 50% on a
near tie.   python t26_recompute_fixed_budget.py R26 d3 0.2 0 1 2"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t26_recompute as RC  # noqa: E402

R, D, rate, seeds = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), [int(s) for s in sys.argv[4:]]
orig = RC.choose


def fixed(IV, keys):
    pick, cuts, SEL = orig(IV, keys)
    pick["s1"] = (rate,)
    cuts["s1"] = {k: RC.cut_at(RC.marg(IV[k][1]["mn10"]), rate) for k in keys}
    return pick, cuts, SEL


RC.choose = fixed
res, _ = RC.part_c(R, D, seeds, False)
out = {k: v for k, v in res.items() if k.startswith("s1|")}
(R / "tables" / "c" / D / "recompute_at_analysis_budget.json").write_text(
    json.dumps({"note": f"own recompute with S1 forced to the analysis budget ({rate:.0%})", **out}, indent=1))
print(json.dumps(out, indent=1))
