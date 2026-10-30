"""T9 verdict from the three tasks' selected.csv: Holm over tasks, the +1.0 point rule, tables for the README.

    python summary.py RESULTS_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1])
IDEA6 = Path(__file__).resolve().parents[1] / "idea6" / "stage2"
sys.path.insert(0, str(IDEA6))
from idea5b_common import holm  # noqa: E402

TASKS = ["esc50", "esc5", "urban"]
NAMES = {"esc50": "ESC-50, 50 classes", "esc5": "ESC-50, five groups", "urban": "UrbanSound8K"}
sel = {t: pd.read_csv(R / "analysis" / t / "selected.csv") for t in TASKS}
pick = {t: json.loads((R / "analysis" / t / "pick.json").read_text()) for t in TASKS}
rows = []
verdict = {}
for fam, cond in (("H1", "field"), ("H2", "clean")):
    sub = pd.DataFrame([sel[t][sel[t].family == fam].iloc[0] for t in TASKS])
    sub.index = TASKS
    for col in ("hull_p_t_fold", "hull_p_nb"):
        sub["holm_" + col] = holm(sub[col].to_numpy())
    sub["reaches_1pt"] = sub.hull_diff >= 0.01
    sub["pass_task"] = sub.reaches_1pt & (sub.holm_hull_p_nb < 0.05)
    verdict[fam] = {"condition": cond, "tasks_reaching_1pt": int(sub.reaches_1pt.sum()),
                    "tasks_passing": int(sub.pass_task.sum()),
                    "per_task": {t: {"diff_pts": float(100 * sub.loc[t, "hull_diff"]),
                                     "holm_p_nb": float(sub.loc[t, "holm_hull_p_nb"]),
                                     "holm_p_t": float(sub.loc[t, "holm_hull_p_t_fold"])} for t in TASKS}}
    for t in TASKS:
        r = sub.loc[t]
        rows.append({"hyp": fam, "task": NAMES[t], "pick": " ".join(str(v) for v in pick[t]["design"]),
                     "f1": r.f1, "cost_T": r.cost / 1e12, "rate1": r.rate1, "rate2": r.rate2,
                     "hull": f"{r.hull_lo} / {r.hull_hi} ({r.lam:.2f})", "hull_f1": r.hull_f1,
                     "diff_pts": 100 * r.hull_diff, "fold_range": f"{100 * r.hull_diff_fold_min:+.2f} to {100 * r.hull_diff_fold_max:+.2f}",
                     "p_t": r.hull_p_t_fold, "holm_t": r["holm_hull_p_t_fold"], "p_nb": r.hull_p_nb, "holm_nb": r["holm_hull_p_nb"],
                     "p_signflip": r.hull_p_signflip})
verdict["rule"] = "PASS if H1 (field) has >= 2 tasks with diff >= +1.0 point and Holm(NB) p < 0.05"
verdict["result"] = "PASS" if verdict["H1"]["tasks_passing"] >= 2 else "CLOSED"
(R / "analysis" / "verdict.json").write_text(json.dumps(verdict, indent=1))
T = pd.DataFrame(rows)
T.to_csv(R / "analysis" / "hypotheses.csv", index=False)

# descriptive: all designs
al = []
for t in TASKS:
    d = pd.read_csv(R / "analysis" / t / "all_designs.csv")
    for c in ("field", "clean"):
        g = d[d.condition == c]
        b = g.sort_values("hull_diff", ascending=False).iloc[0]
        al.append({"task": NAMES[t], "condition": c, "n_designs": len(g), "n_diff_ge_1pt": int((g.hull_diff >= 0.01).sum()),
                   "n_diff_gt_0": int((g.hull_diff > 0).sum()), "best_diff_pts": 100 * b.hull_diff,
                   "best_design": f"{b.setting} {b.rule} {b.b1} {b.b2}", "best_p_t_raw": b.hull_p_t_fold,
                   "best_cost_T": b.cost / 1e12})
A_ = pd.DataFrame(al)
A_.to_csv(R / "analysis" / "all_designs_summary.csv", index=False)

# sensitivities of the picks
sens = []
for t in TASKS:
    s = sel[t]
    for fam in ("H1", "sens_plain_nested_price", "sens_nested_statics_k_plus_1"):
        g = s[s.family == fam]
        for _, r in g.iterrows():
            sens.append({"task": NAMES[t], "family": fam, "condition": r.condition, "diff_pts": 100 * r.hull_diff,
                         "cost_T": r.cost / 1e12, "p_t": r.hull_p_t_fold})
S_ = pd.DataFrame(sens)
for t in TASKS:   # H1 is the field row; add H2 rows under the same name
    pass
S_.to_csv(R / "analysis" / "sensitivities.csv", index=False)

diag = pd.concat([pd.read_csv(R / "analysis" / t / "tier_diagnostics.csv").assign(task=NAMES[t]) for t in TASKS])
diag.to_csv(R / "analysis" / "tier_diagnostics.csv", index=False)

def md(df, fmt=None):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((f"{r[c]:.4g}" if isinstance(r[c], (float, np.floating)) else str(r[c])) for c in cols) + " |")
    return "\n".join(out)

with open(R / "analysis" / "summary.md", "w") as f:
    f.write(f"# T9 summary\n\nResult: {verdict['result']}. {verdict['rule']}.\n\n## Picks against the hull\n\n{md(T)}\n\n")
    f.write(f"## Every design (descriptive)\n\n{md(A_)}\n\n## Sensitivities of the picks\n\n{md(S_)}\n\n")
    f.write(f"## Where clips end and accuracy by precision (pick)\n\n{md(diag)}\n")
print(json.dumps(verdict, indent=1))
print(T[["hyp", "task", "pick", "f1", "cost_T", "hull_f1", "diff_pts", "p_t", "p_nb", "holm_nb"]].to_string())
