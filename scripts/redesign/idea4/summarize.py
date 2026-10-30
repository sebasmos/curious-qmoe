"""Compact table and Stage 1 go-rule verdict from router_diagnostics output.

Go rule (ESC-50, seed 0, 5 folds): the routing uncertainty predicts the
model's errors with AUROC >= 0.7 (mean of per-fold AUROCs), the uncertainty
rule changes >= 10% of routing decisions (rule on vs off, same weights), and
macro F1 >= the uniform control's (fold mean, same seed and folds).
    python summarize.py diag.json [control_name]
"""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
ctrl = sys.argv[2] if len(sys.argv) > 2 else "uniform_ctrl"
f1_ctrl = d[ctrl]["mean"]["f1_macro"] if ctrl in d else None
NAMES = ["BitNet", "Q4", "Q8"]


def fmt(x, p=3):
    return "n/a" if x is None else f"{x:.{p}f}"


rows, verdict = [], {}
print("| arm | macro F1 (sd) | F1 rule off | shares B/Q4/Q8 | rule changes | AUROC u, model err | AUROC u, BitNet err "
      "| AUROC MC var, model err | Q8 share easiest / hardest decile | expert BOPs x Q4 | total BOPs x Q4 |")
print("|" + "---|" * 11)
for name, r in d.items():
    m, folds = r["mean"], r["folds"]
    sd = float(np.std([f["f1_macro"] for f in folds]))
    dec = m["hi_share_by_difficulty_decile"]
    au = m.get("auroc_rule_u_model_err")
    print(f"| {name} | {m['f1_macro']:.4f} ({sd:.4f}) | {m['f1_macro_rule_off']:.4f} | "
          f"{'/'.join(f'{s:.2f}' for s in m['shares'])} | {m['rule_changed_share']:.3f} | {fmt(au)} | "
          f"{fmt(m.get('auroc_rule_u_cheap_err'))} | {fmt(m.get('auroc_mc_variance_model_err'))} | "
          f"{dec[0]:.2f} / {dec[-1]:.2f} | {fmt(m.get('expert_bops_rel_q4'), 2)} | {fmt(m.get('bops_rel_q4'), 1)} |")
    if f1_ctrl is not None and name != ctrl:
        checks = {"auroc>=0.7": au is not None and au >= 0.7, "changes>=0.10": m["rule_changed_share"] >= 0.10,
                  "f1>=uniform": m["f1_macro"] >= f1_ctrl}
        verdict[name] = {"checks": checks, "go": all(checks.values())}
print()
print("per fold macro F1:")
for name, r in d.items():
    print(f"  {name}: " + ", ".join(f"{f['f1_macro']:.4f}" for f in r["folds"]))
print("per fold AUROC(rule u, model err) / rule changes:")
for name, r in d.items():
    print(f"  {name}: " + ", ".join(f"{fmt(f.get('auroc_rule_u_model_err'))}/{f['rule_changed_share']:.3f}" for f in r["folds"]))
print("reference AUROCs (signals that need the experts' outputs): MoE conf for model err / BitNet conf for BitNet err:")
for name, r in d.items():
    m = r["mean"]
    print(f"  {name}: {fmt(m.get('auroc_ref_model_conf_model_err'))} / {fmt(m.get('auroc_ref_cheap_conf_cheap_err'))}")
print("Q8 share by difficulty decile (1 = easiest):")
for name, r in d.items():
    print(f"  {name}: " + " ".join(f"{x:.2f}" for x in r["mean"]["hi_share_by_difficulty_decile"]))
print("expert accuracy on every clip (BitNet, Q4, Q8):")
for name, r in d.items():
    print(f"  {name}: " + " ".join(f"{x:.3f}" for x in r["mean"]["expert_acc"]))
print()
print(f"uniform control F1 {fmt(f1_ctrl, 4)}")
for name, v in verdict.items():
    print(f"  {name}: {'GO' if v['go'] else 'no go'} {v['checks']}")
json.dump(verdict, open(sys.argv[1].replace(".json", "_verdict.json"), "w"), indent=1)
