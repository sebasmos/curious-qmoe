"""Tables of the idea 6 option B analysis (analysis/tau/*.csv, int8_*.csv when present) -> analysis/summary.md and
analysis/verdict.json. Every number comes from those files.

    python scripts/redesign/idea6/optionB/summary_optb.py --r R
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HEAD = {"esc": "FP32", "8": "8-bit", "4": "4-bit", "bitnet": "BitNet"}
DEVICES = ["b", "c", "s1", "s2", "s3", "s4", "s5", "s6"]


def T(x):
    return f"{x / 1e12:.3f}"


def fp(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "-"
    return f"{p:.2g}" if p < 0.01 else f"{p:.3f}"


def pts(x):
    return f"{100 * x:+.2f}"


def design_str(d):
    m, k, e, r, b = d
    return f"{m} W8A8 + {HEAD[k]} head -> AST {e} (FP32 head), {r}, {b}"


def md(df, cols, names=None):
    names = names or cols
    lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--r", required=True, type=Path)
    a = ap.parse_args(argv)
    an = a.r / "analysis" / "tau"
    H = pd.read_csv(an / "hypotheses.csv")
    SEL = pd.read_csv(an / "selected.csv")
    HR = pd.read_csv(an / "headroom.csv")
    S = pd.read_csv(an / "static.csv")
    E = pd.read_csv(an / "escalation.csv")
    AU = pd.read_csv(an / "auroc.csv")
    SO = pd.read_csv(an / "selection.csv")
    E2 = pd.read_csv(an / "e2.csv")
    picks = json.loads((an / "picks.json").read_text())
    meta = json.loads((an / "meta.json").read_text())
    pick = picks["pick"]["design"]
    m, k, e, r, b = pick
    out = ["# Idea 6 option B: summary tables (TAU 2020 Mobile)", "",
           f"Pick (clean inner validation, {picks['n_designs']} designs, {picks['n_positive_gain']} with a positive gain): "
           f"{design_str(pick)}; inner-validation gain {100 * picks['pick']['gain_iv']:+.2f} points over the hull "
           f"{picks['pick']['hull_iv'][0]} / {picks['pick']['hull_iv'][1]} ({picks['pick']['hull_iv'][2]:.2f}); runner-up "
           f"{design_str(picks['runner_up']['design'])} at {100 * picks['runner_up']['gain_iv']:+.2f}.", ""]
    # H1, H2
    out += ["## Primary: H1 (clean) and H2 (shift), Holm over the two", ""]
    rows = []
    for _, h in H.iterrows():
        rows.append({"hypothesis": h.hypothesis, "condition": h.condition, "F1": f"{h.f1:.4f}", "T": T(h.cost), "rate": f"{h.rate:.2f}",
                     "hull pair (share)": h.hull_pair, "hull F1": f"{h.hull_f1:.4f}", "diff (pts)": pts(h["diff"]), "fold diffs (pts)":
                     ", ".join(f"{100 * v:+.1f}" for v in json.loads(h.fold_diffs)), "t p raw / Holm": f"{fp(h.p_t_fold)} / {fp(h.p_t_fold_holm)}",
                     "NB p raw / Holm": f"{fp(h.p_nb)} / {fp(h.p_nb_holm)}", "sign-flip": fp(h.p_signflip), "verdict": h.verdict})
    out += [md(pd.DataFrame(rows), list(rows[0])), ""]
    # S1 headroom
    out += ["## S1: headroom between the cheap tier and the AST (best static of each kind on the condition's mean F1)", ""]
    rows = []
    for _, h in HR.iterrows():
        rows.append({"condition": h.condition, "cheap kind": h.cheap_kind, "cheap static": h.cheap, "F1": f"{h.cheap_f1:.4f}", "T": T(h.cheap_cost),
                     "AST static": h.ast, "F1 ": f"{h.ast_f1:.4f}", "T ": T(h.ast_cost), "AST minus cheap (pts)": pts(h.headroom),
                     "95% CI": f"{100 * h.ci_lo:+.1f} to {100 * h.ci_hi:+.1f}", "t p": fp(h.p_t_fold)})
    out += [md(pd.DataFrame(rows), list(rows[0])), ""]
    # E1 front
    out += ["## E1: static front (mean macro F1 over seeds and folds; priced T)", ""]
    sm = S.groupby(["system", "condition"]).f1.mean().unstack()
    cost = S.groupby("system").cost.mean()
    conds = ["clean", "shift"] + [f"dev:{d}" for d in DEVICES]
    rows = []
    for sysn in cost.sort_values().index:
        rows.append({"static": sysn, "T": T(cost[sysn]), **{c: f"{sm.loc[sysn, c]:.3f}" for c in conds if c in sm.columns}})
    out += [md(pd.DataFrame(rows), list(rows[0])), ""]
    # the pick on every condition, baselines
    out += ["## The pick on every condition (selected.csv); baselines on clean and shift", ""]
    rows = []
    for _, s in SEL.iterrows():
        row = {"condition": s.condition, "F1": f"{s.f1:.4f}", "T": T(s.cost), "rate": f"{s.rate:.2f}", "hull": f"{s.hull_lo} / {s.hull_hi} ({s.lam:.2f})",
               "hull F1": f"{s.hull_f1:.4f}", "diff": pts(s.hull_diff), "t p": fp(s.hull_p_t_fold)}
        if "random_f1" in s and not pd.isna(s.random_f1):
            row.update({"random gate": f"{s.random_f1:.4f} ({pts(s.random_diff)}, p {fp(s.random_p_t_fold)})",
                        "shift detector gate": f"{s.detector_f1:.4f} ({pts(s.detector_diff)}, p {fp(s.detector_p_t_fold)})",
                        "best static <= cost": f"{s.best_static} {s.best_static_f1:.4f} ({pts(s.static_diff)}, p {fp(s.static_p_t_fold)})",
                        "oracle at count": f"{s.oracle_k_f1:.4f}", "NI 1 pt vs best AST": f"{s.ni_ast} {s.ni_ast_f1:.4f} ({pts(s.ni_diff)}, p {fp(s.ni_p)})"})
        rows.append(row)
    cols = list(rows[0])
    out += [md(pd.DataFrame(rows).fillna(""), cols), ""]
    # S2 humility
    out += [f"## S2: humility of the pick's cheap tier ({m} W8A8 + {HEAD[k]} head): escalation rate with the clean thresholds", ""]
    Ek = E[(E.cheap_bb == m) & (E.cheap == k)].groupby(["budget", "condition"]).rate.mean().unstack()
    rows = []
    for bud in ["b05", "b10", "b20", "b30", "b50"]:
        if bud not in Ek.index:
            continue
        rr = Ek.loc[bud]
        rises = sum(rr[f"dev:{d}"] > rr["clean"] for d in DEVICES)
        rows.append({"budget": bud, **{c: f"{rr[c]:.3f}" for c in conds}, "devices above clean": f"{rises} of 8"})
    out += [md(pd.DataFrame(rows), list(rows[0])), ""]
    AUk = AU[(AU.cheap_bb == m) & (AU.cheap == k) & (AU.expensive == e)].groupby("condition")[["auroc_wrong", "auroc_fixable", "error"]].mean()
    rows = [{"condition": c, "AUROC cheap wrong": f"{AUk.loc[c, 'auroc_wrong']:.3f}", "AUROC cheap wrong, AST right": f"{AUk.loc[c, 'auroc_fixable']:.3f}",
             "cheap error rate": f"{AUk.loc[c, 'error']:.3f}"} for c in conds if c in AUk.index]
    out += ["Gate AUROC of 1 - margin (pooled over folds, mean over seeds):", "", md(pd.DataFrame(rows), list(rows[0])), ""]
    allk = E.groupby(["cheap_bb", "cheap", "budget", "condition"]).rate.mean().unstack()
    rows = []
    for (mm, kk, bud), rr in allk.iterrows():
        if bud != b:
            continue
        rows.append({"cheap tier": f"{mm} W8A8 + {HEAD[kk]}", "clean": f"{rr['clean']:.3f}", "shift": f"{rr['shift']:.3f}",
                     "devices above clean": f"{sum(rr[f'dev:{d}'] > rr['clean'] for d in DEVICES)} of 8"})
    out += [f"Every cheap W8A8 tier at the pick's budget {b}:", "", md(pd.DataFrame(rows), list(rows[0])), ""]
    # E2
    out += ["## E2: every design on clean and shift against the hull (Holm over all)", ""]
    rows = []
    for c in ["clean", "shift"]:
        g = E2[E2.condition == c]
        best = g.sort_values("hull_diff", ascending=False).iloc[0]
        rows.append({"condition": c, "designs": len(g), "positive mean": int((g.hull_diff > 0).sum()), "pass Holm over all": int((g.p_holm_all < 0.05).sum()),
                     "best": design_str([best.cheap_bb, best.cheap, best.expensive, best.rule, best.budget]), "best diff": pts(best.hull_diff),
                     "best p raw / Holm": f"{fp(best.hull_p_t_fold)} / {fp(best.p_holm_all)}"})
    out += [md(pd.DataFrame(rows), list(rows[0])), ""]
    # S4 int8
    verdict = {"pick": pick, "H1": H[H.hypothesis == "H1"].iloc[0].to_dict(), "H2": H[H.hypothesis == "H2"].iloc[0].to_dict(),
               "headroom": HR.to_dict("records"), "n_rows": meta["n_rows"]}
    if (an / "int8_ni.csv").exists():
        NI = pd.read_csv(an / "int8_ni.csv")
        CR = pd.read_csv(an / "int8_cascade_vs_hull.csv")
        i8 = json.loads((an / "int8_summary.json").read_text())
        out += [f"## S4: real int8 {m} (ONNX Runtime, {i8['calibration_method']} calibration), the pick's head ({HEAD[k]}) with its scaler refit on int8 features", ""]
        rows = [{"way": n.way, "condition": n.condition, "int8 F1": f"{n.f1_int8:.4f}", "W8A8 F1": f"{n.f1_w8a8:.4f}", "diff (pts)": pts(n.ni_diff),
                 "NI p raw / Holm": f"{fp(n.ni_p)} / {fp(n.ni_p_holm)}", "NB p": fp(n.ni_p_nb), "verdict": n.verdict} for _, n in NI.iterrows()]
        out += [md(pd.DataFrame(rows), list(rows[0])), ""]
        rows = [{"way": c.way, "condition": c.condition, "F1": f"{c.f1:.4f}", "T": T(c.cost), "rate": f"{c.rate:.2f}", "hull": f"{c.hull_lo} / {c.hull_hi} ({c.lam:.2f})",
                 "hull F1": f"{c.hull_f1:.4f}", "diff": pts(c["diff"]), "t p": fp(c.p_t_fold), "NB p": fp(c.p_nb)} for _, c in CR.iterrows()]
        out += ["The pick with the int8 tier against the hull with the int8 statics added (raw p):", "", md(pd.DataFrame(rows), list(rows[0])), "",
                "Escalation with the clean thresholds (int8 = scaler refit, int8a = FP32 scaler): " + json.dumps(i8["escalation"]), "",
                "Gate AUROC cheap wrong: " + json.dumps(i8["auroc"]), "", "Embedding error: " + json.dumps(i8["embedding_error"]), ""]
        verdict["S4"] = NI.to_dict("records")
        verdict["S4_cascade"] = CR.to_dict("records")
    (a.r / "analysis" / "summary.md").write_text("\n".join(out))
    (a.r / "analysis" / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    print("\n".join(out[:12]))


if __name__ == "__main__":
    main()
