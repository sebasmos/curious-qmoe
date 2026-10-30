"""Idea 6 option C summary over the three tasks: H1 and H2 (Holm over 3), S1 baselines, S2 humility, S3 headroom,
S4 unpriced view, S5 new tiers as statics, E1 three tiers, E2 over every design, E3 reproduction, the static front.
Writes analysis/summary.md, analysis/verdict.json, analysis/hypotheses.csv and analysis/*.csv.

    python scripts/redesign/idea6/optionC/summary_c.py --r R [--unpriced analysis_unpriced]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage2"))
from idea5b_common import holm  # noqa: E402

FAM = ["esc50", "esc5", "urban"]
NAMES = {"esc50": "ESC-50, 50 classes", "esc5": "ESC-50, five fixed groups", "urban": "UrbanSound8K"}
HEAD = {"esc": "FP32", "8": "8-bit", "4": "4-bit", "bitnet": "BitNet"}
SNR_STEPS = [(f"{k}__{a}db", f"{k}__{b}db") for k in ("bg", "pink", "white") for a, b in ((20, 10), (10, 5), (5, 0))]
STEPS_FROM_CLEAN = [("clean", f"{k}__20db") for k in ("bg", "pink", "white")]
CONDS = ["clean"] + [f"{kk}__{s}db" for kk in ("bg", "pink", "white") for s in (20, 10, 5, 0)] + ["field"]


def design_str(r):
    return (f"{r['cheap_src']} {str(r['cheap_bb']).upper()} + {HEAD[str(r['cheap'])]} head -> {r['expensive']}, "
            f"{r['rule']}, {r['budget']}")


def design3_str(r):
    return (f"{r['cheap_src']} {str(r['cheap_bb']).upper()} + {HEAD[str(r['cheap'])]} head -> {r['mid']} -> {r['top']}, "
            f"{r['rule']}, {r['budget']}, b2 {r['b2']}")


def T(x):
    return f"{x / 1e12:.3f}"


def fmt_p(p):
    if p is None or pd.isna(p):
        return "-"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def pts(x):
    return f"{100 * x:+.2f}"


def holm_rows(rows, pcol, prefix):
    adj = holm([r[pcol] for r in rows]) if rows else []
    for r, h in zip(rows, adj):
        r[f"{prefix}_holm"] = float(h)
    return rows


def hull_str(r):
    if r["lam"] == 0.0:
        return f"{r['hull_lo']} (static)"
    return f"{r['hull_lo']} / {r['hull_hi']} ({r['lam']:.2f})"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", required=True, type=Path)
    ap.add_argument("--unpriced", type=Path, default=None, help="analysis folder of the unpriced run (S4)")
    a = ap.parse_args(argv)
    an = a.r / "analysis"
    sel = {t: pd.read_csv(an / t / "selected.csv") for t in FAM}
    sel3 = {t: pd.read_csv(an / t / "selected3.csv") for t in FAM}
    picks = {t: json.loads((an / t / "picks.json").read_text()) for t in FAM}
    meta = {t: json.loads((an / t / "meta.json").read_text()) for t in FAM}
    md = ["# Idea 6 option C summary", ""]
    verdict, hyp = {}, []

    def row_of(t, fam, cond):
        g = sel[t]
        g = g[(g.family == fam) & (g.condition == cond)]
        r = g.iloc[0].to_dict()
        r["task"] = t
        return r

    # ── H1, H2 ──────────────────────────────────────────────────────────────
    for fam, cond, title in (("H1", "field", "H1: the pick on the field mixture against the time-sharing hull (priced view)"),
                             ("H2", "clean", "H2: the same pick on clean test folds against the clean hull")):
        rows = holm_rows([row_of(t, fam, cond) for t in FAM], "hull_p_t_fold", "t")
        rows = holm_rows(rows, "hull_p_nb", "nb")
        md += [f"## {title}", "",
               "| task | design | tier new? | F1 at T | escalated | hull pair (share of upper) | hull F1 | diff (pts) | fold diffs | t p raw / Holm | NB p raw / Holm | sign-flip p | verdict |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            v = "PASS" if r["hull_diff"] > 0 and r["t_holm"] < 0.05 else "FAIL"
            r["verdict"], r["family"] = v, fam
            verdict[f"{fam}_{r['task']}"] = {"verdict": v, "diff_pts": 100 * r["hull_diff"], "p_t_fold": r["hull_p_t_fold"],
                                             "holm": r["t_holm"], "p_nb": r["hull_p_nb"], "nb_holm": r["nb_holm"],
                                             "p_signflip": r["hull_p_signflip"], "design": design_str(r),
                                             "tier_is_new": picks[r["task"]]["H1"]["tier_is_new"], "f1": r["f1"], "cost_T": r["cost"] / 1e12,
                                             "hull_f1": r["hull_f1"], "rate": r["rate"]}
            hyp.append({k: r[k] for k in ("family", "task", "cheap_src", "cheap_bb", "cheap", "expensive", "rule", "budget",
                                          "condition", "f1", "cost", "rate", "hull_lo", "hull_hi", "lam", "hull_f1",
                                          "hull_diff", "hull_diff_fold_min", "hull_diff_fold_max", "hull_fold_diffs",
                                          "hull_p_t_fold", "t_holm", "hull_p_nb", "nb_holm", "hull_p_signflip", "verdict")})
            md.append(f"| {NAMES[r['task']]} | {design_str(r)} | {'yes' if picks[r['task']]['H1']['tier_is_new'] else 'no'} | "
                      f"{r['f1']:.3f} at {T(r['cost'])} | {r['rate']:.2f} | {hull_str(r)} | {r['hull_f1']:.3f} | "
                      f"{pts(r['hull_diff'])} | {pts(r['hull_diff_fold_min'])} to {pts(r['hull_diff_fold_max'])} | "
                      f"{fmt_p(r['hull_p_t_fold'])} / {fmt_p(r['t_holm'])} | {fmt_p(r['hull_p_nb'])} / {fmt_p(r['nb_holm'])} | "
                      f"{fmt_p(r['hull_p_signflip'])} | {v} |")
        md.append("")
    verdict["option_C"] = "PASS" if all(verdict[f"H1_{t}"]["verdict"] == "PASS" for t in FAM) else "FAIL"
    md += ["Selection (clean inner validation, priced view):", ""]
    md += ["| task | designs | pick gain | runner-up (gain) | best design with a new tier (gain) |", "|---|---|---|---|---|"]
    for t in FAM:
        p = picks[t]["H1"]
        ru = dict(zip(["cheap_src", "cheap_bb", "cheap", "expensive", "rule", "budget"], p["runner_up"]))
        bn = dict(zip(["cheap_src", "cheap_bb", "cheap", "expensive", "rule", "budget"], p["best_new_tier_design"]))
        md.append(f"| {NAMES[t]} | {p['n_designs']} | {p['gain_iv']:.4f} | {design_str(ru)} ({p['runner_up_gain_iv']:.4f}) | "
                  f"{design_str(bn)} ({p['best_new_tier_gain_iv']:.4f}) |")
    md.append("")

    # ── S1 baselines ───────────────────────────────────────────────────────
    md += ["## S1: gates at the pick's escalation count on the field mixture (Holm over 3 within each baseline)", "",
           "| task | pick F1 | random gate (diff, p raw / Holm) | audio gate (diff, p raw / Holm) | best static at no higher cost (diff, p raw / Holm) | oracle at the count | oracle, every fixable clip |",
           "|---|---|---|---|---|---|---|"]
    rows = [row_of(t, "H1", "field") for t in FAM]
    s1 = []
    for base in ("random", "audio", "static"):
        holm_rows(rows, f"{base}_p_t_fold", base)
    for r in rows:
        md.append(f"| {NAMES[r['task']]} | {r['f1']:.3f} | {r['random_f1']:.3f} ({pts(r['random_diff'])}, {fmt_p(r['random_p_t_fold'])} / {fmt_p(r['random_holm'])}) | "
                  f"{r['audio_f1']:.3f} ({pts(r['audio_diff'])}, {fmt_p(r['audio_p_t_fold'])} / {fmt_p(r['audio_holm'])}) | "
                  f"{r['best_static']} {r['best_static_f1']:.3f} at {T(r['best_static_cost'])} ({pts(r['static_diff'])}, {fmt_p(r['static_p_t_fold'])} / {fmt_p(r['static_holm'])}) | "
                  f"{r['oracle_k_f1']:.3f} | {r['oracle_full_f1']:.3f} |")
        s1.append({k: r[k] for k in r if not k.startswith("hull_") or k in ("hull_f1",)})
    pd.DataFrame(s1).to_csv(an / "s1_baselines.csv", index=False)
    md.append("")

    # ── S3 headroom ────────────────────────────────────────────────────────
    md += ["## S3: static headroom on the field mixture", "",
           "| task | best static at any cost | best static at no higher cost | pick | oracle at the count | oracle, every fixable clip |",
           "|---|---|---|---|---|---|"]
    s3 = []
    for r in rows:
        md.append(f"| {NAMES[r['task']]} | {r['best_static_any']} {r['best_static_any_f1']:.3f} at {T(r['best_static_any_cost'])} | "
                  f"{r['best_static']} {r['best_static_f1']:.3f} at {T(r['best_static_cost'])} | {r['f1']:.3f} at {T(r['cost'])} | "
                  f"{r['oracle_k_f1']:.3f} | {r['oracle_full_f1']:.3f} |")
        s3.append({"task": r["task"], **{k: r[k] for k in ("best_static_any", "best_static_any_f1", "best_static_any_cost",
                                                           "best_static", "best_static_f1", "best_static_cost", "f1", "cost",
                                                           "oracle_k_f1", "oracle_full_f1")}})
    pd.DataFrame(s3).to_csv(an / "s3_headroom.csv", index=False)
    md.append("")

    # ── S2 humility ────────────────────────────────────────────────────────
    md += ["## S2: humility of the pick's cheap tier (clean thresholds applied to every condition)", ""]
    hum = []
    for t in FAM:
        r = row_of(t, "H1", "field")
        x, bb, k, b = r["cheap_src"], r["cheap_bb"], str(r["cheap"]), r["budget"]
        E = pd.read_csv(an / t / "escalation.csv")
        E = E[(E.cheap_src == x) & (E.cheap_bb == bb) & (E.cheap.astype(str) == k)]
        rate = E.groupby(["budget", "condition"]).rate.mean()
        err = E.groupby("condition").error.mean().to_dict()
        AU = pd.read_csv(an / t / "auroc_wrong.csv")
        AU = AU[(AU.cheap_src == x) & (AU.cheap_bb == bb) & (AU.cheap.astype(str) == k)]
        au = AU.groupby("condition").auroc_wrong.mean().to_dict()
        AF = pd.read_csv(an / t / "auroc_fixable.csv")
        fx = AF.groupby("condition").auroc_fixable.mean().to_dict()
        conds = [c for c in CONDS if (b, c) in rate.index]
        steps = {}
        for bud in sorted({b, "b05", "b10", "b20", "b30"}):
            up = tot = 0
            for c1, c2 in SNR_STEPS + STEPS_FROM_CLEAN:
                if (bud, c1) in rate.index and (bud, c2) in rate.index:
                    tot += 1
                    up += rate[(bud, c2)] > rate[(bud, c1)]
            steps[bud] = (int(up), int(tot))
        for c in conds:
            hum.append({"task": t, "cheap": f"{x}:{bb}:{k}", "budget": b, "condition": c, "rate": rate[(b, c)],
                        "error": err.get(c, np.nan), "auroc_wrong": au.get(c, np.nan), "auroc_fixable": fx.get(c, np.nan)})
        md.append(f"**{NAMES[t]}, cheap tier {x} {bb} + {HEAD[k]} head, {b}** (rate rises at "
                  + ", ".join(f"{bud}: {u} of {n}" for bud, (u, n) in steps.items()) + " steps down in SNR)")
        md += ["", "| | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
        md.append("| escalation rate | " + " | ".join(f"{rate[(b, c)]:.2f}" for c in conds) + " |")
        md.append("| cheap error | " + " | ".join(f"{err.get(c, np.nan):.2f}" for c in conds) + " |")
        md.append("| AUROC cheap wrong | " + " | ".join(f"{au.get(c, np.nan):.2f}" for c in conds) + " |")
        md.append("| AUROC cheap wrong, tier right | " + " | ".join(f"{fx[c]:.2f}" if c in fx else "-" for c in conds) + " |")
        md.append("")
        verdict[f"S2_{t}"] = {"steps": steps, "auroc_wrong_clean": au.get("clean"), "auroc_wrong_field": au.get("field")}
    pd.DataFrame(hum).to_csv(an / "s2_humility.csv", index=False)

    # ── S4 unpriced ────────────────────────────────────────────────────────
    if a.unpriced is not None and all((a.unpriced / t / "selected.csv").exists() for t in FAM):
        md += ["## S4: the unpriced view (group scales free), selection and H1 test redone (Holm over 3)", "",
               "| task | design | tier new? | F1 at T (unpriced) | escalated | hull pair (share) | hull F1 | diff | fold diffs | t p raw / Holm | NB p | sign-flip p | verdict |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        urows = []
        for t in FAM:
            g = pd.read_csv(a.unpriced / t / "selected.csv")
            r = g[(g.family == "H1") & (g.condition == "field")].iloc[0].to_dict()
            r["task"] = t
            r["tier_is_new"] = json.loads((a.unpriced / t / "picks.json").read_text())["H1"]["tier_is_new"]
            urows.append(r)
        holm_rows(urows, "hull_p_t_fold", "t")
        for r in urows:
            v = "PASS" if r["hull_diff"] > 0 and r["t_holm"] < 0.05 else "FAIL"
            r["verdict"] = v
            verdict[f"S4_{r['task']}"] = {"verdict": v, "diff_pts": 100 * r["hull_diff"], "p_t_fold": r["hull_p_t_fold"],
                                          "holm": r["t_holm"], "design": design_str(r), "tier_is_new": r["tier_is_new"]}
            md.append(f"| {NAMES[r['task']]} | {design_str(r)} | {'yes' if r['tier_is_new'] else 'no'} | {r['f1']:.3f} at {T(r['cost'])} | "
                      f"{r['rate']:.2f} | {hull_str(r)} | {r['hull_f1']:.3f} | {pts(r['hull_diff'])} | "
                      f"{pts(r['hull_diff_fold_min'])} to {pts(r['hull_diff_fold_max'])} | {fmt_p(r['hull_p_t_fold'])} / {fmt_p(r['t_holm'])} | "
                      f"{fmt_p(r['hull_p_nb'])} | {fmt_p(r['hull_p_signflip'])} | {v} |")
        pd.DataFrame(urows).to_csv(an / "s4_unpriced.csv", index=False)
        md.append("")

    # ── S5 new tiers as statics ────────────────────────────────────────────
    md += ["## S5: the 3-bit tiers as statics, own heads against transferred heads and against w4g128 at the same length", "",
           "| task | tier | head | condition | own F1 | transferred F1 | own minus transferred (95% CI), t p | w4g128 F1 at T | own minus w4g128 (95% CI), t p |",
           "|---|---|---|---|---|---|---|---|---|"]
    s5 = []
    for t in FAM:
        g = pd.read_csv(an / t / "s5_new_tiers.csv")
        g["task"] = t
        s5.append(g)
        for (tier, h, c), gg in g.groupby(["tier", "head", "condition"], sort=False):
            tr = gg[gg.against == "transferred"].iloc[0]
            w4 = gg[gg.against == "w4g128"].iloc[0]
            md.append(f"| {NAMES[t]} | {tier} | {HEAD[str(h)]} | {c} | {tr['f1_own']:.3f} | {tr['f1_ref']:.3f} | "
                      f"{pts(tr['diff'])} ({pts(tr['ci95_lo'])} to {pts(tr['ci95_hi'])}), {fmt_p(tr['p_t_fold'])} | "
                      f"{w4['f1_ref']:.3f} at {T(w4['cost_ref'])} | {pts(w4['diff'])} ({pts(w4['ci95_lo'])} to {pts(w4['ci95_hi'])}), {fmt_p(w4['p_t_fold'])} |")
    pd.concat(s5).to_csv(an / "s5_new_tiers.csv", index=False)
    md.append("")

    # ── E1 three tiers ─────────────────────────────────────────────────────
    md += ["## E1: three tiers (CNN -> 3-bit AST with own head -> w8g64 at the same length), exploratory, raw p", "",
           "| task | design | condition | F1 at T | rate to 3-bit | rate to 8-bit | hull pair (share) | hull F1 | diff, t p | H1 pick F1 at T | diff, t p | best two-tier with the same cheap tier F1 at T | diff, t p |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    e1 = []
    for t in FAM:
        g = sel3[t]
        same_rows = sel[t][sel[t].family == "E1_two_tier_same_cheap"]
        for _, r in g.iterrows():
            r = r.to_dict()
            r["task"] = t
            e1.append(r)
            md.append(f"| {NAMES[t]} | {design3_str(r)} | {r['condition']} | {r['f1']:.3f} at {T(r['cost'])} | {r['rate1']:.2f} | {r['rate2']:.2f} | "
                      f"{hull_str(r)} | {r['hull_f1']:.3f} | {pts(r['hull_diff'])}, {fmt_p(r['hull_p_t_fold'])} | "
                      f"{r['h1pick_f1']:.3f} at {T(r['h1pick_cost'])} | {pts(r['h1pick_diff'])}, {fmt_p(r['h1pick_p_t_fold'])} | "
                      f"{r['same_cheap_f1']:.3f} at {T(r['same_cheap_cost'])} | {pts(r['same_cheap_diff'])}, {fmt_p(r['same_cheap_p_t_fold'])} |")
            if r["condition"] == "field":
                verdict[f"E1_{t}"] = {"design": design3_str(r), "f1": r["f1"], "cost_T": r["cost"] / 1e12, "rate1": r["rate1"],
                                      "rate2": r["rate2"], "hull_diff_pts": 100 * r["hull_diff"], "hull_p": r["hull_p_t_fold"],
                                      "vs_h1_pick_pts": 100 * r["h1pick_diff"], "vs_h1_pick_p": r["h1pick_p_t_fold"],
                                      "vs_same_cheap_pts": 100 * r["same_cheap_diff"], "vs_same_cheap_p": r["same_cheap_p_t_fold"],
                                      "same_cheap_design": design_str(same_rows.iloc[0].to_dict()),
                                      "gain_iv": picks[t]["E1"]["gain_iv"], "n_designs": picks[t]["E1"]["n_designs"]}
    pd.DataFrame(e1).to_csv(an / "e1_three_tier.csv", index=False)
    md += ["", "Two-tier design with the E1 pick's cheap tier, selected the same way: "
           + "; ".join(f"{NAMES[t]}: {design_str(sel[t][sel[t].family == 'E1_two_tier_same_cheap'].iloc[0].to_dict())} "
                       f"(gain {picks[t]['E1']['two_tier_same_cheap_gain_iv']:.4f}; E1 pick's gain {picks[t]['E1']['gain_iv']:.4f})"
                       for t in FAM), ""]

    # ── E2 ─────────────────────────────────────────────────────────────────
    e2 = []
    for t in FAM:
        g = pd.read_csv(an / t / "e2.csv")
        g["task"] = t
        e2.append(g)
    E2 = pd.concat(e2)
    E2["holm"] = holm(E2.hull_p_t_fold.tolist())
    E2.to_csv(an / "e2_holm.csv", index=False)
    md += ["## E2: every two-tier design of the family on the field mixture against the hull (Holm over all designs of the three tasks)", ""]
    for t in FAM:
        g = E2[E2.task == t]
        best = g.sort_values("hull_diff", ascending=False).iloc[0]
        gn = g[g.expensive.isin(meta[t]["new_tiers"])]
        bn = gn.sort_values("hull_diff", ascending=False).iloc[0]
        md.append(f"- {NAMES[t]}: {len(g)} designs, {int((g.hull_diff > 0).sum())} with a positive mean, "
                  f"{int((g.hull_p_t_fold < 0.05).sum())} with raw p < 0.05, {int((g.holm < 0.05).sum())} passing Holm over all; "
                  f"largest mean {design_str(best)} {pts(best.hull_diff)} (raw p {fmt_p(best.hull_p_t_fold)}, Holm {fmt_p(best.holm)}); "
                  f"largest with a new tier {design_str(bn)} {pts(bn.hull_diff)} (raw p {fmt_p(bn.hull_p_t_fold)}, Holm {fmt_p(bn.holm)}); "
                  f"new-tier designs: {int((gn.hull_diff > 0).sum())} of {len(gn)} positive, {int((gn.hull_p_t_fold < 0.05).sum())} raw p < 0.05")
        verdict[f"E2_{t}"] = {"n": int(len(g)), "positive": int((g.hull_diff > 0).sum()), "raw_p05": int((g.hull_p_t_fold < 0.05).sum()),
                              "holm_pass": int((g.holm < 0.05).sum()), "best": design_str(best), "best_diff_pts": 100 * best.hull_diff,
                              "best_new": design_str(bn), "best_new_diff_pts": 100 * bn.hull_diff, "best_new_p": bn.hull_p_t_fold}
    md.append("")

    # ── E3 ─────────────────────────────────────────────────────────────────
    md += ["## E3: reproduction of the reference pick (Stage 3c P3 on the ESC-50 tasks, the exact-length rerun's P3 on UrbanSound8K)", ""]
    for t in FAM:
        p = picks[t]["E3"]
        worst = max(v for d in meta[t]["e3_reproduction"].values() for v in d.values())
        md.append(f"- {NAMES[t]}: reselection gives {p['design']} gain {p['gain_iv']:.6f}; reference {p['reference']} gain "
                  f"{p['reference_gain_iv']:.6f}: {'equal' if p['equal'] else 'different'}; largest difference of the test rows {worst:.1e}")
        verdict[f"E3_{t}"] = {"equal": p["equal"], "worst_row_difference": worst}
    md.append("")

    # ── front ──────────────────────────────────────────────────────────────
    fr = []
    for t in FAM:
        S = pd.read_csv(an / t / "static.csv")
        g = S[S.condition.isin(["clean", "field"])].groupby(["system", "condition"]).agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
        g["task"] = t
        fr.append(g)
    FR = pd.concat(fr)
    FR.to_csv(an / "e1_front.csv", index=False)
    md += ["## The static front, FP32 head, field-mixture F1 (clean in brackets), priced T", "",
           "| static | " + " | ".join(NAMES[t] for t in FAM) + " |", "|---|---|---|---|"]
    show = ["cnn:mn20:w8a8:esc", "cnn:mn10:w8a8:esc", "cnn:mn10unpr:w8a8:esc", "cnn:mn20:fp32:esc", "ast:n4@256+esc", "ast:n4@512+esc",
            "ast:n4@400+esc", "ast:w4g128@512+esc", "ast:w4g128@400+esc", "ast:w3g64@512+esc", "ast:w3g64@400+esc",
            "ast:w3g64o@512+esc", "ast:w3g64o@400+esc", "ast:n3g64@512+esc", "ast:n3g64@400+esc", "ast:n3g64o@512+esc",
            "ast:n3g64o@400+esc", "ast:w3g32@512+esc", "ast:w3g32@400+esc", "ast:w3g32o@512+esc", "ast:w3g32o@400+esc",
            "ast:w8g64@512+esc", "ast:w8g64@400+esc", "ast:w4g128+esc", "ast:fp32@512+esc", "ast:fp32@400+esc"]
    for sname in show:
        cells = []
        for t in FAM:
            g = FR[(FR.task == t) & (FR.system == sname)]
            if len(g) == 0:
                cells.append("-")
                continue
            f = g[g.condition == "field"].iloc[0]
            c = g[g.condition == "clean"].iloc[0]
            cells.append(f"{f.f1:.3f} ({c.f1:.3f}) at {T(f.cost)}")
        if any(x != "-" for x in cells):
            md.append(f"| {sname} | " + " | ".join(cells) + " |")
    md.append("")

    pd.DataFrame(hyp).to_csv(an / "hypotheses.csv", index=False)
    (an / "verdict.json").write_text(json.dumps(verdict, indent=1, default=float))
    (an / "summary.md").write_text("\n".join(md))
    print(an / "summary.md")


if __name__ == "__main__":
    main()
