"""Idea 6 Stage 3c follow-up summary, UrbanSound8K only: the exact-length mn10 with heads retrained on exact-length
embeddings, replacing the padded mn10 in the hull (analyze_3c.py --unp-retrained). One task, so every p is raw.
Writes analysis/summary.md and analysis/verdict.json from analysis/urban/.

    python scripts/redesign/idea6/stage3c/summary_unp.py --r R
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HEAD = {"esc": "FP32", "8": "8-bit", "4": "4-bit", "bitnet": "BitNet"}


def design_str(r):
    src = "EfficientNet" if r["cheap_src"] == "eff" else r["cheap_src"]
    return f"{src} {str(r['cheap_bb']).upper()} + {HEAD[str(r['cheap'])]} head -> {r['expensive']}, {r['rule']}, {r['budget']}"


def T(x):
    return f"{x / 1e12:.3f}"


def fmt_p(p):
    if pd.isna(p):
        return "-"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", required=True, type=Path)
    a = ap.parse_args(argv)
    an = a.r / "analysis" / "urban"
    sel = pd.read_csv(an / "selected.csv")
    picks = json.loads((an / "picks.json").read_text())
    pad = pd.read_csv(an / "padding_retrained.csv")
    static = pd.read_csv(an / "static.csv")
    e2 = pd.read_csv(an / "e2.csv")
    md = ["# Stage 3c follow-up: UrbanSound8K with the exact-length mn10 and retrained heads (one task, raw p)", ""]
    verdict = {}

    def hull_rows(title, g, note=None):
        md.extend([f"## {title}", ""])
        if note:
            md.extend([note, ""])
        md.extend(["| family | design | hull set | condition | F1 | bit-ops (T) | escalated | hull pair (share of upper) | "
                   "hull F1 | diff (pts) | fold diffs (pts) | fold t p | Nadeau-Bengio p | sign-flip p (floor) | verdict |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
        for _, r in g.iterrows():
            fd = json.loads(r["hull_fold_diffs"])
            passes = bool(r["hull_diff"] > 0 and r["hull_p_t_fold"] < 0.05)
            md.append(f"| {r.family} ({r.base_design}) | {design_str(r)} | {r.hull_set} | {r.condition} | {r.f1:.3f} | "
                      f"{T(r.cost)} | {r.rate:.2f} | {r.hull_lo} / {r.hull_hi} ({r.lam:.2f}) | {r.hull_f1:.3f} | "
                      f"{100 * r.hull_diff:+.2f} | {100 * min(fd):+.1f} to {100 * max(fd):+.1f} | {fmt_p(r.hull_p_t_fold)} | "
                      f"{fmt_p(r.hull_p_nb)} | {r.hull_p_signflip:.3g} ({r.hull_signflip_floor:.3g}) | "
                      f"{'PASS' if passes else 'FAIL'} (raw p) |")
            verdict[f"{r.family}_{r.base_design}_{r.hull_set}_{r.condition}"] = {
                k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in
                ("cheap_src", "cheap_bb", "cheap", "expensive", "rule", "budget", "f1", "cost", "rate", "hull_lo",
                 "hull_hi", "lam", "hull_f1", "hull_diff", "hull_p_t_fold", "hull_p_nb", "hull_p_signflip")}
            verdict[f"{r.family}_{r.base_design}_{r.hull_set}_{r.condition}"]["passes_raw"] = passes
        md.append("")

    hull_rows("P5c: Stage 3b's P2 design and Stage 3c's P3 design with the retrained exact-length mn10 tier",
              sel[sel.family == "P5c"].sort_values(["base_design", "hull_set", "condition"]),
              "Hull sets: stage3b = Stage 2 and Stage 3b statics with the exact-length mn10 in place of the padded one; "
              "new = that plus mn05, mn20, dymn04. Thresholds set on clean inner validation of the retrained heads.")
    p3 = picks["P3"]
    hull_rows("P3 (this run): clean-inner-validation pick over every design with the exact-length tier",
              sel[sel.family == "P3"].sort_values(["condition"]),
              f"Pick: {p3['design']} (inner-validation gain {p3['gain_iv']:.4f}; runner-up {p3['runner_up']} at "
              f"{p3['runner_up_gain_iv']:.4f}; {p3['n_designs']} designs). "
              f"P2 reselected with the Stage 3b family: {picks['P2']['reselected']} "
              f"(gain {picks['P2']['gain_iv_reselected']:.4f}; Stage 3b's pick {picks['P2']['design']}).")

    md.extend(["## Baselines at the same escalation count (field mixture and clean)", "",
               "| family | design | hull set | condition | cascade F1 | random gate F1 (diff, p) | noise-detector gate F1 (diff, p) | "
               "best static at no higher cost (F1, diff, p) | oracle at same count |",
               "|---|---|---|---|---|---|---|---|---|"])
    for _, r in sel.sort_values(["family", "base_design", "hull_set", "condition"]).iterrows():
        aud = (f"{r.audio_f1:.3f} ({100 * r.audio_diff:+.2f}, {fmt_p(r.audio_p_t_fold)})"
               if "audio_f1" in r and not pd.isna(r.get("audio_f1")) else "-")
        md.append(f"| {r.family} ({r.base_design}) | {design_str(r)} | {r.hull_set} | {r.condition} | {r.f1:.3f} | "
                  f"{r.random_f1:.3f} ({100 * r.random_diff:+.2f}, {fmt_p(r.random_p_t_fold)}) | {aud} | "
                  f"{r.best_static} ({r.best_static_f1:.3f}, {100 * r.static_diff:+.2f}, {fmt_p(r.static_p_t_fold)}) | "
                  f"{r.oracle_k_f1:.3f} |")
    md.append("")

    md.extend(["## Retrained exact-length heads against the padded model with its padded heads (every head kind)", "",
               "| backbone | head | condition | exact length, retrained heads | padded | diff (pts), 95% CI | fold t p | NB p | cost T exact / padded |",
               "|---|---|---|---|---|---|---|---|---|"])
    for _, r in pad[pad.condition.isin(["clean", "field"])].iterrows():
        md.append(f"| {r.cheap_bb.upper()} | {HEAD[str(r.cheap)]} | {r.condition} | {r.f1_unpadded_retrained:.3f} | {r.f1_padded:.3f} | "
                  f"{100 * r['diff']:+.2f} ({100 * r.ci95_lo:+.2f} to {100 * r.ci95_hi:+.2f}) | {fmt_p(r.p_t_fold)} | "
                  f"{fmt_p(r.p_nb)} | {T(r.cost_unpadded)} / {T(r.cost_padded)} |")
    w = pad[(pad.cheap_bb == "w8a8") & (pad.cheap == "esc")]
    md.extend(["", f"W8A8 + FP32 head over the {len(w)} conditions: retrained minus padded from {100 * w['diff'].min():+.2f} "
               f"to {100 * w['diff'].max():+.2f} points.", ""])

    md.extend(["## The hull's statics (set new; FP32 head; field F1 with clean in brackets; T)", "",
               "| static | set | T | field (clean) |", "|---|---|---|---|"])
    st = static[static["set"].isin(["stage3b", "new"]) & static.system.str.endswith(":esc")]
    agg = st.groupby(["system", "set", "condition"]).agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
    fld = agg[agg.condition == "field"].set_index("system")
    cln = agg[agg.condition == "clean"].set_index("system")
    for s_, r in fld.sort_values("cost").iterrows():
        md.append(f"| {s_} | {r['set']} | {T(r.cost)} | {r.f1:.3f} ({cln.loc[s_].f1:.3f}) |")
    md.append("")

    pos = e2[e2.hull_diff > 0]
    best = e2.sort_values("hull_diff", ascending=False).iloc[0]
    md.extend(["## E2: every design against the new hull on the field mixture", "",
               f"{len(e2)} designs, {len(pos)} with a positive mean, {int((pos.hull_p_t_fold < 0.05).sum())} with raw fold t p "
               f"below 0.05 (no correction; the pick above is the one pre-specified test). Largest mean: {design_str(best)} "
               f"({100 * best.hull_diff:+.2f}, raw p {fmt_p(best.hull_p_t_fold)}).", ""])
    (a.r / "analysis" / "summary.md").write_text("\n".join(md))
    (a.r / "analysis" / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
