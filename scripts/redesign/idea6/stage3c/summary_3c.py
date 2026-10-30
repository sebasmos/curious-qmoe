"""Idea 6 Stage 3c summary over the four tasks: the Holm families of the plan (P3; P4a clean and field; P4b;
S1, S3, S4, S5), P5a and P5b on UrbanSound8K, humility (S2), the front (E1), E2 over every P3 design, E3 and the
reproduction check (E4). Writes analysis/summary.md, analysis/verdict.json and analysis/*.csv.

    python scripts/redesign/idea6/stage3c/summary_3c.py --r R --s3b S3B
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
ALL = FAM + ["esc5off"]
NAMES = {"esc50": "ESC-50, 50 classes", "esc5": "ESC-50, five fixed groups", "urban": "UrbanSound8K",
         "esc5off": "ESC-50, official categories (secondary)"}
HEAD = {"esc": "FP32", "8": "8-bit", "4": "4-bit", "bitnet": "BitNet"}
SNR_STEPS = [(f"{k}__{a}db", f"{k}__{b}db") for k in ("bg", "pink", "white") for a, b in ((20, 10), (10, 5), (5, 0))]
STEPS_FROM_CLEAN = [("clean", f"{k}__20db") for k in ("bg", "pink", "white")]
CONDS = ["clean"] + [f"{kk}__{s}db" for kk in ("bg", "pink", "white") for s in (20, 10, 5, 0)] + ["field"]


def design_str(r):
    src = "EfficientNet" if r["cheap_src"] == "eff" else r["cheap_src"]
    return f"{src} {str(r['cheap_bb']).upper()} + {HEAD[str(r['cheap'])]} head -> {r['expensive']}, {r['rule']}, {r['budget']}"


def T(x):
    return f"{x / 1e12:.3f}"


def fmt_p(p):
    if pd.isna(p):
        return "-"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def holm_table(rows, pcol, prefix):
    fam = [r for r in rows if r["task"] in FAM and not pd.isna(r.get(pcol, np.nan))]
    adj = holm([r[pcol] for r in fam]) if fam else []
    for r, h in zip(fam, adj):
        r[f"{prefix}_holm"] = float(h)
    for r in rows:
        r.setdefault(f"{prefix}_holm", np.nan)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", required=True, type=Path)
    ap.add_argument("--s3b", required=True, type=Path)
    a = ap.parse_args(argv)
    an = a.r / "analysis"
    sel = {t: pd.read_csv(an / t / "selected.csv") for t in ALL}
    picks = {t: json.loads((an / t / "picks.json").read_text()) for t in ALL}
    meta = {t: json.loads((an / t / "meta.json").read_text()) for t in ALL}
    s3b_verdict = json.loads((a.s3b / "analysis" / "verdict.json").read_text())
    md = ["# Idea 6 Stage 3c summary", ""]
    verdict = {}

    def rows_of(fam, base, cond, src_suffix=None):
        rows = []
        for t in ALL:
            g = sel[t]
            g = g[(g.family == fam) & (g.base_design == base) & (g.condition == cond)]
            if src_suffix is not None:
                g = g[g.cheap_src.str.endswith(src_suffix)]
            if len(g) == 0:
                continue
            r = g.iloc[0].to_dict()
            r["task"] = t
            rows.append(r)
        return rows

    def hull_block(title, rows, verdict_key=None, hold_ref=None, note=None):
        """A hull-comparison table with Holm over the primary tasks; verdict PASS/FAIL per task."""
        holm_table(rows, "hull_p_t_fold", "hull_t")
        holm_table(rows, "hull_p_nb", "hull_nb")
        for r in rows:
            r["passes"] = bool(r["task"] in FAM and r["hull_diff"] > 0 and r["hull_t_holm"] < 0.05)
        md.extend([f"## {title}", ""])
        if note:
            md.extend([note, ""])
        md.extend(["| task | design | F1 | bit-ops (T) | escalated | hull pair (share of upper) | hull F1 | diff (pts) | "
                   "fold diffs (pts) | fold t p raw / Holm 3 | Nadeau-Bengio p raw / Holm 3 | sign-flip p (floor) | verdict |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
        for r in rows:
            fd = json.loads(r["hull_fold_diffs"])
            v = ("PASS" if r["passes"] else "FAIL") if r["task"] in FAM else "raw p only"
            if hold_ref is not None and r["task"] in FAM:
                v += " (holds)" if r["passes"] == hold_ref[r["task"]] else " (changes)"
            md.append(f"| {NAMES[r['task']]} | {design_str(r)} | {r['f1']:.3f} | {T(r['cost'])} | {r['rate']:.2f} | "
                      f"{r['hull_lo']} / {r['hull_hi']} ({r['lam']:.2f}) | {r['hull_f1']:.3f} | {100 * r['hull_diff']:+.2f} | "
                      f"{100 * min(fd):+.1f} to {100 * max(fd):+.1f} | {fmt_p(r['hull_p_t_fold'])} / "
                      f"{fmt_p(r['hull_t_holm']) if r['task'] in FAM else '-'} | {fmt_p(r['hull_p_nb'])} / "
                      f"{fmt_p(r['hull_nb_holm']) if r['task'] in FAM else '-'} | {r['hull_p_signflip']:.3g} "
                      f"({r['hull_signflip_floor']:.3g}) | {v} |")
        md.append("")
        if verdict_key:
            for r in rows:
                verdict.setdefault(r["task"], {})[verdict_key] = {
                    k: r[k] for k in ("hull_diff", "hull_p_t_fold", "hull_t_holm", "hull_p_nb", "f1", "cost", "hull_f1",
                                      "hull_lo", "hull_hi", "lam", "passes", "cheap_src", "cheap_bb", "cheap",
                                      "expensive", "rule", "budget", "hull_set")}
        return rows

    all_rows = []
    s3b_p2_pass = {t: bool(s3b_verdict[t]["P2"]["passes"]) for t in FAM}
    # P3
    all_rows += hull_block("P3: one design per task selected on clean inner validation from the whole EfficientAT family "
                           "(field mixture, hull with every static)", rows_of("P3", "P3", "field"), "P3")
    all_rows += hull_block("S1: the P3 design on clean test folds (clean hull)", rows_of("P3", "P3", "clean"), "S1")
    # S4
    all_rows += hull_block("S4: Stage 3b's P2 designs, unchanged, against the enlarged hull (field mixture)",
                           rows_of("S4", "P2", "field"), "S4", hold_ref=s3b_p2_pass)
    # E4
    e4 = rows_of("E4", "P1", "field") + rows_of("E4", "P2", "field")
    md.extend(["## E4: P1 and P2 recomputed with Stage 3b's systems only (reproduction)", "",
               "Largest absolute difference to Stage 3b's selected.csv (F1, cost, hull F1, diff, p) per task: "
               + "; ".join(f"{t} {max(v for d in meta[t]['e4_reproduction'].values() for v in d.values()):.1e}" for t in ALL),
               ""])
    # P4b both ways
    p4b = {}
    for way, suf in (("a", "int8a"), ("b", "int8b")):
        rows = rows_of("P4b", "P2", "field", suf)
        if rows:
            p4b[way] = hull_block(f"P4b (way {way}): Stage 3b's P2 design with the real-int8 cheap tier (field mixture, "
                                  f"hull with the real-int8 statics added)", rows, f"P4b_{way}", hold_ref=s3b_p2_pass)
            all_rows += p4b[way]
    for way, suf in (("a", "int8a"), ("b", "int8b")):
        rows = rows_of("S5", "P3", "field", suf)
        if rows:
            all_rows += hull_block(f"S5 (way {way}): the P3 design with the real-int8 cheap tier (field mixture)", rows,
                                   f"S5_{way}")
    # P5b
    p5b = rows_of("P5b", "P2", "field") + rows_of("P5b", "P3", "field")
    if p5b:
        md.extend(["## P5b: UrbanSound8K designs with the exact-length mn10 cheap tier (field mixture, hull with the "
                   "exact-length statics added; one task, raw p)", "",
                   "| base design | design | F1 | bit-ops (T) | escalated | hull pair (share) | hull F1 | diff (pts) | "
                   "fold t p | Nadeau-Bengio p | verdict | padded verdict |", "|---|---|---|---|---|---|---|---|---|---|---|---|"])
        for r in p5b:
            ok = r["hull_diff"] > 0 and r["hull_p_t_fold"] < 0.05
            ref = s3b_p2_pass["urban"] if r["base_design"] == "P2" else bool(verdict["urban"]["P3"]["passes"])
            md.append(f"| {r['base_design']} | {design_str(r)} | {r['f1']:.3f} | {T(r['cost'])} | {r['rate']:.2f} | "
                      f"{r['hull_lo']} / {r['hull_hi']} ({r['lam']:.2f}) | {r['hull_f1']:.3f} | {100 * r['hull_diff']:+.2f} | "
                      f"{fmt_p(r['hull_p_t_fold'])} | {fmt_p(r['hull_p_nb'])} | {'PASS' if ok else 'FAIL'} "
                      f"({'holds' if ok == ref else 'changes'}) | {'PASS' if ref else 'FAIL'} |")
            verdict.setdefault("urban", {})[f"P5b_{r['base_design']}"] = {
                "hull_diff": r["hull_diff"], "hull_p_t_fold": r["hull_p_t_fold"], "passes": bool(ok), "holds": bool(ok == ref),
                "f1": r["f1"], "cost": r["cost"], "hull_f1": r["hull_f1"]}
        md.append("")
    pd.DataFrame(all_rows).to_csv(an / "hypotheses.csv", index=False)

    # ── P4a / S6: non-inferiority of the real-int8 static ──────────────────
    ni = []
    for t in ALL:
        p = an / t / "ni_int8.csv"
        if p.exists():
            g = pd.read_csv(p)
            g["task"] = t
            ni.append(g)
    if ni:
        NI = pd.concat(ni, ignore_index=True)
        NI["cheap"] = NI.cheap.astype(str)
        pick_head = {t: str(picks[t]["P2"]["design"][2]) for t in ALL}
        pick_base = {t: picks[t]["P2"]["design"][0] for t in ALL}
        md.extend(["## P4a: real int8 against fake-quant W8A8, same head, non-inferiority within 1 point (one-sided)", "",
                   "Primary rows: mn10 with the head kind of the task's Stage 3b P2 pick (FP32 head where the pick is not "
                   "mn10), way (a); families of 3 per condition. Way (b) and the pick's own backbone alongside.", "",
                   "| condition | way | task | backbone + head | int8 F1 | W8A8 F1 | FP32 F1 | diff (pts) | fold diffs | "
                   "one-sided p raw / Holm 3 | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"])
        p4a_rows = []
        for cond in ("clean", "field"):
            for way in ("a", "b"):
                rows = []
                for t in ALL:
                    k = pick_head[t] if pick_base[t] == "mn10" else "esc"
                    g = NI[(NI.task == t) & (NI.base == "mn10") & (NI.way == way) & (NI.cheap == k) & (NI.condition == cond)]
                    if len(g):
                        r = g.iloc[0].to_dict()
                        r["primary"] = True
                        rows.append(r)
                holm_table(rows, "ni_p", "ni")
                for r in rows:
                    r["holds"] = bool(r["task"] in FAM and r["ni_holm"] < 0.05)
                    fd = json.loads(r["t_fold_diffs"])
                    md.append(f"| {cond} | {way} | {NAMES[r['task']]} | mn10 + {HEAD[r['cheap']]} | {r['f1_int8']:.3f} | "
                              f"{r['f1_w8a8']:.3f} | {r['f1_fp32']:.3f} | {100 * r['ni_diff']:+.2f} | {100 * min(fd):+.1f} to "
                              f"{100 * max(fd):+.1f} | {fmt_p(r['ni_p'])} / {fmt_p(r['ni_holm']) if r['task'] in FAM else '-'} | "
                              f"{('HOLDS' if r['holds'] else 'FAILS') if r['task'] in FAM else 'raw p only'} |")
                    if way == "a":
                        verdict.setdefault(r["task"], {})[f"P4a_{cond}"] = {
                            "diff": r["ni_diff"], "p": r["ni_p"], "holm": r["ni_holm"], "holds": r["holds"], "head": r["cheap"],
                            "f1_int8": r["f1_int8"], "f1_w8a8": r["f1_w8a8"]}
                    else:
                        verdict.setdefault(r["task"], {})[f"P4a_{cond}_way_b"] = {
                            "diff": r["ni_diff"], "p": r["ni_p"], "holm": r["ni_holm"], "holds": r["holds"], "head": r["cheap"]}
                p4a_rows += rows
                # the pick's own backbone when it is not mn10 (alongside, raw p)
                for t in ALL:
                    if pick_base[t] != "mn10":
                        g = NI[(NI.task == t) & (NI.base == pick_base[t]) & (NI.way == way) & (NI.cheap == pick_head[t])
                               & (NI.condition == cond)]
                        if len(g):
                            r = g.iloc[0]
                            fd = json.loads(r["t_fold_diffs"])
                            md.append(f"| {cond} | {way} | {NAMES[t]} | {pick_base[t]} + {HEAD[r['cheap']]} (the pick's own, raw p) | "
                                      f"{r['f1_int8']:.3f} | {r['f1_w8a8']:.3f} | {r['f1_fp32']:.3f} | {100 * r['ni_diff']:+.2f} | "
                                      f"{100 * min(fd):+.1f} to {100 * max(fd):+.1f} | {fmt_p(r['ni_p'])} / - | - |")
        md.append("")
        pd.DataFrame(p4a_rows).to_csv(an / "p4a_noninferiority.csv", index=False)
        md.extend(["### S6: every head kind, way (a) and (b), int8 minus W8A8 (points, raw one-sided p)", "",
                   "| task | backbone | way | condition | " + " | ".join(f"{HEAD[k]} head" for k in HEAD) + " |",
                   "|---|---|---|---|" + "---|" * len(HEAD)])
        for t in ALL:
            for base in sorted(NI.base.unique()):
                for way in ("a", "b"):
                    for cond in ("clean", "field"):
                        g = NI[(NI.task == t) & (NI.base == base) & (NI.way == way) & (NI.condition == cond)]
                        if len(g) == 0:
                            continue
                        cells = []
                        for k in HEAD:
                            h = g[g.cheap == k]
                            cells.append("-" if len(h) == 0 else f"{100 * h.iloc[0].ni_diff:+.2f} ({fmt_p(h.iloc[0].ni_p)})")
                        md.append(f"| {NAMES[t]} | {base} | {way} | {cond} | " + " | ".join(cells) + " |")
        md.append("")
        NI.to_csv(an / "s6_int8_all_heads.csv", index=False)

    # ── P5a: padding ───────────────────────────────────────────────────────
    pp = an / "urban" / "padding.csv"
    if pp.exists():
        P = pd.read_csv(pp)
        P["cheap"] = P.cheap.astype(str)

        def pverdict(r):
            if abs(r["diff"]) > 0.005 and r["p_t_fold"] < 0.05:
                return "differs"
            if -0.005 < r["ci95_lo"] and r["ci95_hi"] < 0.005:
                return "within"
            return "undecided"

        P["verdict"] = P.apply(pverdict, axis=1)
        P.to_csv(an / "p5a_padding.csv", index=False)
        md.extend(["## P5a: UrbanSound8K mn10 at exact lengths minus padded, same Stage 3b heads (points)", "",
                   "Headline: W8A8 backbone with the FP32 head (the Stage 3b P2 cheap tier). 'differs': |mean| > 0.5 points "
                   "and fold t p < 0.05; 'within': the fold-level 95% CI lies inside (-0.5, +0.5).", "",
                   "| backbone | head | condition | unpadded F1 | padded F1 | diff (pts) | 95% CI | fold t p | cost unpadded / padded (T) | verdict |",
                   "|---|---|---|---|---|---|---|---|---|---|"])
        for _, r in P[P.condition.isin(["clean", "field"])].sort_values(["cheap_bb", "cheap", "condition"]).iterrows():
            md.append(f"| {r.cheap_bb} | {HEAD[r.cheap]} | {r.condition} | {r.f1_unpadded:.3f} | {r.f1_padded:.3f} | "
                      f"{100 * r['diff']:+.2f} | {100 * r.ci95_lo:+.2f} to {100 * r.ci95_hi:+.2f} | {fmt_p(r.p_t_fold)} | "
                      f"{T(r.cost_unpadded)} / {T(r.cost_padded)} | {r.verdict} |")
        md.append("")
        for cond in ("clean", "field"):
            h = P[(P.cheap_bb == "w8a8") & (P.cheap == "esc") & (P.condition == cond)].iloc[0]
            verdict.setdefault("urban", {})[f"P5a_{cond}"] = {"diff": h["diff"], "ci95": [h.ci95_lo, h.ci95_hi],
                                                             "p": h.p_t_fold, "verdict": h.verdict}
        md.extend(["Every condition, W8A8 + FP32 head: " + "; ".join(
            f"{r.condition} {100 * r['diff']:+.2f}" for _, r in P[(P.cheap_bb == 'w8a8') & (P.cheap == 'esc')].iterrows()), ""])

    # ── S3: baselines for the P3 pick and the substituted picks ────────────
    s4 = []
    for fam, base in (("P3", "P3"), ("P4b", "P2"), ("S5", "P3"), ("P5b", "P2"), ("P5b", "P3")):
        for bl in ("random", "audio", "static"):
            rows = []
            for t in ALL:
                g = sel[t]
                g = g[(g.family == fam) & (g.base_design == base) & (g.condition == "field")]
                for _, r in g.iterrows():
                    r = r.to_dict()
                    if f"{bl}_p_t_fold" not in r or pd.isna(r.get(f"{bl}_p_t_fold")):
                        continue
                    rows.append({"task": t, "family": fam, "cheap_src": r["cheap_src"], "baseline": bl, "f1": r["f1"],
                                 "base_f1": r["best_static_f1" if bl == "static" else f"{bl}_f1"],
                                 "base_name": r["best_static"] if bl == "static" else bl,
                                 "diff": r[f"{bl}_diff"], "p": r[f"{bl}_p_t_fold"], "p_nb": r[f"{bl}_p_nb"],
                                 "oracle_k": r["oracle_k_f1"]})
            if fam == "P3":
                holm_table(rows, "p", "b")
            else:
                for r in rows:
                    r["b_holm"] = np.nan
            s4 += rows
    S4 = pd.DataFrame(s4)
    S4.to_csv(an / "s3_baselines.csv", index=False)
    md.extend(["## S3: picks against baselines at the same escalation count (field mixture; Holm 3 for the P3 pick, raw p otherwise)", "",
               "| family | cheap tier | baseline | task | baseline F1 | cascade F1 | diff (pts) | fold t p raw / Holm 3 | oracle at same count |",
               "|---|---|---|---|---|---|---|---|---|"])
    for _, r in S4.iterrows():
        md.append(f"| {r.family} | {r.cheap_src} | {r.baseline} ({r.base_name}) | {NAMES[r.task]} | {r.base_f1:.3f} | {r.f1:.3f} | "
                  f"{100 * r['diff']:+.2f} | {fmt_p(r.p)} / {fmt_p(r.b_holm)} | {r.oracle_k:.3f} |")
    md.append("")

    # ── E1: the front ──────────────────────────────────────────────────────
    fr = []
    for t in ALL:
        S = pd.read_csv(an / t / "static.csv")
        g = S[S.condition.isin(["clean", "field"])].groupby(["system", "condition"]).agg(
            f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
        g["task"] = t
        fr.append(g)
    FR = pd.concat(fr)
    FR.to_csv(an / "e1_front.csv", index=False)
    piv = FR.pivot_table(index=["system"], columns=["task", "condition"], values="f1")
    cost = FR[FR.task == "esc50"].groupby("system").cost.mean()
    cost_u = FR[FR.task == "urban"].groupby("system").cost.mean()
    keep = [s for s in piv.index if (s.startswith("cnn:") and s.endswith(":esc")) or s in ("ast:n4@256+esc", "ast:n4@384+esc",
            f"ast:n4@512+esc", "ast:n4@400+esc", "ast:n4+esc", "ast:w4g128+esc", "eff:w8a8:esc")]
    md.extend(["## E1: static systems with the FP32 head, mean macro F1 (cost in T: ESC-50 / UrbanSound8K)", "",
               "| system | cost | ESC-50/50 clean | field | five groups clean | field | UrbanSound8K clean | field | "
               "official categories clean | field |", "|---|---|---|---|---|---|---|---|---|---|"])
    for s in sorted(keep, key=lambda z: cost.get(z, cost_u.get(z, 0))):
        vals = []
        for t in ALL:
            for c in ("clean", "field"):
                v = piv.loc[s].get((t, c), np.nan)
                vals.append("-" if pd.isna(v) else f"{v:.3f}")
        cs = f"{T(cost[s]) if s in cost else '-'} / {T(cost_u[s]) if s in cost_u else '-'}"
        md.append(f"| {s} | {cs} | " + " | ".join(vals) + " |")
    md.append("")

    # ── S2: humility of every pick's cheap tier ────────────────────────────
    hum = []
    md.extend(["## S2: humility of the picks' cheap tiers (clean thresholds)", ""])
    for t in ALL:
        E = pd.read_csv(an / t / "escalation.csv")
        AU = pd.read_csv(an / t / "auroc_wrong.csv")
        FX = pd.read_csv(an / t / "auroc_fixable.csv")
        seen = set()
        for c_ in meta[t]["comparisons"]:
            if c_["family"] == "E4":
                continue
            x, bb, k, e, r_, b = c_["design"]
            k = str(k)
            if (x, bb, k, b) in seen:
                continue
            seen.add((x, bb, k, b))
            g = E[(E.cheap_src == x) & (E.cheap_bb == bb) & (E.cheap.astype(str) == k)]
            rate = g.groupby(["budget", "condition"]).rate.mean()
            err = g[g.budget == b].groupby("condition").error.mean()
            au = AU[(AU.cheap_src == x) & (AU.cheap_bb == bb) & (AU.cheap.astype(str) == k)].groupby("condition").auroc_wrong.mean()
            fxg = FX[(FX.cheap_src == x) & (FX.cheap_bb == bb) & (FX.cheap.astype(str) == k) & (FX.budget == b)]
            fx = fxg.groupby("condition").auroc_fixable.mean()
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
                hum.append({"task": t, "family": c_["family"], "base": c_["base"], "cheap": f"{x}:{bb}:{k}", "budget": b,
                            "condition": c, "rate": rate[(b, c)], "error": err.get(c, np.nan),
                            "auroc_wrong": au.get(c, np.nan), "auroc_fixable": fx.get(c, np.nan)})
            md.append(f"**{NAMES[t]}, {c_['family']} ({c_['base']} design) cheap tier {x} {bb} + {HEAD[k]} head, {b}** (rate rises at "
                      + ", ".join(f"{bud}: {u} of {n}" for bud, (u, n) in steps.items()) + " steps down in SNR)")
            md.append("")
            md.append("| condition | " + " | ".join(conds) + " |")
            md.append("|---|" + "---|" * len(conds))
            md.append("| escalated | " + " | ".join(f"{rate[(b, c)]:.2f}" for c in conds) + " |")
            md.append("| cheap error | " + " | ".join(f"{err.get(c, np.nan):.2f}" for c in conds) + " |")
            md.append("| AUROC cheap wrong | " + " | ".join(f"{au.get(c, np.nan):.2f}" for c in conds) + " |")
            md.append("| AUROC cheap wrong, tier right | " + " | ".join(
                "-" if pd.isna(fx.get(c, np.nan)) else f"{fx.get(c):.2f}" for c in conds) + " |")
            md.append("")
    pd.DataFrame(hum).to_csv(an / "s2_humility.csv", index=False)

    # ── E2: every P3 design, Holm over all ─────────────────────────────────
    e2 = []
    for t in FAM:
        g = pd.read_csv(an / t / "e2.csv")
        g["task"] = t
        e2.append(g)
    E2 = pd.concat(e2, ignore_index=True)
    E2["holm_all"] = holm(E2.hull_p_t_fold.to_numpy())
    E2.to_csv(an / "e2_holm.csv", index=False)
    md.extend(["## E2: every P3 design on the field mixture against the enlarged hull", "",
               f"{len(E2)} designs over the three tasks, Holm over all.", "",
               "| task | designs | positive mean | pass Holm | best design (diff, Holm p) | cheap tiers among the positive |",
               "|---|---|---|---|---|---|"])
    for t in FAM:
        g = E2[E2.task == t]
        best = g.sort_values("hull_diff", ascending=False).iloc[0]
        pos = g[g.hull_diff > 0]
        md.append(f"| {NAMES[t]} | {len(g)} | {len(pos)} | {int(((g.hull_diff > 0) & (g.holm_all < 0.05)).sum())} | "
                  f"{design_str(best)} ({100 * best.hull_diff:+.2f}, {fmt_p(best.holm_all)}) | "
                  f"{ {k: int(v) for k, v in pos.cheap_src.value_counts().items()} } |")
    md.append("")

    # ── E3: W8A8 minus FP32 for every CNN ──────────────────────────────────
    e3 = []
    for t in ALL:
        S = pd.read_csv(an / t / "static.csv")
        S = S[S.system.str.startswith("cnn:") & ~S.system.str.contains("int8")]
        g = S.groupby(["system", "condition"]).f1.mean().reset_index()
        g[["x", "model", "bb", "head"]] = g.system.str.split(":", expand=True)
        p = g.pivot_table(index=["model", "head", "condition"], columns="bb", values="f1").reset_index()
        p["diff"] = p["w8a8"] - p["fp32"]
        p["task"] = t
        e3.append(p)
    E3 = pd.concat(e3)
    E3.to_csv(an / "e3_cnn_w8a8_minus_fp32.csv", index=False)
    md.extend(["## E3: EfficientAT W8A8 minus FP32, static F1 (points; range over heads)", "",
               "| task | model | clean | field | worst condition |", "|---|---|---|---|---|"])
    for t in ALL:
        for m in sorted(E3[E3.task == t].model.unique()):
            g = E3[(E3.task == t) & (E3.model == m)]
            c = g[g.condition == "clean"]["diff"]
            f = g[g.condition == "field"]["diff"]
            w = g.loc[g["diff"].idxmin()]
            md.append(f"| {NAMES[t]} | {m} | {100 * c.min():+.1f} to {100 * c.max():+.1f} | {100 * f.min():+.1f} to "
                      f"{100 * f.max():+.1f} | {w.condition} {100 * w['diff']:+.1f} ({HEAD[str(w['head'])]} head) |")
    md.append("")

    for t in ALL:
        verdict.setdefault(t, {})["P3_design"] = picks[t]["P3"]["design"]
        verdict[t]["P2_design"] = picks[t]["P2"]["design"]
        verdict[t]["P3_n_designs"] = picks[t]["P3"]["n_designs"]
    (an / "verdict.json").write_text(json.dumps(verdict, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (an / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
