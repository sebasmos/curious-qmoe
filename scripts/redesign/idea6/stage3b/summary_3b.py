"""Idea 6 Stage 3b summary over the four tasks: Holm families of the plan
(P1, P2, S1, S3, S4 over esc50, esc5, urban; esc5off with raw p), the
humility tables (S2), the new front (E1), E2 over every design, E3 (W8A8
minus FP32 for the CNNs) and the Stage 2 reproduction check (E4). Writes
analysis/summary.md, analysis/verdict.json and analysis/*.csv.

    python scripts/redesign/idea6/stage3b/summary_3b.py --r R --s2 S2
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
SNR_STEPS = [(f"{k}__{a}db", f"{k}__{b}db") for k in ("bg", "pink", "white") for a, b in ((20, 10), (10, 5), (5, 0))]
STEPS_FROM_CLEAN = [("clean", f"{k}__20db") for k in ("bg", "pink", "white")]


def design_str(r):
    src = "EfficientNet" if r["cheap_src"] == "eff" else r["cheap_src"]
    head = {"esc": "FP32", "8": "8-bit", "4": "4-bit", "bitnet": "BitNet"}[str(r["cheap"])]
    return f"{src} {r['cheap_bb'].upper()} + {head} head -> {r['expensive']}, {r['rule']}, {r['budget']}"


def T(x):
    return f"{x / 1e12:.3f}"


def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def holm_table(rows, pcol, prefix):
    fam = [r for r in rows if r["task"] in FAM]
    adj = holm([r[pcol] for r in fam]) if fam else []
    for r, h in zip(fam, adj):
        r[f"{prefix}_holm"] = float(h)
    for r in rows:
        r.setdefault(f"{prefix}_holm", np.nan)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", required=True, type=Path)
    ap.add_argument("--s2", required=True, type=Path)
    a = ap.parse_args(argv)
    an = a.r / "analysis"
    sel = {t: pd.read_csv(an / t / "selected.csv") for t in ALL}
    picks = {t: json.loads((an / t / "picks.json").read_text()) for t in ALL}
    meta = {t: json.loads((an / t / "meta.json").read_text()) for t in ALL}
    md = ["# Idea 6 Stage 3b summary", ""]
    verdict = {}

    def get(t, fam, cond, which="new"):
        g = sel[t]
        return g[(g.family == fam) & (g.condition == cond) & (g.hull_set == which)].iloc[0].to_dict()

    # ── P1, P2 (field), S1 (clean), E4 ─────────────────────────────────────
    out_rows = []
    for fam in ("P1", "P2"):
        for cond in ("field", "clean"):
            for which in ("new", "stage2"):
                if which == "stage2" and fam == "P2":
                    continue
                rows = []
                for t in ALL:
                    r = get(t, fam, cond, which)
                    r["task"] = t
                    rows.append(r)
                holm_table(rows, "hull_p_t_fold", "hull_t")
                holm_table(rows, "hull_p_nb", "hull_nb")
                for r in rows:
                    r["passes"] = bool(r["task"] in FAM and r["hull_diff"] > 0 and r["hull_t_holm"] < 0.05)
                    r["family"], r["condition"], r["hull_set"] = fam, cond, which
                out_rows += rows
    H = pd.DataFrame(out_rows)
    # Holm over all six primary tests (P1 and P2 on the field mixture, new hull), stricter check
    prim = H[(H.condition == "field") & (H.hull_set == "new") & (H.task.isin(FAM))]
    adj6 = holm(prim.hull_p_t_fold.to_numpy())
    H.loc[prim.index, "hull_t_holm6"] = adj6
    H.to_csv(an / "hypotheses.csv", index=False)

    def table(fam, cond, which, title):
        g = H[(H.family == fam) & (H.condition == cond) & (H.hull_set == which)]
        md.extend([f"## {title}", "",
                   "| task | design | F1 | bit-ops (T) | escalated | hull pair (share of upper) | hull F1 | diff (pts) | "
                   "fold diffs (pts) | fold t p raw / Holm 3 | Nadeau-Bengio p raw / Holm 3 | sign-flip p (floor) | verdict |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
        for _, r in g.iterrows():
            fd = json.loads(r.hull_fold_diffs)
            v = ("PASS" if r.passes else "FAIL") if r.task in FAM else "raw p only"
            md.append(f"| {NAMES[r.task]} | {design_str(r)} | {r.f1:.3f} | {T(r.cost)} | {r.rate:.2f} | "
                      f"{r.hull_lo} / {r.hull_hi} ({r.lam:.2f}) | {r.hull_f1:.3f} | {100 * r.hull_diff:+.2f} | "
                      f"{100 * min(fd):+.1f} to {100 * max(fd):+.1f} | {fmt_p(r.hull_p_t_fold)} / "
                      f"{fmt_p(r.hull_t_holm) if r.task in FAM else '-'} | {fmt_p(r.hull_p_nb)} / "
                      f"{fmt_p(r.hull_nb_holm) if r.task in FAM else '-'} | {r.hull_p_signflip:.3g} "
                      f"({r.hull_signflip_floor:.3g}) | {v} |")
        md.append("")

    table("P1", "field", "new", "P1: Stage 2's H1 designs against the hull with every new static (field mixture)")
    table("P2", "field", "new", "P2: one design per task selected on clean inner validation (field mixture)")
    md.append("Holm over all six primary tests (P1 and P2, field mixture): " + ", ".join(
        f"{r.family} {r.task} {fmt_p(r.hull_t_holm6)}" for _, r in H[H.hull_t_holm6.notna()].iterrows()))
    md.append("")
    table("P1", "clean", "new", "S1: P1 designs, clean test folds")
    table("P2", "clean", "new", "S1: P2 designs, clean test folds")
    table("P1", "field", "stage2", "E4: P1 against Stage 2's hull (reproduction check)")
    rep = []
    for t in ALL:
        s2sel = pd.read_csv(a.s2 / "analysis" / t / "selected.csv")
        s2r = s2sel[(s2sel.family == "H1") & (s2sel.condition == "field")].iloc[0]
        mine = H[(H.family == "P1") & (H.condition == "field") & (H.hull_set == "stage2") & (H.task == t)].iloc[0]
        rep.append({"task": t, "stage2_diff": float(s2r.hull_diff), "recomputed_diff": float(mine.hull_diff),
                    "stage2_p": float(s2r.hull_p_t_fold), "recomputed_p": float(mine.hull_p_t_fold)})
    md.append("Stage 2's own H1 rows against this recomputation: " + "; ".join(
        f"{r['task']} {100 * r['stage2_diff']:+.3f} / {100 * r['recomputed_diff']:+.3f} pts" for r in rep))
    md.append("")

    # ── S3 non-inferiority, S4 baselines ───────────────────────────────────
    s3 = []
    for fam in ("P1", "P2"):
        for cond in ("field", "clean"):
            rows = []
            for t in ALL:
                r = get(t, fam, cond)
                rows.append({"task": t, "family": fam, "condition": cond, "f1": r["f1"], "cost": r["cost"],
                             "ref": r["ni_ref"], "ref_f1": r["ni_ref_f1"], "ref_cost": r["ni_ref_cost"],
                             "diff": r["ni_diff"], "p": r["ni_p"], "p_nb": r["ni_p_nb"]})
            holm_table(rows, "p", "ni")
            for r in rows:
                r["matches_at_lower_cost"] = bool(r["task"] in FAM and r["ni_holm"] < 0.05 and r["cost"] < r["ref_cost"])
            s3 += rows
    S3 = pd.DataFrame(s3)
    S3.to_csv(an / "s3_noninferiority.csv", index=False)
    md.extend(["## S3: non-inferiority to the static short AST (n4 at 512 or 400 frames, FP32 head), margin 1 point", "",
               "| design | condition | task | cascade F1 at T | static F1 at T | diff (pts) | one-sided p raw / Holm 3 | "
               "Nadeau-Bengio p | matches at lower cost |", "|---|---|---|---|---|---|---|---|---|"])
    for _, r in S3.iterrows():
        md.append(f"| {r.family} | {r.condition} | {NAMES[r.task]} | {r.f1:.3f} at {T(r.cost)} | {r.ref_f1:.3f} at "
                  f"{T(r.ref_cost)} | {100 * r['diff']:+.2f} | {fmt_p(r.p)} / "
                  f"{fmt_p(r.ni_holm) if r.task in FAM else '-'} | {fmt_p(r.p_nb)} | "
                  f"{'yes' if r.matches_at_lower_cost else ('no' if r.task in FAM else '-')} |")
    md.append("")
    s4 = []
    for base in ("random", "audio", "static"):
        rows = []
        for t in ALL:
            r = get(t, "P2", "field")
            if f"{base}_p_t_fold" not in r or pd.isna(r.get(f"{base}_p_t_fold")):
                continue
            rows.append({"task": t, "baseline": base, "f1": r["f1"],
                         "base_f1": r["best_static_f1" if base == "static" else f"{base}_f1"],
                         "base_name": r["best_static"] if base == "static" else base,
                         "diff": r[f"{base}_diff"], "p": r[f"{base}_p_t_fold"], "p_nb": r[f"{base}_p_nb"],
                         "oracle_k": r["oracle_k_f1"]})
        holm_table(rows, "p", "b")
        s4 += rows
    S4 = pd.DataFrame(s4)
    S4.to_csv(an / "s4_baselines.csv", index=False)
    md.extend(["## S4: P2 design against baselines at the same escalation count (field mixture)", "",
               "| baseline | task | baseline F1 | cascade F1 | diff (pts) | fold t p raw / Holm 3 | oracle at same count |",
               "|---|---|---|---|---|---|---|"])
    for _, r in S4.iterrows():
        md.append(f"| {r.baseline} ({r.base_name}) | {NAMES[r.task]} | {r.base_f1:.3f} | {r.f1:.3f} | "
                  f"{100 * r['diff']:+.2f} | {fmt_p(r.p)} / {fmt_p(r.b_holm) if r.task in FAM else '-'} | "
                  f"{r.oracle_k:.3f} |")
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
    keep = [s for s in piv.index if s.startswith("cnn:") or "@256" in s or "@384" in s or "n4@" in s
            or s.startswith("eff:w8a8") or s in ("ast:n4+esc", "ast:w4g128+esc")]
    md.extend(["## E1: static systems, mean macro F1 (cost in T: ESC-50 / UrbanSound8K)", "",
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

    # ── S2: humility ───────────────────────────────────────────────────────
    hum = []
    md.extend(["## S2: humility of the cheap tiers (clean thresholds)", ""])
    for t in ALL:
        E = pd.read_csv(an / t / "escalation.csv")
        AU = pd.read_csv(an / t / "auroc_wrong.csv")
        FX = pd.read_csv(an / t / "auroc_fixable.csv")
        for fam in ("P1", "P2"):
            x, bb, k, e, r_, b = picks[t][fam]["design"]
            k = str(k)
            g = E[(E.cheap_src == x) & (E.cheap_bb == bb) & (E.cheap.astype(str) == k)]
            rate = g.groupby(["budget", "condition"]).rate.mean()
            err = g[g.budget == b].groupby("condition").error.mean()
            au = AU[(AU.cheap_src == x) & (AU.cheap_bb == bb) & (AU.cheap.astype(str) == k)].groupby(
                "condition").auroc_wrong.mean()
            fx = FX[FX.family == fam].groupby("condition").auroc_fixable.mean()
            conds = [c for c in ["clean"] + [f"{kk}__{s}db" for kk in ("bg", "pink", "white") for s in (20, 10, 5, 0)]
                     + ["field"] if (b, c) in rate.index]
            steps = {}
            for bud in sorted({b, "b10", "b20", "b30"}):
                up = tot = 0
                for c1, c2 in SNR_STEPS + STEPS_FROM_CLEAN:
                    if (bud, c1) in rate.index and (bud, c2) in rate.index:
                        tot += 1
                        up += rate[(bud, c2)] > rate[(bud, c1)]
                steps[bud] = (int(up), int(tot))
            for c in conds:
                hum.append({"task": t, "family": fam, "cheap": f"{x}:{bb}:{k}", "budget": b, "condition": c,
                            "rate": rate[(b, c)], "error": err.get(c, np.nan), "auroc_wrong": au.get(c, np.nan),
                            "auroc_fixable": fx.get(c, np.nan)})
            md.append(f"**{NAMES[t]}, {fam} cheap system {x} {bb} + {k} head, {b}** (rate rises at "
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

    # ── E2: every design, Holm over all ────────────────────────────────────
    e2 = []
    for t in FAM:
        g = pd.read_csv(an / t / "e2.csv")
        g["task"] = t
        e2.append(g)
    E2 = pd.concat(e2, ignore_index=True)
    E2["holm_all"] = holm(E2.hull_p_t_fold.to_numpy())
    E2.to_csv(an / "e2_holm.csv", index=False)
    md.extend(["## E2: every P2 design on the field mixture against the new hull", "",
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

    # ── E3: W8A8 minus FP32 for the CNNs ───────────────────────────────────
    e3 = []
    for t in ALL:
        S = pd.read_csv(an / t / "static.csv")
        S = S[S.system.str.startswith("cnn:")]
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
        for m in ("mn04", "mn10"):
            g = E3[(E3.task == t) & (E3.model == m)]
            c = g[g.condition == "clean"]["diff"]
            f = g[g.condition == "field"]["diff"]
            w = g.loc[g["diff"].idxmin()]
            md.append(f"| {NAMES[t]} | {m} | {100 * c.min():+.1f} to {100 * c.max():+.1f} | {100 * f.min():+.1f} to "
                      f"{100 * f.max():+.1f} | {w.condition} {100 * w['diff']:+.1f} ({w['head']} head) |")
    md.append("")

    for t in ALL:
        verdict[t] = {
            "P1": {k: H[(H.family == "P1") & (H.condition == "field") & (H.hull_set == "new") & (H.task == t)]
                   .iloc[0][k] for k in ("hull_diff", "hull_p_t_fold", "hull_t_holm", "hull_p_nb", "f1", "cost",
                                         "hull_f1", "hull_lo", "hull_hi", "lam", "passes")},
            "P2": {k: H[(H.family == "P2") & (H.condition == "field") & (H.hull_set == "new") & (H.task == t)]
                   .iloc[0][k] for k in ("hull_diff", "hull_p_t_fold", "hull_t_holm", "hull_p_nb", "f1", "cost",
                                         "hull_f1", "hull_lo", "hull_hi", "lam", "passes")},
            "P2_design": picks[t]["P2"]["design"], "P1_design": picks[t]["P1"]["design"],
            "mean2_included": meta[t]["mean2"]}
    (an / "verdict.json").write_text(json.dumps(verdict, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (an / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
