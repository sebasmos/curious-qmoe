"""T14 report builder: stability_all_ideas.csv / .md, compact table, from tables/stability_rows.csv, stability_singles.csv, not_computed.csv and the latency run.
python t14_report.py --r RESULTS_DIR"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--r", required=True, type=Path); R = ap.parse_args().r; T = R / "tables"
D = pd.read_csv(T / "stability_rows.csv"); Sg = pd.read_csv(T / "stability_singles.csv"); NR = pd.read_csv(T / "not_computed.csv"); L = pd.read_csv(R / "latency/latency_bootstrap.csv")
def lat(id_, task):
    key = {("I6d", "esc50"): ("esc50", "ort_int8 CNN + n4 AST kernel, esc50 P3 pick"), ("I6c", "urban"): ("urban", "ort_int8 CNN + int8-dynamic AST kernel, urban stage 3b P2 pick (not the P3 pick)")}.get((id_, task))
    if not key: return np.nan, ""
    g = L[(L.task == key[0]) & (L.threads == 2) & (L.condition == "field") & (L.rate_which == "rate")]
    if g.empty: return np.nan, ""
    g = g.iloc[0]; return float(g.pair_ratio), f"p95/median {g.pair_ratio:.1f} (interval {g.pair_ratio_lo:.1f} to {g.pair_ratio_hi:.1f}) against {g.ast_ratio:.2f} for the AST alone; 2 threads, {key[1]}"
rows = []
for r in D.itertuples(index=False):
    r = r._asdict(); lr, ltxt = lat(r["id"], r["task"]); r["p95_over_median_pair"] = lr; r["latency_note"] = ltxt; rows.append(r)
A = pd.DataFrame(rows)
for c, lab in (("clean", "clean"), ("field", "field")):
    A[f"{c}_ratio_ci"] = [f"{a:.2f} ({b:.2f} to {d:.2f})" if np.isfinite(a) else "" for a, b, d in zip(A.get(f"{c}_ratio", np.nan), A.get(f"{c}_ratio_lo", np.nan), A.get(f"{c}_ratio_hi", np.nan))]
A["row_kind"] = "system"
S = Sg.copy(); S["id"] = "Sg"; S["system"] = "single: " + S.system; S["row_kind"] = "single"; S["overall_verdict"] = [("more stable" if (str(a).startswith("more")) else "less stable" if str(a).startswith("less") else "as stable" if str(a).startswith("as") else "not determinable at five folds") for a in S.get("clean_label", pd.Series(["x"] * len(S)))]
S = S.rename(columns={"worst": "shift_worst_sys", "drop": "shift_drop_sys"})
N = NR.copy(); N["system"] = N["id"]; N["row_kind"] = "not computed"; N["overall_verdict"] = [("not run" if i in ("T4", "IBL") else "not determinable (data)" if i in ("I2", "T9", "T12") else "not computed") for i in N["id"]]; N["protocol"] = ""
out = pd.concat([A, S, N], ignore_index=True)
out.to_csv(R / "stability_all_ideas.csv", index=False)
def f2(x): return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"
def table(df, cols):
    h = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return h + "".join("| " + " | ".join(str(c[1](r)) for c in cols) + " |\n" for r in df.itertuples(index=False))
def get(r, k): v = getattr(r, k, np.nan); return v
comp = A.copy()
cols = [("System", lambda r: r.system), ("Protocol", lambda r: r.protocol), ("Task", lambda r: r.task), ("Seeds x folds", lambda r: f"{int(r.n_seed)}x{int(r.n_fold)}"),
        ("SD folds clean / field", lambda r: f"{f2(get(r,'clean_sd_fold'))} / {f2(get(r,'field_sd_fold'))}"),
        ("Variance ratio vs best single, clean (95% CI)", lambda r: r.clean_ratio_ci if isinstance(r.clean_ratio_ci, str) else ""), ("field", lambda r: r.field_ratio_ci if isinstance(r.field_ratio_ci, str) else ""),
        ("Shift worst case: system vs reference (diff)", lambda r: (f"{f2(get(r,'shift_worst_sys'))} vs {f2(get(r,'shift_worst_ref'))} ({get(r,'shift_dworst'):+.2f})" if np.isfinite(get(r, 'shift_dworst')) else "")),
        ("Verdict", lambda r: r.overall_verdict)]
comp = comp.rename(columns={"worst_sys": "shift_worst_sys", "worst_ref": "shift_worst_ref"}); A = A.rename(columns={"worst_sys": "shift_worst_sys"})
comp = A
lines = ["# T14 stability of every idea and system (stored per-seed, per-fold values)\n",
         "Protocol 2 = random inner split, reshuffled folds, leaky; protocol 3 = inner split grouped by recording. A ratio is formed only inside one protocol and one source; spreads of different protocols are never compared. "
         "Ratio = variance of fold-level F1 of the system over that of the reference (best single model at equal cost, chosen on inner validation of the same condition family where stored, otherwise `ref_selection` in the csv). "
         "Below 0.7 with an interval below 1 would be more stable; with 5 or 10 folds almost every interval spans 1.\n", table(comp, cols),
         "\n## Singles (stability_singles.csv, 284 rows; selected)\n", table(S[S.system.str.contains("cnn:mn20:w8a8:4|cnn:mn10:w8a8:esc|cnn:mn20:fp32:4|ast:fp32|ast:w4g128|cnn:mn20:fp32:8|ast:w8")].head(40), [("System", lambda r: r.system), ("Task", lambda r: r.task), ("Cost", lambda r: f"{r.cost:.2e}"), ("SD folds clean / field", lambda r: f"{f2(getattr(r,'clean_sd_fold'))} / {f2(getattr(r,'field_sd_fold'))}"), ("Ratio clean", lambda r: f2(getattr(r, 'clean_ratio', np.nan))), ("Worst case", lambda r: f2(getattr(r, 'shift_worst_sys', np.nan)))]),
         "\n## Not run, not computed, not determinable\n", table(N, [("Id", lambda r: r.id), ("Status", lambda r: r.overall_verdict), ("Reason", lambda r: r.reason)])]
(R / "stability_all_ideas.md").write_text("\n".join(lines))
(R / "tables/compact_table.md").write_text(table(comp, cols))
print(len(out), "rows written")

# ---- by-system compact table (ranges over tasks), pasted at the top of the README ----------------------------------------------------------------------------------
def rng(s):
    s = s.dropna()
    return "" if s.empty else (f"{s.min():.2f}" if len(s) == 1 or abs(s.max() - s.min()) < 0.005 else f"{s.min():.2f} to {s.max():.2f}")
def sh(s):
    s = s.dropna(); return "not scored" if s.empty else (f"{s.min():+.1f}" if len(s) == 1 else f"{s.min():+.1f} to {s.max():+.1f}")
LAB = {"T13": "T13 uniform mixture (pre-selected pair per task)"}
rows = []
for i, g in D.groupby("id", sort=False):
    rows.append(dict(id=i, system=LAB.get(i, g.system.iloc[0].split(" (one 5-fold")[0]), protocol=g.protocol.iloc[0], tasks=",".join(g.task), clean=rng(g.clean_ratio), field=rng(g.field_ratio),
                     control=rng(g.clean_ratio_vs_control) if "clean_ratio_vs_control" in g else "", worst=sh(g.shift_dworst), reference=g.reference_kind.iloc[0], verdict=g.overall_verdict.iloc[0]))
BS = pd.DataFrame(rows); BS.to_csv(T / "by_system.csv", index=False)
order = {"3": 0, "2": 1, "2/old": 2}; BS = BS.assign(o=BS.protocol.astype(str).map(order)).sort_values("o", kind="stable")
L_ = ["| System | Protocol | Tasks | Variance ratio vs best single (clean; field) | Ratio vs same-source uniform control (clean) | Shift worst case, system minus reference (points) | Reference | Verdict |", "|---|---|---|---|---|---|---|---|"]
for r_ in BS.itertuples():
    L_.append(f"| {r_.system} | {r_.protocol} | {r_.tasks} | {r_.clean or 'n/a'}; {r_.field or 'n/a'} | {r_.control or 'n/a'} | {r_.worst} | {r_.reference} | {r_.verdict} |")
for nm, pr, ts, why in (("RL router control (T4)", "n/a", "n/a", "not run (planned, never built)"), ("Information-based routing losses", "n/a", "n/a", "not run (planned, never built)"), ("Idea 2 learned deferral", "2", "esc, urban, quinn", "not determinable (seed 0 only, aggregates only)"),
                        ("Idea 3 disagreement", "2", "esc, urban, quinn", "not computed (logits only, 1 to 2 seeds)"), ("Idea 1 stage 1 head cascades", "2", "esc, urban, quinn", "not computed (logits only, 1 to 2 seeds)"),
                        ("Idea 5b hull", "3", "esc50, esc5, urban", "not computed (no pre-registered pick)"), ("T9 any-precision", "3", "esc50, esc5, urban", "not determinable (per-fold differences only)"), ("T12 adapter expert", "3", "urban", "not determinable (fold 1 only)")):
    L_.append(f"| {nm} | {pr} | {ts} | n/a | n/a | n/a | n/a | {why} |")
(T / "compact_by_system.md").write_text("\n".join(L_) + "\n"); print("compact table", len(L_) - 2, "rows")
