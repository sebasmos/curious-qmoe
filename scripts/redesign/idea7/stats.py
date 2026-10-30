"""Idea 7 stage 1: primary test, exploratory family, humility and cost tables
from defer.py outputs (see defer.py for the pre-specified analysis).

Primary (9 hypotheses): per task, device n3g64 + FP32 head, rule pre, field
mixture; Claim A (non-inferiority to the cloud by 0.5 points), Claim B against
random deferral and against the noise_lr detector at the same realized rate.
One-sided one-sample t-tests on paired differences, over folds (seeds
averaged within fold) and over (seed, fold); Holm over the 9 within each
pairing; a hypothesis holds when both Holm-adjusted p < 0.05; GO when all 9
hold.

Exploratory: every other (task, device, head, rule, condition) against the
cloud (non-inferiority, 0.5 points), random, noise_lr, snr_est, flatness,
err_lr, the device alone and the next device up alone, one-sided; family size
and Holm within the family are reported.

    python scripts/redesign/idea7/stats.py --out OUT/outputs --tasks esc50 esc5 urban
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

NI = 0.005
PRIMARY = {"device": "n3g64", "head": "esc", "rule": "pre", "condition": "field"}
# esc5: class_id from data/esc/esc-50-data_processing.py, five fixed groups of ten ESC-50 targets
# ({0-4, 25-29}, {5-9, 30-34}, {10-14, 35-39}, {15-19, 40-44}, {20-24, 45-49}); the official ESC-50
# categories are target // 10, a different grouping
TASK_NAMES = {"esc50": "ESC-50 (50 classes)", "esc5": "ESC-50 (5 groups of 10)", "urban": "UrbanSound8K"}
BASELINES = ["random", "noise_lr", "snr_est", "flatness", "err_lr"]
COND_ORDER = ["clean"] + [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (20, 10, 5, 0)] + ["field"]


def cond_key(c):
    return COND_ORDER.index(c) if c in COND_ORDER else 99


def short(c):
    return c.replace("__", " ").replace("db", " dB")


def ttest_greater(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    if np.std(x) == 0:
        return 0.0 if x.mean() > 0 else 1.0
    return float(ss.ttest_1samp(x, 0.0, alternative="greater").pvalue)


def ttest_two(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.std(x) == 0:
        return np.nan
    return float(ss.ttest_1samp(x, 0.0).pvalue)


def holm(p):
    p = np.asarray(p, float)
    adj = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m, run = len(ok), 0.0
    for i, j in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def two_level(df, col):
    """(mean, fold-level p, pair-level p) of a one-sided test on df[col] > 0."""
    fold = df.groupby("fold")[col].mean()
    return float(df[col].mean()), ttest_greater(fold.values), ttest_greater(df[col].values), len(fold), len(df)


def load(out, tasks):
    P, D, costs = [], [], {}
    for t in tasks:
        P.append(pd.read_csv(out / t / "pairs.csv.gz"))
        D.append(pd.read_csv(out / t / "device.csv.gz"))
        costs[t] = json.loads((out / t / "costs.json").read_text())
    P, D = pd.concat(P, ignore_index=True), pd.concat(D, ignore_index=True)
    keys = ["task", "device", "head", "seed", "fold", "condition"]
    M = P.merge(D[keys + ["f1_device", "f1_cloud", "f1_next", "next", "f1_oracle", "oracle_rate"]], on=keys)
    assert len(M) == len(P)
    M["d_A"] = M["f1_sys"] - M["f1_cloud"] + NI
    for b in BASELINES:
        M[f"d_{b}"] = M["f1_sys"] - M[f"f1_{b}"]
    M["d_device"] = M["f1_sys"] - M["f1_device"]
    M["d_next"] = M["f1_sys"] - M["f1_next"]
    return M, D, costs


def primary(M, tasks):
    rows = []
    for t in tasks:
        g = M[(M["task"] == t) & np.logical_and.reduce([M[k] == v for k, v in PRIMARY.items()])]
        for claim, col in [("A: within 0.5 of cloud", "d_A"), ("B: beats random", "d_random"),
                           ("B: beats noise detector", "d_noise_lr")]:
            mean, pf, pp, nf, npairs = two_level(g, col)
            rows.append({"task": t, "claim": claim, "col": col, "mean_diff_pts": 100 * (mean - (NI if col == "d_A"
                                                                                                else 0)),
                         "p_fold": pf, "p_pair": pp, "n_folds": nf, "n_pairs": npairs,
                         "f1_sys": g["f1_sys"].mean(), "f1_cloud": g["f1_cloud"].mean(),
                         "f1_random": g["f1_random"].mean(), "f1_noise_lr": g["f1_noise_lr"].mean(),
                         "f1_device": g["f1_device"].mean(), "f1_next": g["f1_next"].mean(),
                         "f1_oracle_k": g["f1_oracle_k"].mean(), "rate": g["rate"].mean(),
                         "iv_rate": g["iv_rate"].mean()})
    R = pd.DataFrame(rows)
    R["holm_fold"], R["holm_pair"] = holm(R["p_fold"]), holm(R["p_pair"])
    R["holds"] = (R["holm_fold"] < 0.05) & (R["holm_pair"] < 0.05)
    return R


def exploratory(M):
    rows = []
    comps = [("A_cloud_NI", "d_A")] + [(f"B_{b}", f"d_{b}") for b in BASELINES] + [("vs_device", "d_device"),
                                                                                    ("vs_next", "d_next")]
    for key, g in M.groupby(["task", "device", "head", "rule", "condition"]):
        is_primary = dict(zip(["device", "head", "rule", "condition"], key[1:])) == PRIMARY
        for name, col in comps:
            if g[col].isna().all():
                continue
            if is_primary and name in ("A_cloud_NI", "B_random", "B_noise_lr"):
                continue
            mean, pf, pp, _, _ = two_level(g, col)
            rows.append(dict(zip(["task", "device", "head", "rule", "condition"], key), comparison=name,
                             mean_diff_pts=100 * (mean - (NI if col == "d_A" else 0)), p_fold=pf, p_pair=pp))
    E = pd.DataFrame(rows)
    E["holm_fold"], E["holm_pair"] = holm(E["p_fold"]), holm(E["p_pair"])
    E["holds"] = (E["holm_fold"] < 0.05) & (E["holm_pair"] < 0.05)
    return E


def fmt(x, nd=3):
    return "-" if not np.isfinite(x) else f"{x:.{nd}f}"


def fp(p):
    if not np.isfinite(p):
        return "-"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def md_table(df, cols, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(getattr(r, c)) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=["esc50", "esc5", "urban"])
    args = ap.parse_args(argv)
    M, D, costs = load(args.out, args.tasks)
    tasks = args.tasks

    R = primary(M, tasks)
    R.to_csv(args.out / "primary.csv", index=False)
    go = bool(R["holds"].all()) and len(R) == 3 * len(tasks) == 9
    E = exploratory(M)
    E.to_csv(args.out / "exploratory.csv", index=False)

    # per-condition means (FP32 head unless stated)
    agg = {c: (c, "mean") for c in ["rate", "iv_rate", "f1_sys", "f1_cloud", "f1_device", "f1_next", "f1_random",
                                    "f1_noise_lr", "f1_snr_est", "f1_flatness", "f1_err_lr", "f1_oracle_k",
                                    "f1_oracle", "oracle_rate", "cap_sys", "cap_random", "cap_noise_lr",
                                    "cap_snr_est", "cap_flatness", "cap_err_lr", "bits_sent_per_1000", "fixed",
                                    "broken", "n_rec", "n"]}
    C = M.groupby(["task", "device", "head", "rule", "condition"]).agg(**agg).reset_index()
    C["ck"] = C["condition"].map(cond_key)
    C = C.sort_values(["task", "device", "head", "rule", "ck"]).drop(columns="ck")
    C.to_csv(args.out / "conditions.csv", index=False)
    Dm = D.groupby(["task", "device", "head", "condition"]).mean(numeric_only=True).reset_index()
    Dm["ck"] = Dm["condition"].map(cond_key)
    Dm = Dm.sort_values(["task", "device", "head", "ck"]).drop(columns="ck")
    Dm.to_csv(args.out / "device_conditions.csv", index=False)

    # AUROC: margin against each audio score, fold-level paired t-test (two-sided), per condition
    au = []
    for key, g in D.groupby(["task", "device", "head", "condition"]):
        for tgt in ("wrong", "rec"):
            row = dict(zip(["task", "device", "head", "condition"], key), target=tgt,
                       margin=g[f"auc_{tgt}_margin"].mean())
            for b in ("noise_lr", "snr_est", "flatness", "err_lr"):
                row[b] = g[f"auc_{tgt}_{b}"].mean()
                diff = (g[f"auc_{tgt}_margin"] - g[f"auc_{tgt}_{b}"]).groupby(g["fold"]).mean()
                row[f"p_{b}"] = ttest_two(diff.values)
            au.append(row)
    AU = pd.DataFrame(au)
    AU["ck"] = AU["condition"].map(cond_key)
    AU = AU.sort_values(["task", "device", "head", "target", "ck"]).drop(columns="ck")
    AU.to_csv(args.out / "auroc.csv", index=False)

    # humility: monotone deferral as SNR falls, per noise type
    mono = []
    for (t, dev, head, rule), g in C.groupby(["task", "device", "head", "rule"]):
        r = dict(zip(g["condition"], g["rate"]))
        for kind in ("bg", "pink", "white"):
            seq = [c for c in ["clean"] + [f"{kind}__{s}db" for s in (20, 10, 5, 0)] if c in r]
            if len(seq) >= 3:
                v = [r[c] for c in seq]
                mono.append({"task": t, "device": dev, "head": head, "rule": rule, "kind": kind,
                             "conds": " ".join(seq), "rates": " ".join(f"{x:.3f}" for x in v),
                             "monotone": bool(np.all(np.diff(v) > 0))})
    MO = pd.DataFrame(mono)
    MO.to_csv(args.out / "monotone.csv", index=False)

    verdict = {"go": go, "primary": R.to_dict("records"), "exploratory_family_size": int(E["p_fold"].notna().sum()),
               "exploratory_holds": int(E["holds"].sum()), "monotone_rows": int(MO["monotone"].sum()),
               "monotone_total": int(len(MO)), "costs": costs}
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=float))

    # ── summary.md ──
    L = [f"# Idea 7 stage 1 summary\n\nVerdict: {'GO' if go else 'NO GO'} "
         f"({int(R['holds'].sum())} of {len(R)} primary hypotheses hold under both pairings after Holm).\n"]
    R2 = R.copy()
    R2["task_"] = R2["task"].map(TASK_NAMES)
    for c in ["f1_sys", "f1_cloud", "f1_random", "f1_noise_lr", "rate"]:
        R2[c + "_"] = R2[c].map(fmt)
    R2["diff_"] = R2["mean_diff_pts"].map(lambda x: f"{x:+.2f}")
    for c in ["p_fold", "p_pair", "holm_fold", "holm_pair"]:
        R2[c + "_"] = R2[c].map(fp)
    L.append("## Primary (n3g64 + FP32 head, clean-calibrated threshold, field mixture)\n")
    L.append(md_table(R2, ["task_", "claim", "f1_sys_", "f1_cloud_", "f1_random_", "f1_noise_lr_", "rate_", "diff_",
                           "p_fold_", "p_pair_", "holm_fold_", "holm_pair_", "holds"],
                      ["task", "claim", "F1 device+defer", "F1 cloud", "F1 random", "F1 noise det.", "deferral",
                       "diff (pts)", "p fold", "p pair", "Holm fold", "Holm pair", "holds"]))
    L.append("\nFor Claim A, diff is F1(device + deferral) minus F1(cloud); the test asks whether it is above -0.5.\n")

    for t in tasks:
        L.append(f"\n## {TASK_NAMES[t]}: every device, FP32 head, rule pre\n")
        g = C[(C["task"] == t) & (C["head"] == "esc") & (C["rule"] == "pre")].copy()
        for c in ["rate", "f1_sys", "f1_cloud", "f1_device", "f1_next", "f1_random", "f1_noise_lr", "f1_oracle_k",
                  "cap_sys", "cap_noise_lr"]:
            g[c + "_"] = g[c].map(fmt)
        g["cond_"] = g["condition"].map(short)
        L.append(md_table(g, ["device", "cond_", "rate_", "f1_sys_", "f1_cloud_", "f1_device_", "f1_next_",
                              "f1_random_", "f1_noise_lr_", "f1_oracle_k_", "cap_sys_", "cap_noise_lr_"],
                          ["device", "condition", "deferral", "F1 dev+defer", "F1 cloud", "F1 device", "F1 next up",
                           "F1 random", "F1 noise det.", "F1 oracle at k", "recoverable caught", "caught by noise det."]))
    L.append("\n## Humility curve: realized deferral rate, n3g64 + FP32 head, clean-calibrated thresholds\n")
    for t in tasks:
        g = C[(C["task"] == t) & (C["device"] == "n3g64") & (C["head"] == "esc")]
        pv = g.pivot(index="rule", columns="condition", values="rate")
        cols = sorted(pv.columns, key=cond_key)
        pv = pv[cols].reset_index()
        for c in cols:
            pv[c] = pv[c].map(lambda x: f"{x:.3f}")
        L.append(f"\n{TASK_NAMES[t]}\n")
        L.append(md_table(pv, ["rule"] + cols, ["rule"] + [short(c) for c in cols]))
    L.append(f"\nMonotone rise as SNR falls (clean, 20, 10, 5, 0 dB where present), every device, head and rule: "
             f"{int(MO['monotone'].sum())} of {len(MO)} rows. Not monotone:\n")
    bad = MO[~MO["monotone"]]
    L.append(md_table(bad, ["task", "device", "head", "rule", "kind", "rates"],
                      ["task", "device", "head", "rule", "noise", "rates"]) if len(bad) else "none\n")

    L.append("\n## Calibration of the device alone (ECE, 15 bins), FP32 head, before / after temperature scaling\n")
    for t in tasks:
        g = Dm[(Dm["task"] == t) & (Dm["head"] == "esc")]
        pv = g.assign(v=g["ece"].map(lambda x: f"{x:.3f}") + " / " + g["ece_temp"].map(lambda x: f"{x:.3f}"))
        pv = pv.pivot(index="device", columns="condition", values="v")
        cols = sorted(pv.columns, key=cond_key)
        pv = pv[cols].reset_index()
        temps = g[g["condition"] == "clean"].set_index("device")["temperature"]
        pv["T"] = pv["device"].map(lambda d: f"{temps[d]:.2f}")
        L.append(f"\n{TASK_NAMES[t]}\n")
        L.append(md_table(pv, ["device", "T"] + cols, ["device", "T"] + [short(c) for c in cols]))

    L.append("\n## Finding the device's errors: AUROC for 'device wrong', FP32 head (margin / noise det. / SNR "
             "estimate / err_lr)\n")
    for t in tasks:
        g = AU[(AU["task"] == t) & (AU["head"] == "esc") & (AU["target"] == "wrong")].copy()
        g["v"] = (g["margin"].map(fmt) + " / " + g["noise_lr"].map(fmt) + " / " + g["snr_est"].map(fmt) + " / "
                  + g["err_lr"].map(fmt))
        pv = g.pivot(index="device", columns="condition", values="v")
        cols = sorted(pv.columns, key=cond_key)
        L.append(f"\n{TASK_NAMES[t]}\n")
        L.append(md_table(pv[cols].reset_index(), ["device"] + cols, ["device"] + [short(c) for c in cols]))

    L.append("\n## Costs (FP32 head, rule pre)\n")
    rows = []
    for t in tasks:
        cs = costs[t]
        for dev in ["n3g64", "w3g64", "n4", "w4g128", "w8g64", "fp32"]:
            b = cs["backbone"][dev]
            r = {"task": TASK_NAMES[t], "device": dev, "mem": f"{b['storage_MB'] + cs['head_MB']['esc']:.1f}",
                 "bits": f"{cs['mean_clip_bits'] / 1e6:.3f}"}
            for cond in ("clean", "field"):
                g = C[(C["task"] == t) & (C["device"] == dev) & (C["head"] == "esc") & (C["rule"] == "pre")
                      & (C["condition"] == cond)]
                r[cond] = (f"{1000 * g['rate'].iloc[0]:.0f} calls, {g['bits_sent_per_1000'].iloc[0] / 1e6:.0f} Mb"
                           if len(g) else "-")
            rows.append(r)
    L.append(md_table(pd.DataFrame(rows), ["task", "device", "mem", "bits", "clean", "field"],
                      ["task", "device", "memory (MB, with FP32 head)", "Mb per deferred clip",
                       "per 1000 clean clips", "per 1000 field clips"]))
    L.append(f"\nExploratory family: {verdict['exploratory_family_size']} one-sided tests; "
             f"{verdict['exploratory_holds']} hold after Holm within the family under both pairings.\n")
    (args.out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:3]))
    print(f"GO: {go}")


if __name__ == "__main__":
    main()
