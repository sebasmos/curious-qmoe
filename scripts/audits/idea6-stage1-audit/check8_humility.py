"""Check 8: does the margin gate know when the cheap tier does not know?

For each task's primary design (the claim's inner-validation pick, view i):
  * AUROC of 1 - margin for "cheap wrong, escalated decision right" (the rule's decision) and for
    "cheap wrong", per seed pooled over folds; also per (seed, fold).
  * Top-label ECE (15 bins) and confidence minus accuracy of every cheap head and of the AST tier.
  * Random gate: escalate a uniformly random set of exactly the margin gate's test count, 2000 draws
    per (seed, fold), expected F1.
  * Audio-statistic gates: escalate by clip loudness (RMS dBFS, active RMS dBFS) or spectral flatness.
    Direction (low or high) and threshold are set on each fold's inner validation only, at the
    design's inner-validation escalation target (30%), the same way as the margin gate.
  * Paired fold-level comparison of the margin gate with each alternative.

    python check8_humility.py OUT
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

H = Path.home() / "Documents"
import os; RD = Path(os.environ.get("RESULTS_DIR", str(H / "results")))  # results root (archive edit: path only)
I6 = RD / "2026-10-22_0953_redesign-idea6-stage1_code-ec70260"
S2 = RD / "2026-10-20_2359_redesign-idea5-stage2_code-3c18c38"
OUT = Path(sys.argv[1])
TASK = {"esc50": ("effesc50_s{}", "astesc50_s{}", range(5), 50, "esc50", ("esc", "ast", 0.30)),
        "esc5": ("esc_singles_s{}", "astesc5_s{}", range(5), 5, "esc50", ("4", "ast", 0.30)),
        "urban": ("urban_singles_s{}", "asturban_s{}", range(3), 10, "urban", ("esc", "avg", 0.30))}
EFF = ["esc", "8", "4", "bitnet"]
STATS = ["rms_db", "active_rms_db", "flatness"]


def softmax(z):
    z = z.astype(np.float64)
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load(p):
    z = np.load(p)
    folds = sorted(int(f.split("_")[-1]) for f in z.files if f.startswith("iv_idx_"))
    return {"y": z["y"].astype(int), "tf": z["test_fold"].astype(int), "p": softmax(z["test_logits"]),
            "iv": {f: (z[f"iv_idx_{f}"].astype(int), softmax(z[f"iv_logits_{f}"])) for f in folds}}


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def mf1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    b = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((b == i).mean() * abs((pred[b == i] == y[b == i]).mean() - conf[b == i].mean())
                     for i in range(bins) if (b == i).any())), float(conf.mean() - (pred == y).mean())


def thr_rate(score_iv, rate):
    """escalate score < t; tie-free cut with rate closest to `rate` (ties: fewer)."""
    ms = np.sort(score_iv)
    n = len(ms)
    js = np.array([0] + [j for j in range(1, n) if ms[j - 1] < ms[j]] + [n])
    ts = np.array([-np.inf] + [(ms[j - 1] + ms[j]) / 2 for j in js[1:-1]] + [np.inf])
    r = js / n
    return ts[int(np.lexsort((r, np.abs(r - rate)))[0])]


def main():
    au = pd.read_csv(OUT / "audio_stats.csv")
    rows, auc_rows, cal_rows = [], [], []
    for task, (etag, atag, seeds, K, ds, (k, rule, rate)) in TASK.items():
        A = au[au.dataset == ("esc50" if ds == "esc50" else "urban")].sort_values("row")
        stat = {s: A[s].to_numpy() for s in STATS}
        rng = np.random.default_rng(11)
        for s in seeds:
            ch = {kk: load(I6 / "scores" / task / etag.format(s) / f"{kk}.npz") for kk in EFF}
            at = load(S2 / "scores" / task / atag.format(s) / "esc" / "w4g128__clean.npz")
            y_all = ch[k]["y"]
            pc_all, pe_all = ch[k]["p"], at["p"]
            de_all = pe_all.argmax(1) if rule == "ast" else ((pc_all + pe_all) / 2).argmax(1)
            wrong = pc_all.argmax(1) != y_all
            tgt = wrong & (de_all == y_all)
            sc = 1 - margin(pc_all)
            r = {"task": task, "seed": s, "n": len(y_all), "frac_cheap_wrong": wrong.mean(), "frac_fixable": tgt.mean(),
                 "frac_breakable": float(((~wrong) & (de_all != y_all)).mean()),
                 "auroc_margin_fixable": roc_auc_score(tgt, sc), "auroc_margin_wrong": roc_auc_score(wrong, sc)}
            for st in STATS:
                r[f"auroc_{st}_fixable"] = roc_auc_score(tgt, stat[st])
            auc_rows.append(r)
            for kk in EFF + ["ast:w4g128"]:
                for f in sorted(ch[k]["iv"]):
                    te = ch[k]["tf"] == f
                    p = at["p"][te] if kk == "ast:w4g128" else ch[kk]["p"][te]
                    e, gap = ece(p, y_all[te])
                    cal_rows.append({"task": task, "system": kk, "seed": s, "fold": f, "ece": e, "conf_minus_acc": gap})
            for f in sorted(ch[k]["iv"]):
                te = np.flatnonzero(ch[k]["tf"] == f)
                ivx, pcv = ch[k]["iv"][f]
                pev = at["iv"][f][1]
                y, yv = y_all[te], y_all[ivx]
                pc, pe = pc_all[te], pe_all[te]
                ac, acv = pc.argmax(1), pcv.argmax(1)
                de = de_all[te]
                dev = pev.argmax(1) if rule == "ast" else ((pcv + pev) / 2).argmax(1)
                t = thr_rate(margin(pcv), rate)
                esc = margin(pc) < t
                n_esc = int(esc.sum())
                f_margin = mf1(y, np.where(esc, de, ac))
                # random gate with the same count
                fr = []
                for _ in range(2000):
                    e2 = np.zeros(len(y), bool)
                    e2[rng.choice(len(y), n_esc, replace=False)] = True
                    fr.append(mf1(y, np.where(e2, de, ac)))
                row = {"task": task, "seed": s, "fold": f, "rate_margin": esc.mean(), "f1_margin": f_margin,
                       "f1_random_same_count": float(np.mean(fr)), "f1_cheap": mf1(y, ac), "f1_ast": mf1(y, pe.argmax(1)),
                       "auroc_margin_fixable_fold": roc_auc_score(tgt[te], sc[te]) if 0 < tgt[te].sum() < len(te) else np.nan}
                for st in STATS:
                    best = None
                    for sign in (1, -1):        # +1: escalate low values; -1: escalate high values
                        sv, stt = sign * stat[st][ivx], sign * stat[st][te]
                        tt = thr_rate(sv, rate)
                        ev = sv < tt
                        fiv = mf1(yv, np.where(ev, dev, acv))
                        if best is None or fiv > best[0]:
                            best = (fiv, sign, tt)
                    _, sign, tt = best
                    e3 = sign * stat[st][te] < tt
                    row[f"f1_{st}"] = mf1(y, np.where(e3, de, ac))
                    row[f"rate_{st}"] = float(e3.mean())
                    row[f"dir_{st}"] = "low" if sign == 1 else "high"
                rows.append(row)
        print(task, "done", flush=True)
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "check8_gates_per_pair.csv", index=False)
    AU = pd.DataFrame(auc_rows)
    AU.to_csv(OUT / "check8_auroc.csv", index=False)
    CA = pd.DataFrame(cal_rows)
    CA.to_csv(OUT / "check8_calibration.csv", index=False)
    summ = []
    for task, g in G.groupby("task", sort=False):
        fm = g.groupby("fold").mean(numeric_only=True)
        out = {"task": task, "rate_margin": g.rate_margin.mean(), "f1_margin": g.f1_margin.mean(),
               "f1_cheap": g.f1_cheap.mean(), "f1_ast": g.f1_ast.mean()}
        for alt in ["f1_random_same_count"] + [f"f1_{s}" for s in STATS]:
            d = fm.f1_margin - fm[alt]
            out[alt] = g[alt].mean()
            out[f"margin_minus_{alt[3:]}_pts"] = 100 * d.mean()
            out[f"p_fold_{alt[3:]}"] = float(stats.ttest_1samp(d, 0).pvalue)
            out[f"pos_folds_{alt[3:]}"] = f"{int((d > 0).sum())} of {len(d)}"
        for s in STATS:
            out[f"rate_{s}"] = g[f"rate_{s}"].mean()
            out[f"dir_{s}"] = g[f"dir_{s}"].value_counts().to_dict()
        a = AU[AU.task == task]
        out.update({"auroc_margin_fixable_mean": a.auroc_margin_fixable.mean(), "auroc_margin_fixable_min_seed": a.auroc_margin_fixable.min(),
                    "auroc_margin_wrong_mean": a.auroc_margin_wrong.mean(),
                    "auroc_margin_fixable_min_fold": g.auroc_margin_fixable_fold.min(),
                    **{f"auroc_{s}_fixable": a[f"auroc_{s}_fixable"].mean() for s in STATS},
                    "frac_fixable": a.frac_fixable.mean(), "frac_breakable": a.frac_breakable.mean()})
        c = CA[CA.task == task].groupby("system")[["ece", "conf_minus_acc"]].mean()
        out["ece"] = {i: [round(v.ece, 4), round(v.conf_minus_acc, 4)] for i, v in c.iterrows()}
        summ.append(out)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "check8_summary.csv", index=False)
    print(S.T.to_string())


if __name__ == "__main__":
    main()
