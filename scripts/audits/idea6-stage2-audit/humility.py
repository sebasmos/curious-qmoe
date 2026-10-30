"""Check 8: escalation against SNR with clean thresholds, decomposed into "more cheap errors" and "less
confidence at the same correctness"; gate AUROC per condition with its spread over seeds and folds; the H1
design against the hull in every condition (E1); calibration before and after temperature scaling.

    python -B humility.py esc50|esc5|urban
Writes outputs/humility_<task>.json and .csv files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AST_HEADS, CELLS, OUT, R, S2, SHORT_CFGS, TASKS, Run, f1m, fold_means, fold_tests,  # noqa
                    hull_pair, margin, mix_f1_exact, sm, threshold)

task = sys.argv[1]
T = TASKS[task]
K, seeds, SL = T["K"], T["seeds"], T["SL"]
pick = json.loads((R / f"analysis/{task}/picks.json").read_text())["H1"]["design"]
bb, kind, tier, rule, budget = pick
rate_target = {"b05": .05, "b10": .1, "b20": .2, "b30": .3, "b50": .5}[budget]
sys.argv = [sys.argv[0], task]
C = json.loads((OUT / "costs_audit.json").read_text())
claimed = json.loads((R / f"analysis/{task}/costs.json").read_text())

conds = ["clean"] + [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (20, 10, 5, 0)] + ["field"]
eff = {s: Run(R / "scores" / task / T["eff"].format(s) / kind) for s in seeds}
full = {(s, h): Run(S2 / "scores" / task / T["ast"].format(s) / h) for s in seeds for h in AST_HEADS}
short = {(s, h): Run(R / "scores_ast_short" / task / T["short"].format(s) / h) for s in seeds for h in AST_HEADS}
cheap_all = {(s, k): Run(R / "scores" / task / T["eff"].format(s) / k) for s in seeds for k in ("esc", "8", "4", "bitnet")}
src = {}
for p in sorted((S2 / "scores" / task / T["ast"].format(seeds[0]) / "esc").glob("*__clean.npz")):
    src[p.name.split("__")[0]] = (full, p.name.split("__")[0])
for c in SHORT_CFGS:
    src[f"{c}@{SL}"] = (short, c)


def has(name, cond):
    """Every (seed, head) file exists and, outside the field mixture, holds finite logits on every row
    (UrbanSound8K short-AST field cells hold only the rows the field assignments need)."""
    d, b = src[name]
    if not all(r.has(b, cond) for r in d.values()):
        return False
    if cond == "field":
        return True
    return all(np.isfinite(r.raw(b, cond)["z"]).all() for r in d.values())


def ast_cost(name, h):
    cfg, _, L_ = name.partition("@")
    L_ = int(L_) if L_ else 1024
    key = f"{cfg}@{L_}"
    if key in C["ast_priced"]:
        b = C["ast_priced"][key]
    else:
        from costs_audit import ast_bitops
        b = ast_bitops(cfg, L_)
    return C["fbank"][T["emb"]] + b + claimed["ast_heads"][h]["bitops"]


def cheap_cost(b_, k):
    return C["mel_mean_bitops"][T["emb"]] + (C["effnet_bitops"]["fp32"] if b_ == "fp32" else
                                            C["effnet_bitops"]["w8a8_priced"]) + claimed["eff_heads"][k]["bitops"]


def fit_T(z, y):
    def nll(lt):
        q = z / np.exp(lt)
        q = q - q.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(q).sum(1)) - q[np.arange(len(y)), y]))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def ece(conf, ok, bins=15):
    e = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, e[1:-1]), 0, bins - 1)
    out = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            out += abs(ok[m].mean() - conf[m].mean()) * m.sum()
    return out / len(conf)


ref = eff[seeds[0]].get(bb, "clean")
folds = sorted(ref["iv"])
n_all = len(ref["y"])
n_te = np.array([np.sum(ref["fold"] == f) for f in folds])
NB = float(np.mean(n_te / (n_all - n_te)))
thr, temps = {}, {}
for s in seeds:
    r = eff[s].get(bb, "clean")
    for f in folds:
        z = r["iv"][f][1]
        thr[(s, f)] = threshold(margin(sm(z)), rate_target)
        temps[(s, f)] = fit_T(z, r["y"][r["iv"][f][0]])

esc_rows, auc_rows, cal_rows, e1_rows = [], [], [], []
for cond in conds:
    try:
        rs = {s: eff[s].get(bb, cond) for s in seeds}
    except FileNotFoundError:
        continue
    casc_ok = has(tier, cond)
    statics = [n for n in src if has(n, cond)] if casc_ok else []
    per = {}
    x_c, h_c, keys = [], [], []
    st_f1 = {}
    st_pred = {}
    cas = {}
    for s in seeds:
        r = rs[s]
        for f in folds:
            te = np.flatnonzero(r["fold"] == f)
            y = r["y"][te]
            p = sm(r["z"][te])
            m = margin(p)
            esc = m < thr[(s, f)]
            wrong = p.argmax(1) != y
            per[(s, f)] = dict(rate=esc.mean(), p_wrong=wrong.mean(), esc_w=esc[wrong].mean() if wrong.any() else np.nan,
                               esc_r=esc[~wrong].mean() if (~wrong).any() else np.nan)
            pt = sm(r["z"][te] / temps[(s, f)])
            cal_rows.append(dict(task=task, cond=cond, seed=s, fold=f, T=temps[(s, f)], ece=ece(p.max(1), ~wrong),
                                 gap=p.max(1).mean() - (~wrong).mean(), ece_T=ece(pt.max(1), pt.argmax(1) == y),
                                 gap_T=pt.max(1).mean() - (pt.argmax(1) == y).mean()))
            if casc_ok:
                pe = sm(src[tier][0][(s, "esc")].get(src[tier][1], cond)["z"][te])
                de = pe.argmax(1) if rule == "ast" else (p + pe).argmax(1)
                pred = np.where(esc, de, p.argmax(1))
                cas[(s, f)] = (f1m(y, pred, K), esc.mean())
                fixable = wrong & (pe.argmax(1) == y)
                per[(s, f)]["fixable"] = fixable
                for n in statics:
                    for h in AST_HEADS:
                        pr = sm(src[n][0][(s, h)].get(src[n][1], cond)["z"][te]).argmax(1)
                        st_pred[(f"ast:{n}+{h}", s, f)] = pr
                        st_f1.setdefault(f"ast:{n}+{h}", []).append(f1m(y, pr, K))
                for b_ in ("fp32", "w8a8"):
                    for k in ("esc", "8", "4", "bitnet"):
                        pr = sm(cheap_all[(s, k)].get(b_, cond)["z"][te]).argmax(1)
                        st_pred[(f"eff:{b_}:{k}", s, f)] = pr
                        st_f1.setdefault(f"eff:{b_}:{k}", []).append(f1m(y, pr, K))
            per[(s, f)]["score"], per[(s, f)]["wrong"] = 1 - m, wrong
            per[(s, f)]["y"] = y
    # escalation decomposition
    df = pd.DataFrame([{k: v for k, v in d.items() if np.isscalar(v)} | {"seed": s, "fold": f}
                       for (s, f), d in per.items()])
    esc_rows.append(dict(task=task, cond=cond, rate=df.rate.mean(), p_wrong=df.p_wrong.mean(),
                         esc_given_wrong=df.esc_w.mean(), esc_given_right=df.esc_r.mean()))
    # AUROC: per seed pooled over folds, and per (seed, fold)
    for s in seeds:
        sc = np.concatenate([per[(s, f)]["score"] for f in folds])
        w = np.concatenate([per[(s, f)]["wrong"] for f in folds])
        row = dict(task=task, cond=cond, seed=s, auroc_wrong=roc_auc_score(w, sc),
                   auroc_wrong_fold_min=min(roc_auc_score(per[(s, f)]["wrong"], per[(s, f)]["score"]) for f in folds))
        bs = np.random.default_rng(s)
        boots = []
        for _ in range(300):
            i = bs.integers(0, len(sc), len(sc))
            if 0 < w[i].sum() < len(i):
                boots.append(roc_auc_score(w[i], sc[i]))
        row["auroc_wrong_ci_lo"], row["auroc_wrong_ci_hi"] = np.percentile(boots, [2.5, 97.5])
        if casc_ok:
            fx = np.concatenate([per[(s, f)]["fixable"] for f in folds])
            row["auroc_fixable"] = roc_auc_score(fx, sc) if 0 < fx.sum() < len(fx) else np.nan
        auc_rows.append(row)
    # E1: H1 design against the hull of this condition's statics
    if casc_ok:
        names = sorted(st_f1)
        f1s = [np.mean(st_f1[n]) for n in names]
        cs = [cheap_cost(*n.split(":")[1:]) if n.startswith("eff") else ast_cost(*n[4:].split("+")) for n in names]
        keys = [(s, f) for s in seeds for f in folds]
        x = np.array([cas[k][0] for k in keys])
        cost = float(np.mean([cheap_cost(bb, kind) + cas[k][1] * ast_cost(tier, "esc") for k in keys]))
        v, lo, hi, lam = hull_pair(names, f1s, cs, cost)
        h = np.array([st_f1[lo][i] if lam == 0 else mix_f1_exact(per[k]["y"], st_pred[(lo, *k)], st_pred[(hi, *k)], lam, K)
                      for i, k in enumerate(keys)])
        fm = fold_means(x - h, [f for _, f in keys])
        tt = fold_tests(fm, NB)
        best_le = max((f1s[i], names[i]) for i in range(len(names)) if cs[i] <= cost)
        e1_rows.append(dict(task=task, cond=cond, f1=x.mean(), cost_T=cost / 1e12, rate=np.mean([cas[k][1] for k in keys]),
                            hull=f"{lo}/{hi}({lam:.2f})", hull_f1=h.mean(), diff_pts=100 * tt["mean"], p_t=tt["p_t"],
                            p_nb=tt["p_nb"], fold_min_pts=100 * tt["fold_min"], fold_max_pts=100 * tt["fold_max"],
                            best_static_le_cost=best_le[1], best_static_f1=best_le[0],
                            n_statics=len(names)))
    for d in list(eff.values()) + list(full.values()) + list(short.values()) + list(cheap_all.values()):
        d.clear()
    print(task, cond, "done", flush=True)

ESC = pd.DataFrame(esc_rows)
clean = ESC[ESC.cond == "clean"].iloc[0]
ESC["rate_if_only_errors_changed"] = ESC.p_wrong * clean.esc_given_wrong + (1 - ESC.p_wrong) * clean.esc_given_right
ESC["share_of_rise_from_more_errors"] = (ESC.rate_if_only_errors_changed - clean.rate) / (ESC.rate - clean.rate)
ESC.to_csv(OUT / f"humility_{task}_escalation.csv", index=False)
AU = pd.DataFrame(auc_rows)
AU.to_csv(OUT / f"humility_{task}_auroc.csv", index=False)
CAL = pd.DataFrame(cal_rows).groupby(["task", "cond"]).mean(numeric_only=True).reset_index()
CAL.to_csv(OUT / f"humility_{task}_calibration.csv", index=False)
E1 = pd.DataFrame(e1_rows)
E1.to_csv(OUT / f"humility_{task}_e1.csv", index=False)
pd.set_option("display.width", 250)
print(ESC.round(3).to_string())
print(AU.groupby("cond").agg(auroc_wrong=("auroc_wrong", "mean"), lo=("auroc_wrong_ci_lo", "min"),
                             fold_min=("auroc_wrong_fold_min", "min"), fixable=("auroc_fixable", "mean")).round(3).to_string())
print(CAL.round(3).to_string())
print(E1.round(4).to_string())
