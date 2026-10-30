#!/usr/bin/env python3
"""Offline cost-aware router pilot (redesign idea 2, stage 1).

For every outer test fold k of one dataset, a router learns from held-in clips
which precision model is right on which clip, and is then applied once to
fold k with the protocol-v2 single models of that fold.

Held-in sources (never the test fold):
  iv  the fold's inner validation split, scored by the fold's own singles
      (the split that selected their checkpoints; a random split, so it leaks
      recordings and its error rate is far below the test fold's)
  cf  cross-fitting: every training fold j of outer fold k, scored by the
      single model trained without folds j and k (crossfit_train.py)
  oof every training fold j of outer fold k, scored by the protocol-v2
      single of outer fold j (its own test fold). Correctness on fold j is
      honest, but that model trained on fold k, so fold k's labels reach the
      router one step removed: a free check on cf, not a clean result

Tasks:
  pair L->H   defer target "L wrong and H right"; defer when the router's
              score is above a threshold
  3way        {bitnet, 4, esc}, target "cheapest correct model" (4 classes,
              the last one "none right"); the clip goes to the model m that
              maximises P(cheapest correct <= m) - lambda * cost(m) / cost(esc)

Routers (sklearn, standardised inputs):
  lr_out, mlp_out    on the cheap model's log-probabilities, max prob, margin
  lr_emb             on the 1536-d embedding
  lr_both, mlp_both  on both
Baseline (idea 1): the cheap model's softmax margin; for 3way the two-stage
margin cascade bitnet -> 4 -> esc.

Operating points are fixed on held-in data only: pair tasks match held-in
deferral rates; 3way picks lambda for a held-in mean bit-op budget, the
margin cascade the two rates with the best held-in F1 within the budget. The
router's own held-in scores are out of fold (3 groups of training folds for
cf, 3-fold for iv); the final router is refit on all held-in clips and scores
the test fold once.

Cost per clip in bit operations (MACs x weight bits x activation bits; the
router counted as FP32 linear layers, 32 x 32, its scaler folded into the
first layer). A router that reads the cheap model's output pays for the cheap
model on every clip; an embedding-only router pays only for the model it
picks.

    python router_pilot.py --res <results dir> --ds urban --csv <embeddings csv> [--sources iv cf]
    python router_pilot.py ... --summarize      # tables from the saved fold records
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True)
ap.add_argument("--ds", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--label-col", default="class_id")
ap.add_argument("--sources", nargs="+", default=["iv", "cf"])
ap.add_argument("--folds", type=int, nargs="*")
ap.add_argument("--summarize", action="store_true")
args = ap.parse_args()

RES = Path(args.res)
SING = RES / "singles" / args.ds
CF = RES / "crossfit" / args.ds
OUT = RES / "router" / args.ds
OUT.mkdir(parents=True, exist_ok=True)
SINGLES = ["bitnet", "1", "2", "4", "8", "16", "esc", "qesc"]
PAIRS = [("bitnet", "4"), ("bitnet", "esc"), ("1", "esc"), ("4", "esc")]
THREE = ["bitnet", "4", "esc"]
RATES = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
ROUTERS = {"lr_out": ("lr", "out"), "lr_emb": ("lr", "emb"), "lr_both": ("lr", "both"),
           "mlp_out": ("mlp", "out"), "mlp_both": ("mlp", "both")}
HIDDEN = {"out": 32, "both": 16}
LR_C = {"out": 1.0, "emb": 0.01, "both": 0.01}
LAMBDAS = np.concatenate([[0.0], np.logspace(-4, 2, 241)])
R1 = [0, .02, .05, .1, .15, .2, .3, .4, .5, .7, 1.0]
R2 = [0, .05, .1, .2, .3, .5]

chk = json.loads((SING / "checks.json").read_text())
assert max(c["diff_te"] for c in chk["checks"]) < 1e-9, "singles do not reproduce metrics.json"
C = {m: float(c[1]) for m, c in chk["costs"].items()}
BUDGETS = [C["bitnet"] * x for x in (1.2, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30)]
ALL_FOLDS = sorted(int(p.stem.split("_")[1]) for p in SING.glob("fold_*.npz"))
SF = {k: dict(np.load(SING / f"fold_{k}.npz")) for k in ALL_FOLDS}
NC = SF[ALL_FOLDS[0]]["p_te_esc"].shape[1]


def f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def margin(P):
    s = np.sort(P, 1)
    return s[:, -1] - s[:, -2]


def out_feats(P):
    p = np.clip(P, 1e-12, 1)
    s = np.sort(p, 1)
    return np.hstack([np.log(p), s[:, -1:], s[:, -1:] - s[:, -2:-1]])


def router_bitops(name, n_out):
    fam, kind = ROUTERS[name]
    d = {"out": NC + 2, "emb": 1536, "both": 1536 + NC + 2}[kind]
    macs = d * n_out if fam == "lr" else d * HIDDEN[kind] + HIDDEN[kind] * n_out
    return float(macs * 32 * 32)


def threshold(scores, rate):
    """Defer iff score > t, with a share `rate` of held-in clips above t."""
    if rate <= 0:
        return np.inf
    return float(np.quantile(scores, 1 - rate, method="lower")) if rate < 1 else -np.inf


# ── per-fold router runs ───────────────────────────────────────────────────

def make(name):
    fam, kind = ROUTERS[name]
    est = (LogisticRegression(C=LR_C[kind], max_iter=300) if fam == "lr"
           else MLPClassifier((HIDDEN[kind],), alpha=1e-2, max_iter=150, random_state=0))
    return make_pipeline(StandardScaler(), est)


def proba(model, X, n_cls):
    P = np.zeros((len(X), n_cls))
    P[:, model.classes_] = model.predict_proba(X)
    return P


def fit_predict(name, Xh, th, groups, Xt, n_cls):
    """Out-of-fold held-in probabilities, and test probabilities of the
    router refitted on all held-in clips."""
    if groups is not None:
        parts = [np.isin(groups, g) for g in np.array_split(np.unique(groups), 3)]
    else:
        parts = [np.isin(np.arange(len(th)), te) for _, te in KFold(3, shuffle=True, random_state=0).split(Xh)]
    oof = np.zeros((len(th), n_cls))
    for va in parts:
        if len(np.unique(th[~va])) < 2:
            oof[va] = np.bincount(th[~va], minlength=n_cls) / (~va).sum()
        else:
            oof[va] = proba(make(name).fit(Xh[~va], th[~va]), Xh[va], n_cls)
    if len(np.unique(th)) < 2:
        return oof, np.tile(np.bincount(th, minlength=n_cls) / len(th), (len(Xt), 1))
    return oof, proba(make(name).fit(Xh, th), Xt, n_cls)


def held_in(k, source):
    """(row idx, {model: probs}, y, groups, cross-fit files used) of outer
    fold k's held-in clips."""
    d = SF[k]
    if source == "iv":
        return d["iv_idx"], {m: d[f"p_iv_{m}"] for m in SINGLES}, d["y_iv"], None, None
    if source == "oof":
        js = [j for j in ALL_FOLDS if j != k]
        return (np.concatenate([SF[j]["test_idx"] for j in js]),
                {m: np.concatenate([SF[j][f"p_te_{m}"] for j in js]) for m in SINGLES},
                np.concatenate([SF[j]["y_te"] for j in js]),
                np.concatenate([np.full(len(SF[j]["y_te"]), j) for j in js]), None)
    idx = y = g = None
    P, used = {}, {}
    for mdir in sorted(p for p in CF.iterdir() if p.is_dir()) if CF.exists() else []:
        # for training fold j, the model that held out both j and k with the
        # most training folds (ho_<held-out folds>.npz)
        held = {f: set(map(int, f.stem.split("_")[1:])) for f in mdir.glob("ho_*.npz")}
        js = [j for j in ALL_FOLDS if j != k]
        files = []
        for j in js:
            cand = [f for f, s in held.items() if {j, k} <= s]
            files.append(min(cand, key=lambda f: (len(held[f]), f.name)) if cand else None)
        if None in files:
            continue
        zs = [(j, np.load(f)) for j, f in zip(js, files)]
        mi = np.concatenate([z[f"idx_{j}"] for j, z in zs])
        if idx is None:
            idx = mi
            y = np.concatenate([z[f"y_{j}"] for j, z in zs])
            g = np.concatenate([np.full(len(z[f"y_{j}"]), j) for j, z in zs])
        assert np.array_equal(mi, idx)
        P[mdir.name] = np.concatenate([z[f"p_{j}"] for j, z in zs])
        used[mdir.name] = sorted({f.stem for f in files})
    return idx, P, y, g, used


def run_fold(k, source, E, y_all):
    d = SF[k]
    hidx, HP, hy, hg, used = held_in(k, source)
    if hidx is None:
        return None
    assert np.array_equal(y_all[hidx], hy)
    if source in ("cf", "oof"):  # every clip of every training fold, never the test fold
        assert np.array_equal(np.sort(hidx), np.sort(np.concatenate([SF[j]["test_idx"] for j in ALL_FOLDS if j != k])))
    hE = E[hidx]
    tE, ty = E[d["test_idx"]], d["y_te"]
    TP = {m: d[f"p_te_{m}"] for m in SINGLES}
    rec = {"fold": int(k), "source": source, "n_heldin": int(len(hy)) if hy is not None else 0,
           "models": sorted(HP), "cf_files": used, "pairs": {}, "three": None}

    def X(kind, Eh, P):
        return out_feats(P) if kind == "out" else Eh if kind == "emb" else np.hstack([Eh, out_feats(P)])

    for L, H in PAIRS:
        if L not in HP or H not in HP:
            continue
        hl, hh, tl, th = HP[L].argmax(1), HP[H].argmax(1), TP[L].argmax(1), TP[H].argmax(1)
        tgt_h = ((hl != hy) & (hh == hy)).astype(int)
        tgt_t = (tl != ty) & (th == ty)
        scores = {"margin": (-margin(HP[L]), -margin(TP[L]))}
        for r in ROUTERS:
            kind = ROUTERS[r][1]
            oof, tp = fit_predict(r, X(kind, hE, HP[L]), tgt_h, hg, X(kind, tE, TP[L]), 2)
            scores[r] = (oof[:, 1], tp[:, 1])
        pr = {"heldin_L_err": float((hl != hy).mean()), "heldin_target_rate": float(tgt_h.mean()),
              "test_L_err": float((tl != ty).mean()), "test_target_rate": float(tgt_t.mean()),
              "oracle": {"f1": f1(ty, np.where(tgt_t, th, tl)), "rate": float(tgt_t.mean()),
                         "bitops": C[L] + float(tgt_t.mean()) * C[H]}, "methods": {}}
        for meth, (sh, st) in scores.items():
            reads_out = meth == "margin" or ROUTERS[meth][1] != "emb"
            cr = 0.0 if meth == "margin" else router_bitops(meth, 1)
            pts = []
            for rate in RATES:
                dfr = st > threshold(sh, rate)
                fr = float(dfr.mean())
                cost = (C[L] + cr + fr * C[H]) if reads_out else (cr + (1 - fr) * C[L] + fr * C[H])
                pts.append({"op": f"rate{rate:g}", "test_rate": fr, "f1": f1(ty, np.where(dfr, th, tl)),
                            "bitops": cost})
            pr["methods"][meth] = {
                "auroc_heldin": float(roc_auc_score(tgt_h, sh)) if 0 < tgt_h.sum() < len(tgt_h) else None,
                "auroc_test": float(roc_auc_score(tgt_t, st)) if 0 < tgt_t.sum() < len(tgt_t) else None,
                "points": pts}
        rec["pairs"][f"{L}->{H}"] = pr
    if all(m in HP for m in THREE):
        rec["three"] = three_way(hE, HP, hy, hg, tE, TP, ty, X)
    return rec


def cheapest_correct(P, y):
    c = np.full(len(y), 3)
    for i in (2, 1, 0):
        c[P[THREE[i]].argmax(1) == y] = i
    return c


def three_way(hE, HP, hy, hg, tE, TP, ty, X):
    hc, tc = cheapest_correct(HP, hy), cheapest_correct(TP, ty)
    tpred = np.stack([TP[m].argmax(1) for m in THREE], 1)
    hpred = np.stack([HP[m].argmax(1) for m in THREE], 1)
    tr, hr = np.arange(len(ty)), np.arange(len(hy))
    cm = np.array([C[m] for m in THREE])
    orc = np.where(tc == 3, 0, tc)
    out = {"heldin_class_share": (np.bincount(hc, minlength=4) / len(hc)).tolist(),
           "test_class_share": (np.bincount(tc, minlength=4) / len(tc)).tolist(),
           "oracle": {"f1": f1(ty, tpred[tr, orc]), "bitops": float(cm[orc].mean()),
                      "share": (np.bincount(orc, minlength=3) / len(orc)).tolist()},
           "methods": {}}
    for r in ROUTERS:
        kind = ROUTERS[r][1]
        oof, tp = fit_predict(r, X(kind, hE, HP["bitnet"]), hc, hg, X(kind, tE, TP["bitnet"]), 4)
        cr = router_bitops(r, 4)
        base, inc = (C["bitnet"] + cr, np.array([0, C["4"], C["esc"]])) if kind != "emb" else (cr, cm)
        cum_h, cum_t = np.cumsum(oof[:, :3], 1), np.cumsum(tp[:, :3], 1)
        ch_all = np.stack([np.argmax(cum_h - lam * inc / C["esc"], 1) for lam in LAMBDAS])
        hcost = base + inc[ch_all].mean(1)
        pts = []
        for B in BUDGETS:
            i = int(np.argmin(np.abs(hcost - B) - 1e-9 * np.arange(len(LAMBDAS))))  # ties: larger lambda
            ch = np.argmax(cum_t - LAMBDAS[i] * inc / C["esc"], 1)
            pts.append({"op": f"budget{B / 1e6:.1f}M", "lambda": float(LAMBDAS[i]),
                        "heldin_bitops": float(hcost[i]), "heldin_f1": f1(hy, hpred[hr, ch_all[i]]),
                        "test_share": (np.bincount(ch, minlength=3) / len(ch)).tolist(),
                        "f1": f1(ty, tpred[tr, ch]), "bitops": float(base + inc[ch].mean())})
        aucs = {}
        for name, lab, sc in (("defer_from_bitnet", tc > 0, 1 - tp[:, 0]),):
            aucs[name] = float(roc_auc_score(lab, sc)) if 0 < lab.sum() < len(lab) else None
        out["methods"][r] = {"auroc_test": aucs, "points": pts}
    # two-stage margin cascade bitnet -> 4 -> esc
    hb, h4, tb, t4 = -margin(HP["bitnet"]), -margin(HP["4"]), -margin(TP["bitnet"]), -margin(TP["4"])
    combos = []
    for r1 in R1:
        t1 = threshold(hb, r1)
        e1 = hb > t1
        for r2 in R2:
            t2 = threshold(h4[e1], r2) if e1.any() else np.inf
            ch = np.where(e1, np.where(h4 > t2, 2, 1), 0)
            combos.append((r1, r2, t1, t2, f1(hy, hpred[hr, ch]), C["bitnet"] + np.mean(np.array([0, C["4"], C["4"] + C["esc"]])[ch])))
    pts = []
    for B in BUDGETS:
        ok = [c for c in combos if c[5] <= B] or [min(combos, key=lambda c: c[5])]
        r1, r2, t1, t2, hf, hcost = max(ok, key=lambda c: (c[4], -c[5]))
        ch = np.where(tb > t1, np.where(t4 > t2, 2, 1), 0)
        pts.append({"op": f"budget{B / 1e6:.1f}M", "r1": r1, "r2": r2, "heldin_bitops": float(hcost),
                    "heldin_f1": hf, "test_share": (np.bincount(ch, minlength=3) / len(ch)).tolist(),
                    "f1": f1(ty, tpred[tr, ch]),
                    "bitops": float(C["bitnet"] + np.mean(np.array([0, C["4"], C["4"] + C["esc"]])[ch]))})
    out["methods"]["margin"] = {"points": pts}
    return out


# ── summary ─────────────────────────────────────────────────────────────────

def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"diff": float(d.mean()), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": len(d), "p": p}


def hull(points):
    """Upper concave hull of baseline points in (mean bit-ops, mean F1),
    cheapest first, cut where F1 stops rising."""
    P = sorted(points, key=lambda q: (q["mean_bitops"], -q["mean_f1"]))
    H = []
    for q in P:
        while len(H) >= 2:
            (a, b) = H[-2], H[-1]
            cross = ((b["mean_bitops"] - a["mean_bitops"]) * (q["mean_f1"] - a["mean_f1"])
                     - (b["mean_f1"] - a["mean_f1"]) * (q["mean_bitops"] - a["mean_bitops"]))
            if cross >= 0:
                H.pop()
            else:
                break
        H.append(q)
    if not H:
        return []
    out = [H[0]]
    for q in H[1:]:
        if q["mean_f1"] > out[-1]["mean_f1"]:
            out.append(q)
    return out


def step_cmp(p, base):
    ok = [q for q in base if q["mean_bitops"] <= p["mean_bitops"] + 1e-6]
    if not ok:
        return None
    b = max(ok, key=lambda q: q["mean_f1"])
    return {"model": b["name"], **paired(p["f1"], b["f1"])}


def mix_cmp(p, H):
    c = p["mean_bitops"]
    if not H or c < H[0]["mean_bitops"] - 1e-6:
        return None
    for a, b in zip(H, H[1:]):
        if a["mean_bitops"] <= c <= b["mean_bitops"]:
            w = (c - a["mean_bitops"]) / (b["mean_bitops"] - a["mean_bitops"])
            f = (1 - w) * np.asarray(a["f1"]) + w * np.asarray(b["f1"])
            return {"model": f'{1 - w:.2f} {a["name"]} + {w:.2f} {b["name"]}', **paired(p["f1"], f)}
    return {"model": H[-1]["name"], **paired(p["f1"], H[-1]["f1"])}


def summarize():
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("rec_*.json"))]
    folds = sorted({r["fold"] for r in recs})
    singles = {m: {"f1": [f1(SF[k]["y_te"], SF[k][f"p_te_{m}"].argmax(1)) for k in folds], "bitops": C[m]}
               for m in SINGLES}
    pts = []  # every operating point with per-fold f1 and bitops
    for src in sorted({r["source"] for r in recs}):
        R = sorted([r for r in recs if r["source"] == src], key=lambda r: r["fold"])
        if [r["fold"] for r in R] != folds:
            continue
        tasks = set.intersection(*[set(r["pairs"]) for r in R]) | ({"3way"} if all(r["three"] for r in R) else set())
        for task in sorted(tasks):
            blocks = [r["three"] if task == "3way" else r["pairs"][task] for r in R]
            for meth in blocks[0]["methods"]:
                for i, op in enumerate(blocks[0]["methods"][meth]["points"]):
                    pts.append({"source": src, "task": task, "method": meth, "op": op["op"],
                                "f1": [b["methods"][meth]["points"][i]["f1"] for b in blocks],
                                "bitops": [b["methods"][meth]["points"][i]["bitops"] for b in blocks],
                                "rate": [b["methods"][meth]["points"][i].get("test_rate") for b in blocks]})
    for p in pts:
        p["mean_f1"], p["mean_bitops"] = float(np.mean(p["f1"])), float(np.mean(p["bitops"]))
    s_pts = [{"name": m, "f1": singles[m]["f1"], "mean_bitops": C[m], "mean_f1": float(np.mean(singles[m]["f1"]))}
             for m in SINGLES]
    # margin baseline: every margin point fixed on a clean held-in source (iv,
    # cf) or on the router's own source
    m_all = [{"name": f'{q["source"]}/{q["task"]}/{q["op"]}', "source": q["source"],
              **{k: q[k] for k in ("f1", "mean_bitops", "mean_f1")}} for q in pts if q["method"] == "margin"]
    m_pool = {src: [q for q in m_all if q["source"] in ("iv", "cf", src)] for src in {p["source"] for p in pts}}
    s_hull = hull(s_pts)
    m_hull = {src: hull(v) for src, v in m_pool.items()}
    # the front a router has to beat: singles and margin points together
    f_hull = {src: hull(s_pts + v) for src, v in m_pool.items()}
    for p in pts:
        # step: best baseline point costing the same or less (the success bar)
        # mix: the baselines' upper concave hull at the same cost, reachable by
        #      sending a random share of clips to each of two baseline points
        #      (per-fold F1 interpolated linearly, an approximation for macro F1)
        p["vs_single"] = step_cmp(p, s_pts)
        p["vs_single_mix"] = mix_cmp(p, s_hull)
        if p["method"] != "margin":
            p["vs_margin"] = step_cmp(p, m_pool[p["source"]])
            p["vs_margin_mix"] = mix_cmp(p, m_hull[p["source"]])
            p["vs_front_mix"] = mix_cmp(p, f_hull[p["source"]])
    lines = [f"# {args.ds}: folds {folds}", "## singles: mean F1 | Mbit-ops"]
    lines += [f"{m:6s} {np.mean(singles[m]['f1']):.4f} | {C[m] / 1e6:.1f}" for m in SINGLES]
    for src in sorted({p["source"] for p in pts}):
        R = sorted([r for r in recs if r["source"] == src], key=lambda r: r["fold"])
        lines.append(f"\n## source {src}: held-in clips per fold {np.mean([r['n_heldin'] for r in R]):.0f}")
        lines.append("pair | held-in L err | test L err | held-in target | test target | oracle F1 @ Mbit-ops | "
                     "AUROC test (held-in) per method")
        for task in [f"{a}->{b}" for a, b in PAIRS]:
            if not all(task in r["pairs"] for r in R):
                continue
            B = [r["pairs"][task] for r in R]
            au = "  ".join(f"{m} {np.mean([b['methods'][m]['auroc_test'] for b in B]):.3f}"
                           f"({np.mean([b['methods'][m]['auroc_heldin'] or np.nan for b in B]):.3f})"
                           for m in B[0]["methods"])
            lines.append(f"{task} | {np.mean([b['heldin_L_err'] for b in B]):.3f} | {np.mean([b['test_L_err'] for b in B]):.3f} | "
                         f"{np.mean([b['heldin_target_rate'] for b in B]):.3f} | {np.mean([b['test_target_rate'] for b in B]):.3f} | "
                         f"{np.mean([b['oracle']['f1'] for b in B]):.4f} @ {np.mean([b['oracle']['bitops'] for b in B]) / 1e6:.1f} | {au}")
        if all(r["three"] for r in R):
            T = [r["three"] for r in R]
            lines.append(f"3way | held-in class share {np.round(np.mean([t['heldin_class_share'] for t in T], 0), 3).tolist()} | "
                         f"test {np.round(np.mean([t['test_class_share'] for t in T], 0), 3).tolist()} | oracle F1 "
                         f"{np.mean([t['oracle']['f1'] for t in T]):.4f} @ {np.mean([t['oracle']['bitops'] for t in T]) / 1e6:.1f} | "
                         "AUROC(bitnet not cheapest-correct) " + "  ".join(
                             f"{m} {np.mean([t['methods'][m]['auroc_test']['defer_from_bitnet'] for t in T]):.3f}"
                             for m in ROUTERS))
    KEYS = ("vs_single", "vs_single_mix", "vs_margin", "vs_margin_mix", "vs_front_mix")

    def cmp_str(v):
        return f"{v['model']} {v['diff']:+.4f} {v['wins']}/{v['n']} p={v['p']:.3f}" if v else "-"

    full = ["source task method op | mean F1 | Mbit-ops | test rate | " + " | ".join(KEYS)]
    for p in pts:
        rt = f"{np.mean(p['rate']):.3f}" if p["rate"][0] is not None else "-"
        full.append(f"{p['source']} {p['task']} {p['method']} {p['op']} | {p['mean_f1']:.4f} | "
                    f"{p['mean_bitops'] / 1e6:.1f} | {rt} | " + " | ".join(cmp_str(p.get(k)) for k in KEYS))
    # front summary: per source, task and method, how many of its held-in-fixed
    # operating points are above each baseline front (mean paired diff > 0 and
    # more fold wins than losses), and its strongest point against that front
    lines.append("\n## front: source task method | for each baseline (best single at <= cost; singles mix hull; "
                 "best margin point at <= cost; margin mix hull; singles plus margin mix hull): ops above / ops, strongest op")
    for src in sorted({p["source"] for p in pts}):
        for task in sorted({p["task"] for p in pts if p["source"] == src}):
            for meth in ["margin"] + list(ROUTERS):
                P = [p for p in pts if p["source"] == src and p["task"] == task and p["method"] == meth]
                if not P:
                    continue
                cols = []
                for key in (KEYS[:2] if meth == "margin" else KEYS):
                    Q = [p for p in P if p.get(key)]
                    good = [p for p in Q if p[key]["diff"] > 0 and p[key]["wins"] > p[key]["losses"]]
                    b = max(Q, key=lambda p: p[key]["diff"]) if Q else None
                    cols.append(f"{len(good)}/{len(P)} " + (f"[{b['op']} {b['mean_f1']:.4f}@{b['mean_bitops'] / 1e6:.1f}M "
                                                            f"{cmp_str(b[key])}]" if b else "[-]"))
                lines.append(f"{src} {task} {meth} | " + " | ".join(cols))
    (OUT / "summary_points.txt").write_text("\n".join(full) + "\n")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps({"folds": folds, "costs": C, "singles": singles, "points": pts}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    if not args.summarize:
        df = pd.read_csv(args.csv)
        y_all = df[args.label_col].to_numpy()
        E = df.drop(columns=["folder", "name", "label", "category", "class_id"], errors="ignore").to_numpy(np.float32)
        assert E.shape[1] == 1536, E.shape
        for k in ALL_FOLDS:
            d = SF[k]
            assert np.array_equal(y_all[d["test_idx"]], d["y_te"]) and np.array_equal(y_all[d["iv_idx"]], d["y_iv"])
        for source in args.sources:
            for k in (args.folds or ALL_FOLDS):
                t0 = time.time()
                rec = run_fold(k, source, E, y_all)
                if rec is None:
                    print(source, k, "no held-in data yet", flush=True)
                    continue
                rec["seconds"] = time.time() - t0
                (OUT / f"rec_{source}_fold{k}.json").write_text(json.dumps(rec, indent=1))
                print(source, k, f"{rec['seconds']:.0f}s", "models", rec["models"], flush=True)
    summarize()
