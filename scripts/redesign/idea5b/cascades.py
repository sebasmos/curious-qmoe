"""Idea 5b, step 1: every ordered cascade A -> B (bit-ops of A below B's) from the Stage 2 head
scores, plus every static system, per (seed, fold).

Cascade: the same head on both passes (FP32 head `esc` or 4-bit head `4`); gate = softmax margin p1 - p2 of A; a clip escalates when its
margin is below a threshold set on the fold's inner validation split under the same condition
(never on test). Points: inner-validation escalation 1, 2, 3, 5, 10, 20, 30% (closest tie-free
rate), and `match` (smallest escalation whose inner-validation F1 reaches B's; else the best-F1
cut). Rules on escalated clips: `B` (B's prediction) and `avg` (argmax of the mean of A's and B's
probabilities); `match` is searched separately for each rule.

Writes OUT/<dataset>/: static_pairs.csv, static_preds.npz (argmax per system for the hull),
cascade_pairs.csv.gz, meta.json.

    python cascades.py --stage2 STAGE2_DIR --dataset esc50 --out OUT
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (BUDGETS, CELLS, CORE, POINTS, Run, bitops, cut_points, f1_curve, f1_fast, margin,
                    pick_match, pick_rate)

ap = argparse.ArgumentParser()
ap.add_argument("--stage2", type=Path, required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--exclude", nargs="*", default=["20db"],
                help="skip conditions containing these strings (20 dB was still being extracted and scored by "
                     "Stage 2 on 22 Oct; fixing the list keeps the family the same on a re-run)")
a = ap.parse_args()

sc = a.stage2 / "scores" / a.dataset
tags = sorted(p.name for p in sc.iterdir() if p.is_dir() and "headw4" not in p.name)
seeds = [int(t.split("_s")[1]) for t in tags]
heads = ["esc", "4"]
r0 = Run(sc / tags[0] / "esc")
files = sorted(p.stem for p in r0.d.glob("*.npz"))
configs = sorted({f.split("__")[0] for f in files})
conds = sorted({f.split("__", 1)[1] for f in files})
conds = [c for c in conds if not any(x in c for x in a.exclude) and all(r0.complete(cfg, c) for cfg in CORE)]
if all(r0.complete(cfg, "field") for cfg in CORE):
    conds.append("field")
n_cls = r0.raw("fp32", "clean")["p"].shape[1]
pairs = [(x, y) for x in CORE for y in CORE if bitops(x) < bitops(y)]
print(a.dataset, "seeds", seeds, "conditions", conds, "pairs", len(pairs), flush=True)

out = a.out / a.dataset
out.mkdir(parents=True, exist_ok=True)
srows, crows, preds = [], [], {}
y_all = tf_all = None
t0 = time.time()
for seed, tag in zip(seeds, tags):
    runs = {h: Run(sc / tag / h) for h in heads}
    for cond in conds:
        # static systems: every config scored in full for this condition, both heads
        for h, run in runs.items():
            for cfg in configs:
                if not run.complete(cfg, cond):
                    continue
                r = run.get(cfg, cond)
                if y_all is None:
                    y_all, tf_all = r["y"], r["tf"]
                assert (r["y"] == y_all).all() and (r["tf"] == tf_all).all()
                pr = r["p"].argmax(1)
                preds[f"{seed}|{h}|{cfg}|{cond}"] = pr.astype(np.int16)
                for fold in sorted(r["iv"]):
                    m = r["tf"] == fold
                    srows.append(dict(cond=cond, config=cfg, head=h, seed=seed, fold=fold,
                                      f1=f1_fast(r["y"][m], pr[m], n_cls), acc=float((pr[m] == r["y"][m]).mean())))
        # cascades, the same head on both passes
        for head, run in runs.items():
            for A in CORE:
                rA = run.get(A, cond)
                for fold in sorted(rA["iv"]):
                    idx, pAi = rA["iv"][fold]
                    te = np.flatnonzero(rA["tf"] == fold)
                    assert len(np.intersect1d(te, idx)) == 0
                    yi, yt = rA["y"][idx], rA["y"][te]
                    mi = margin(pAi)
                    order = np.argsort(mi, kind="stable")
                    ms = mi[order]
                    js, ts = cut_points(ms)
                    n_iv = len(mi)
                    rate_cut = {pt: pick_rate(js, n_iv, b) for pt, b in zip(POINTS[:-1], BUDGETS)}
                    pAt = rA["p"][te]
                    mt = margin(pAt)
                    aT = pAt.argmax(1)
                    a_ok = aT == yt
                    for A_, B in pairs:
                        if A_ != A:
                            continue
                        rB = run.get(B, cond)
                        idx2, pBi = rB["iv"][fold]
                        assert (idx2 == idx).all()
                        pBt = rB["p"][te]
                        bT, vT = pBt.argmax(1), (pAt + pBt).argmax(1)
                        bI, vI = pBi.argmax(1), (pAi + pBi).argmax(1)
                        target = f1_fast(yi, bI, n_cls)
                        for rule, finI, finT in (("B", bI, bT), ("avg", vI, vT)):
                            cur = f1_curve(yi[order], pAi.argmax(1)[order], finI[order], n_cls)
                            cuts = dict(rate_cut)
                            ci, reached = pick_match(js, cur, target)
                            cuts["match"] = ci
                            for pt in POINTS:
                                t = ts[cuts[pt]]
                                e = mt < t
                                pred = np.where(e, finT, aT)
                                ok = pred == yt
                                crows.append(dict(cond=cond, head=head, A=A, B=B, rule=rule, point=pt, seed=seed, fold=fold,
                                                  f1=f1_fast(yt, pred, n_cls), acc=float(ok.mean()), rate=float(e.mean()),
                                                  iv_rate=js[cuts[pt]] / n_iv, iv_f1=float(cur[js[cuts[pt]]]),
                                                  iv_target=target, thr=t,
                                                  match_reached=reached if pt == "match" else np.nan,
                                                  n_esc=int(e.sum()), n_fixed=int((e & ~a_ok & ok).sum()),
                                                  n_broken=int((e & a_ok & ~ok).sum()),
                                                  n_a_wrong_b_right=int((~a_ok & (bT == yt)).sum())))
        print(f"  seed {seed} {cond} done ({time.time() - t0:.0f}s)", flush=True)
    del runs

S, C = pd.DataFrame(srows), pd.DataFrame(crows)
S.to_csv(out / "static_pairs.csv", index=False)
C.to_csv(out / "cascade_pairs.csv.gz", index=False)
np.savez_compressed(out / "static_preds.npz", y=y_all, tf=tf_all, **preds)
json.dump(dict(dataset=a.dataset, tags=tags, seeds=seeds, conds=conds, configs=configs, n_cls=n_cls,
               pairs=pairs, n_pairs=len(pairs), cells=CELLS), open(out / "meta.json", "w"), indent=1)
print("rows", len(S), len(C), f"{time.time() - t0:.0f}s")
