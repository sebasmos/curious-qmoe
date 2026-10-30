"""T14 step 1 (ANALYSIS_PLAN_T14.md sections 2, 3): per (task, cell, seed, fold) macro-F1 of the margin-routed pair and the always-average of the same two experts,
recomputed from the stored per-clip logits, plus the reproduction gate against the stored stage 3c cascades.csv.gz / static.csv. Reads test labels (run after the plan,
the loaders and the input check were committed). python t14_pair.py --out DIR"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t13_common import *   # Q, S2, S3B, S3C, I5, OPTB, FINAL, A, NOISE, CORR, TASKS, tau_load, tau_rows
BUD = A.BUDGETS
# task -> pair spec: cheap (dir builder args), strong, budget
PAIR = {
    "esc50": dict(cheap=("mn20", "4", "mn20_w8a8"), ast=("short", "w4g128", "esc"), bud="b05", cn="cnn:mn20:w8a8:4", an="ast:w4g128@512+esc", casc=("eff" if False else "mn20", "w8a8", "4", "w4g128@512")),
    "esc5": dict(cheap=("mn20", "esc", "mn20_w8a8"), ast=("full", "w6", "esc"), bud="b20", cn="cnn:mn20:w8a8:esc", an="ast:w6+esc", casc=("mn20", "w8a8", "esc", "w6")),
    "urban": dict(cheap=("mn10", "esc", "mn10_w8a8"), ast=("full", "fp32", "esc"), bud="b50", cn="cnn:mn10:w8a8:esc", an="ast:fp32+esc", casc=("mn10", "w8a8", "esc", "fp32")),
}
_SD = {}
def sd(d):
    if d not in _SD: _SD[d] = A.ScoreDir(d)
    return _SD[d]
def cheap_dir(task, seed, corrupt):
    m, head, cfg = PAIR[task]["cheap"]
    root = FINAL / "scores" if corrupt else (S3C / "scores_cnn" if m == "mn20" else S3B / "scores_cnn")
    return sd(root / task / f"{m}_{task}_s{seed}" / head), cfg
def ast_dir(task, seed, corrupt):
    kind, cfg, head = PAIR[task]["ast"]
    if kind == "short": d = (FINAL / "scores" if corrupt else S2 / "scores_ast_short") / task / f"astshort_{task}_s{seed}" / head
    else: d = I5 / "scores" / task / f"ast{task}_s{seed}" / head
    return sd(d), cfg
def cells_available(task):
    cond = ["clean"] + NOISE + ["field"] + CORR
    out = []
    for c in cond:
        cor = c in CORR
        ok = True
        for s in TASKS[task]["seeds"]:
            for dd, cfg in (cheap_dir(task, s, cor), ast_dir(task, s, cor)):
                ok = ok and dd.d.exists() and dd.has(cfg, c) and (c == "field" or dd.complete(cfg, c))
        if ok: out.append(c)
    return out
def pair_rows(task, cells):
    T = TASKS[task]; K = T["K"]; rate = BUD[PAIR[task]["bud"]]; rows = []
    for s in T["seeds"]:
        dc0, cfgc = cheap_dir(task, s, False); r0 = dc0.get(cfgc, "clean"); folds = sorted(r0["iv"])
        thr = {f: A.thr_rate(A.margin(A.softmax(r0["iv"][f][1])), rate) for f in folds}
        for c in cells:
            cor = c in CORR
            dc, cfgc = cheap_dir(task, s, cor); da, cfga = ast_dir(task, s, cor)
            rc, ra = dc.get(cfgc, c), da.get(cfga, c)
            assert (rc["y"] == ra["y"]).all() and (rc["fold"] == ra["fold"]).all()
            pc, pa = A.softmax(rc["z"]), A.softmax(ra["z"])
            for f in folds:
                te = np.flatnonzero(rc["fold"] == f); y = rc["y"][te]
                m = A.margin(pc[te]); esc = m < thr[f]
                avg = ((pc[te] + pa[te]) / 2).argmax(1); ch = pc[te].argmax(1); st = pa[te].argmax(1)
                rows.append(dict(task=task, cell=c, seed=s, fold=f, rate=float(esc.mean()),
                                 pair=100 * A.f1_fast(y, np.where(esc, avg, ch), K), avg=100 * A.f1_fast(y, avg, K),
                                 cheap=100 * A.f1_fast(y, ch, K), strong=100 * A.f1_fast(y, st, K)))
    return pd.DataFrame(rows)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--tasks", nargs="+", default=list(PAIR)); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    allr, gate = [], []
    for task in a.tasks:
        t0 = time.time(); cells = cells_available(task); print(task, "cells", len(cells), cells, flush=True)
        df = pair_rows(task, cells); allr.append(df)
        # reproduction gate against the stored cascade rows and static rows (clean and field)
        C = pd.read_csv(S3C / f"analysis/{task}/cascades.csv.gz"); S = pd.read_csv(S3C / f"analysis/{task}/static.csv"); P = PAIR[task]; m, bb, k, tier = P["casc"]
        for c in ("clean", "field"):
            g = C[(C.cheap_src == m) & (C.cheap_bb == bb) & (C.cheap.astype(str) == k) & (C.expensive == tier) & (C.rule == "avg") & (C.budget == P["bud"]) & (C.condition == c)]
            mine = df[df.cell == c].set_index(["seed", "fold"]).pair; ref = g.set_index(["seed", "fold"]).f1 * 100
            gate.append(dict(task=task, cell=c, what="pair", max_abs_diff=float((mine - ref.reindex(mine.index)).abs().max()), n=len(mine)))
            for col, sysn in (("cheap", P["cn"]), ("strong", P["an"])):
                s_ = S[(S.system == sysn) & (S.condition == c)].set_index(["seed", "fold"]).f1 * 100
                mm = df[df.cell == c].set_index(["seed", "fold"])[col]
                gate.append(dict(task=task, cell=c, what=col, max_abs_diff=float((mm - s_.reindex(mm.index)).abs().max()), n=len(mm)))
        print(task, "%.0fs" % (time.time() - t0), flush=True)
    pd.concat(allr).to_csv(a.out / "pair_cell_fold.csv", index=False); G = pd.DataFrame(gate); G.to_csv(a.out / "reproduction_gate.csv", index=False); print(G.to_string())
    assert (G.max_abs_diff < 1e-7).all(), "reproduction gate failed"
