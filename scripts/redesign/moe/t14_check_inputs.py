"""T14 input check and best-single selection (ANALYSIS_PLAN_T14.md section 2). Label-free: it lists which cells are complete, reads costs and clean inner-validation F1
(static_iv.csv) and chooses the best single at equal cost. It reads the condition column of static.csv only, never an F1 value. python t14_check_inputs.py --out DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t14_pair import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
FMAP = {"mn20:esc": "cnn:mn20:w8a8:esc", "mn20:8": "cnn:mn20:w8a8:8", "mn20:4": "cnn:mn20:w8a8:4", "mn20:bitnet": "cnn:mn20:w8a8:bitnet",
        "mn10:esc": "cnn:mn10:w8a8:esc", "mn10:8": "cnn:mn10:w8a8:8", "mn10:4": "cnn:mn10:w8a8:4", "mn10:bitnet": "cnn:mn10:w8a8:bitnet",
        "ast:esc": "ast:w4g128@512+esc", "ast:4": "ast:w4g128@512+4"}
out = {}; avail = []
for task in PAIR:
    cells = cells_available(task)
    for c in ["clean"] + NOISE + ["field"] + CORR: avail.append(dict(task=task, cell=c, pair_available=c in cells))
    S = pd.read_csv(S3C / f"analysis/{task}/static.csv", usecols=["system", "condition", "cost", "set"])
    S = S[S.set.isin(["stage3b", "new"])]
    cs = S.groupby("system").agg(cost=("cost", "mean"), conds=("condition", lambda x: frozenset(x))).reset_index()
    have = {r.system: set(r.conds) for r in cs.itertuples()}
    if task == "esc50":
        F = pd.read_csv(FINAL / "analysis/b_esc50/static.csv", usecols=["system", "condition"])
        for k, v in F.groupby("system").condition.unique().items():
            if k in FMAP and FMAP[k] in have: have[FMAP[k]] = have[FMAP[k]] | set(v)
    iv = pd.read_csv(S3C / f"analysis/{task}/static_iv.csv"); iv = iv[iv.set.isin(["stage3b", "new"])].set_index("system")
    pk = json.load(open(S3C / f"analysis/{task}/picks.json"))["P3"]; cost_iv = pk["cost_iv"]
    def best(cells_, limit):
        cand = [s for s in have if set(cells_) <= have[s] and s in iv.index and float(iv.loc[s, "cost"]) <= limit * (1 + 1e-12)]
        cand.sort(key=lambda s: (-float(iv.loc[s, "f1"]), s)); return cand[0], cand[:5], len(cand)
    b, top, n = best(cells, cost_iv)
    d = dict(cells=cells, cost_iv=cost_iv, best_single=b, best_top5=top, n_candidates=n, best_single_iv_f1=float(iv.loc[b, "f1"]), best_single_cost=float(iv.loc[b, "cost"]))
    if task == "esc50":
        c14 = [c for c in cells if c not in CORR]; b14, top14, n14 = best(c14, cost_iv)
        d.update(cells_additive14=c14, best_single_additive14=b14, best_top5_additive14=top14, n_candidates_additive14=n14)
    out[task] = d; print(task, json.dumps({k: v for k, v in d.items() if k != "cells"}, default=str), len(cells), "cells", flush=True)
json.dump(out, open(a.out / "best_single.json", "w"), indent=1); pd.DataFrame(avail).to_csv(a.out / "availability.csv", index=False)
