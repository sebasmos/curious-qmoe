"""T14 singles table (amendment 1, row Sg): the stability descriptors of every stored single model of stage 3c (sets stage3b and new) and the final-round non-additive cells, with
the variance ratio to the best single of cost at most its own (inner-validation selection of the pool of best_single_v2.json; singles above the pair's cost limit get no ratio).
python t14_singles.py --r RESULTS_DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t14_pair import PAIR, TASKS, CORR
from t14_primary_data import static, single
import t14_eval as E, t14_stats as S
ap = argparse.ArgumentParser(); ap.add_argument("--r", required=True, type=Path); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args(); S.B = a.boot; R = a.r
BV = json.load(open(R / "inputs/best_single_v2.json")); rows = []
for task in PAIR:
    S_ = static(task); cost = S_.groupby("system").cost.mean(); key = "main"
    for sysn, g in S_.groupby("system"):
        cells = set(g.cell)
        if not {"clean", "field"} <= cells: continue
        s_ = g[["cell", "seed", "fold", "f1"]]; row = dict(task=task, system=sysn, protocol=3, cost=float(cost[sysn]), n_cells=len(cells))
        for cond in ("clean", "field"):
            d = E.descr(s_, cond); row.update({f"{cond}_mean": d["mean"], f"{cond}_sd_fold": d["sd_fold"], f"{cond}_sd_seed": d["sd_seed"], f"{cond}_sd_inter": d["sd_inter"]})
            k = "additive14" if (task == "esc50" and not any(c in CORR for c in cells)) else "main"; allsc = BV[task][f"{k}:{cond}"]["all_scores"]
            ok = [s for s, (v, c) in allsc.items() if c <= cost[sysn] * (1 + 1e-9) and s != sysn]
            if ok:
                nm = sorted(ok, key=lambda s: (-allsc[s][0], s))[0]; sm = E.sampling(s_, single(task, nm), cond)
                if sm: row.update({f"{cond}_ref": nm, f"{cond}_ratio": sm["ratio_fold"], f"{cond}_ratio_lo": sm["ratio_fold_lo"], f"{cond}_ratio_hi": sm["ratio_fold_hi"], f"{cond}_label": sm["sampling_label"]})
        if len(cells) >= 3 and "clean" in cells:
            cs = ["clean"] + sorted(c for c in cells if c != "clean"); m = S.shift_measures(E.cellmat(s_, cs), cs); row.update(shift_cells=len(cs), worst=float(m["worst"].mean()), drop=float(m["drop"].mean()), sd_cells=float(m["sd_cells"].mean()), slope_per_db=float(np.nanmean(m["slope_per_db"])) if np.isfinite(m["slope_per_db"]).any() else np.nan)
        rows.append(row)
pd.DataFrame(rows).to_csv(R / "tables/stability_singles.csv", index=False); print(len(rows), "single rows")
