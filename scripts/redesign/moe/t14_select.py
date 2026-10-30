"""T14 amendment 2 (ANALYSIS_PLAN_T14.md section 9): best single model at equal cost chosen on the inner-validation F1 of the same condition family (clean, field, and the mean over
the non-clean cells of the shift set). Inner-validation F1 is recomputed from the stored inner-validation logits and labels; no test row is used. python t14_select.py --out DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t14_pair import *   # PAIR, cells_available, sd, S3B, S3C, S2, I5, FINAL, A, NOISE, CORR, TASKS
from t14_primary_data import static, FMAP
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
CNN_NEW = {"mn05", "mn20", "dymn04"}
def sysdir(system, task, seed, corrupt):
    SL = TASKS[task]["short"]; p = system.split(":")
    if p[0] == "eff": return sd(S2 / "scores" / task / f"effp_{task}_s{seed}" / p[2]), p[1]
    if p[0] == "cnn":
        _, m, bb, k = p
        root = FINAL / "scores" if corrupt else ((S3C if m in CNN_NEW else S3B) / "scores_cnn")
        if corrupt and m not in ("mn10", "mn20"): raise KeyError(system)
        return sd(root / task / f"{m}_{task}_s{seed}" / k), f"{m}_{bb}"
    n, h = system[4:].split("+")
    if "@" in n:
        base, L = n.split("@")
        if not L.isdigit() or int(L) != SL: raise KeyError(system)
        root = FINAL / "scores" if corrupt else S2 / "scores_ast_short"
        if corrupt and base != "w4g128": raise KeyError(system)
        return sd(root / task / f"astshort_{task}_s{seed}" / h), base
    if corrupt: raise KeyError(system)
    return sd(I5 / "scores" / task / f"ast{task}_s{seed}" / h), n
def iv_f1(task, system, cell):
    K = TASKS[task]["K"]; v = []
    for s in TASKS[task]["seeds"]:
        d, cfg = sysdir(system, task, s, cell in CORR); r = d.get(cfg, cell)
        for f, (idx, z) in r["iv"].items(): v.append(100 * A.f1_fast(r["y"][idx], z.argmax(1), K))
    return float(np.mean(v))
out, rows = {}, []
for task in PAIR:
    cells = cells_available(task); S_ = static(task); have = {s: set(g.cell) for s, g in S_.groupby("system")}
    iv = pd.read_csv(S3C / f"analysis/{task}/static_iv.csv"); iv = iv[iv.set.isin(["stage3b", "new"])].set_index("system"); limit = json.load(open(S3C / f"analysis/{task}/picks.json"))["P3"]["cost_iv"]
    def pool(cs): return [s for s in have if set(cs) <= have[s] and s in iv.index]   # no cost filter: every mappable candidate is scored; consumers filter by cost
    fams = {"clean": ["clean"], "field": ["field"], "shift": [c for c in cells if c != "clean"]}
    sets = {"main": cells}
    if task == "esc50": sets["additive14"] = [c for c in cells if c not in CORR]
    res = {}
    for sname, cs in sets.items():
        cand, excluded = [], []
        for s in pool(cs):
            try: sysdir(s, task, TASKS[task]["seeds"][0], any(c in CORR for c in cs)); cand.append(s)
            except (KeyError, ValueError): excluded.append(s)
        print(task, sname, len(cand), "candidates,", len(excluded), "excluded", flush=True)
        for fam, fc in fams.items():
            fc = [c for c in fc if c in cs]
            sc = {}
            for s in cand:
                try: sc[s] = float(np.mean([iv_f1(task, s, c) for c in fc]))
                except Exception as e: excluded.append(f"{s} ({type(e).__name__})")
            lim_ok = {t: v for t, v in sc.items() if float(iv.loc[t, "cost"]) <= limit * (1 + 1e-12)}   # the pair's own comparator: cost at most cost_iv of the P3 pick
            top = sorted(lim_ok, key=lambda s: (-lim_ok[s], s))[:5]
            res[f"{sname}:{fam}"] = dict(best=top[0], top5=[(t, lim_ok[t]) for t in top], n_scored=len(sc), n_within_limit=len(lim_ok), cells=fc, all_scores={t: [sc[t], float(iv.loc[t, "cost"])] for t in sc})
            rows.append(dict(task=task, set=sname, family=fam, best=top[0], iv_f1=lim_ok[top[0]], n_scored=len(sc), n_within_limit=len(lim_ok), cost=float(iv.loc[top[0], "cost"])))
        res[f"{sname}:excluded"] = sorted(set(excluded))
    out[task] = res
json.dump(out, open(a.out / "best_single_v2.json", "w"), indent=1); pd.DataFrame(rows).to_csv(a.out / "best_single_v2.csv", index=False); print(pd.DataFrame(rows).to_string())
