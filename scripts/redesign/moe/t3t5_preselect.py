"""T3/T5 step 2 (ANALYSIS_PLAN_T3T5.md section 4): the best single model per task, chosen on INNER-VALIDATION accuracy of the same condition family (the task's shifted pool; TAU: clean inner
validation, the only one stored), among the experts with priced cost at or below the pair's cost and stored in every pool cell. Reads inner-validation labels and logits only, no test clip.
python t3t5_preselect.py --out DIR"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t3t5_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
pre = json.load(open(T13RUN / "inputs/t13_preselected.json")); out = {}
for task in T35:
    CT = cost_table(task); pair = pre[task]["pair"].split("+"); cost = sum(CT[e] for e in pair)
    assert abs(cost - pre[task]["cost"]) < 1e-3 * cost, (task, cost, pre[task]["cost"])
    pool = POOL[task] if task != "tau" else ["clean"]; res = {}
    for e in experts_of(task):
        if CT[e] > cost * (1 + 1e-12) or not all(available(task, e, c) for c in pool): continue
        accs = []
        for c in pool:
            for s in TASKS[task]["seeds"]:
                D = cellload(task, e, c, s)
                for f, (idx, z) in D["iv"].items(): accs.append(100 * (z.argmax(1) == D["y"][idx]).mean())
        res[e] = float(np.mean(accs))
    best = max(res, key=lambda e: (res[e], -CT[e]))
    out[task] = dict(pair=pre[task]["pair"], cheap=CHEAP, strong=PAIR[task], pair_cost=cost, pool_cells=pool, candidates_inner_val_acc=res, best_single=best, best_single_cost=CT[best],
                     note="TAU: clean inner validation (device A only), the shifted family has no inner validation" if task == "tau" else "inner validation of the shifted pool cells")
    print(task, out[task]["pair"], {k: round(v, 2) for k, v in res.items()}, "->", best, flush=True)
json.dump(out, open(a.out / "t3t5_preselected.json", "w"), indent=1)
