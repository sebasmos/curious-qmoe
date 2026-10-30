"""T1c specialization test (ANALYSIS_PLAN_T1C.md section 6) on the conditions whose gate passed (strength_verdict.json), plus the clean condition as a descriptive control.
Reads test labels. python t1c_t1.py --run DIR"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1c_common import *
from common import write_json
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); a = ap.parse_args()
ARMS_B = ["U", "M0", "R05", "R10", "R20", "F", "T"]; ARMS_A = ["M0A", "R10A"]; PAIRS = [(x, "U") for x in ARMS_B[1:]] + [(x, "UA") for x in ARMS_A]; PRIMARY = ("R10", "U")
rng = np.random.default_rng(0)
all_tested = [c for b in CONDITIONS for c in CONDITIONS[b]]; gate = {}
for bed in BEDS: gate.update(json.load(open(a.run / bed / "strength_verdict.json"))["gate_pass"])
rows, ratios = [], {}
for bed in BEDS:
    D = load_bed(bed); y = D["y"]
    todo = [c for c in CONDITIONS[bed] if gate[c]] + ["clean"]
    for f in sorted(glob.glob(str(a.run / bed / "raw_arms/s*_f*.npz"))):
        A = np.load(f); stem = Path(f).stem; S = np.load(a.run / bed / f"raw_strength/{stem}.npz"); te, tr = S["te"], S["tr"]; yte = y[te]; cond = S["cond"] if bed == "tau" else None; tau1 = float(S["tau1"])
        for cn in todo:
            g = groups(bed, cn, te, D["assign"], cond); yr = gather_y(yte, g)
            z1 = gather({c: S[f"z1|{c}"] for c, _ in g}, g).astype(np.float64); fl = pmargin(softmax(z1)) < tau1
            ratios[(bed, cn, int(S["seed"]), int(S["fold"]))] = (len(g[0][1]) if bed == "tau" else len(te)) / len(tr)
            row = dict(bed=bed, cond=cn, seed=int(S["seed"]), fold=int(S["fold"]), n_flag=int(fl.sum()), flag_share=fl.mean(), E1=(z1.argmax(1) == yr)[fl].mean() * 100)
            for arm in ARMS_B + ARMS_A:
                z = gather({c: A[f"{arm}|{c}"] for c, _ in g}, g); ok = z.argmax(1) == yr; row[arm] = ok[fl].mean() * 100; row[arm + "_all"] = ok.mean() * 100
            zu = gather({c: S[f"UA_E2a|{c}"] for c, _ in g}, g); okua = zu.argmax(1) == yr; row["UA"] = okua[fl].mean() * 100; row["UA_all"] = okua.mean() * 100
            rows.append(row)
Dd = pd.DataFrame(rows); (a.run / "t1").mkdir(exist_ok=True); Dd.to_csv(a.run / "t1/t1c_arm_metrics_per_seed_fold.csv", index=False)
Fd = Dd.groupby(["bed", "cond", "fold"]).mean(numeric_only=True).drop(columns="seed").reset_index(); Fd.to_csv(a.run / "t1/t1c_arm_metrics_per_fold.csv", index=False)
def nb(d, ratio):
    K = len(d); s2 = d.var(ddof=1); se = np.sqrt((1 / K + ratio) * s2); t = d.mean() / se if se > 0 else 0.0; p = float(2 * stats.t.sf(abs(t), K - 1)); h = stats.t.ppf(.975, K - 1) * se
    bs = np.sort(d[rng.integers(0, K, (10000, K))].mean(1))
    return dict(mean=float(d.mean()), ci_lo=float(d.mean() - h), ci_hi=float(d.mean() + h), p=p, boot_lo=float(bs[249]), boot_hi=float(bs[9749]), dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0)
res = []
for (bed, cn), g in Fd.groupby(["bed", "cond"]):
    K = len(g); ratio = float(np.mean([v for k, v in ratios.items() if k[0] == bed and k[1] == cn])); maj = K // 2 + 1; old = 7 if K == 10 else 4
    for arm, tw in PAIRS:
        d = (g[arm] - g[tw]).values; r = nb(d, ratio)
        res.append(dict(bed=bed, cond=cn, arm=arm, twin=tw, primary=(arm, tw) == PRIMARY, K=K, acc_arm=g[arm].mean(), acc_twin=g[tw].mean(), acc_E1=g.E1.mean(), contrast=r["mean"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_raw=r["p"],
                        boot_lo=r["boot_lo"], boot_hi=r["boot_hi"], dz=r["dz"], folds_ge_0_5=int((d >= .5).sum()), folds_ge_1=int((d >= 1).sum()), majority_needed=maj, old_count_needed=old, fold_diffs=[round(float(v), 2) for v in d],
                        all_rows_contrast=float((g[arm + "_all"] - g[tw + "_all"]).mean())))
C = pd.DataFrame(res)
# Holm within the secondary arms of each condition
C["p_holm_secondary"] = np.nan
for (bed, cn), g in C[~C.primary].groupby(["bed", "cond"]):
    o = g.sort_values("p_raw"); run, m = 0.0, len(o)
    for i, (ix, r) in enumerate(o.iterrows()): run = max(run, min(1.0, (m - i) * r.p_raw)); C.loc[ix, "p_holm_secondary"] = run
# Holm over the five T1c conditions, primary contrast; conditions that failed the gate carry p = 1
P = {c: 1.0 for c in all_tested}
for _, r in C[C.primary & C.cond.isin(all_tested)].iterrows(): P[r.cond] = r.p_raw
order = sorted(P, key=lambda c: P[c]); holm, run = {}, 0.0
for i, c in enumerate(order): run = max(run, min(1.0, (len(P) - i) * P[c])); holm[c] = run
C.to_csv(a.run / "t1/t1c_contrasts.csv", index=False)
V = {"family_conditions": all_tested, "gate_pass": gate, "p_primary_raw": P, "p_primary_holm5": holm, "conditions": {}}
for c in all_tested:
    if not gate[c]: V["conditions"][c] = dict(verdict="NOT RUN (gate failed)", p_raw=1.0, p_holm=1.0); continue
    r = C[C.primary & (C.cond == c)].iloc[0]
    v = "PASS" if (r.contrast >= 1.0 and r.folds_ge_1 >= r.majority_needed and holm[c] < 0.05) else ("STOP" if r.contrast < 0.3 else "INCONCLUSIVE")
    V["conditions"][c] = dict(verdict=v, contrast=float(r.contrast), ci=[float(r.ci_lo), float(r.ci_hi)], p_raw=float(r.p_raw), p_holm=float(holm[c]), folds_ge_1=int(r.folds_ge_1), folds_ge_0_5=int(r.folds_ge_0_5), majority_needed=int(r.majority_needed),
                              acc_R10=float(r.acc_arm), acc_U=float(r.acc_twin), acc_E1=float(r.acc_E1), fold_diffs=r.fold_diffs)
write_json(a.run / "t1/t1c_verdict.json", V)
print(C[C.primary][["bed", "cond", "acc_arm", "acc_twin", "acc_E1", "contrast", "ci_lo", "ci_hi", "p_raw", "folds_ge_1"]].round(3).to_string()); print(json.dumps({c: V["conditions"][c]["verdict"] for c in all_tested}))
