"""T2b step 3: score the fitted gates on the test folds and apply the pre-registered rule (ANALYSIS_PLAN_T2B.md sections 5 and 6).
The only script that reads test labels. python t2b_score.py --run DIR"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_table, f1_macro, write_json
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); a = ap.parse_args()
EXP = ["E1", "E2c", "E2b", "E2a"]; ARMS = ["A1", "A2h", "A2f", "A3h", "A3f", "A5"]; GATES = ["A1", "A2h", "A2f", "A3h", "A3f"]
y_all = load_table().label.values.astype(np.int64)
rows, ntr_ratio = [], {}
for f in sorted(glob.glob(str(a.run / "urban/fit/s*_f*.npz"))):
    F = np.load(f); stem = Path(f).stem; R = np.load(a.run / f"urban/raw/{stem}.npz"); te, tr, flag = R["te"], R["tr"], R["flag"]; yt = y_all[te]
    seed, fold = int(R["seed"]), int(R["fold"]); ntr_ratio[(seed, fold)] = len(te) / len(tr); COST = F["cost"]; kb = int(F["kb"])
    P = {e: np.exp(R[k] - R[k].max(1, keepdims=True)) for e, k in zip(EXP, ["z1_te", "E2c_te", "E2b_te", "E2a_te"])}
    P = {e: v / v.sum(1, keepdims=True) for e, v in P.items()}
    out = {e: (P[e], np.full(len(te), COST[i])) for i, e in enumerate(EXP)}
    out["BEST"] = (P[EXP[kb]], np.full(len(te), COST[kb]))
    out["A4"] = (F["A4_p"], F["A4_cost"]); out["AVG2"] = (F["AVG2_p"], np.full(len(te), COST[0] + COST[3]))
    for k in ARMS: out[k] = (F[f"{k}_p"], F[f"{k}_cost"])
    out["A3h_all"] = (F["A3h_p_all"], F["A3h_cost_all"])
    correct = np.stack([P[e].argmax(1) == yt for e in EXP], 1); anyc = correct.any(1)
    first = np.where(anyc, correct.argmax(1), 0); orc_cost = np.where(anyc, COST[first], COST[0]) + 0 * COST[0]
    for k, (p, c) in out.items():
        pr = p.argmax(1); ok = pr == yt
        rows.append(dict(seed=seed, fold=fold, arm=k, acc=ok.mean() * 100, acc_flag=ok[flag].mean() * 100, f1=f1_macro(yt, pr, 10) * 100, cost=c.mean(), n_flag=int(flag.sum())))
    for nm, v, c in [("ORACLE4", anyc, orc_cost), ("ORACLE_E1E2a", correct[:, 0] | correct[:, 3], np.where(correct[:, 0], COST[0], COST[3]))]:
        rows.append(dict(seed=seed, fold=fold, arm=nm, acc=v.mean() * 100, acc_flag=v[flag].mean() * 100, f1=np.nan, cost=c.mean(), n_flag=int(flag.sum())))
    at = F["A2h_action"]; ch = correct[np.arange(len(te)), at]
    rows.append(dict(seed=seed, fold=fold, arm="A2h_gate_diag", acc=ch.mean() * 100, acc_flag=ch[flag].mean() * 100, f1=np.nan, cost=np.nan, n_flag=int(flag.sum()),
                     named_correct_when_any=ch[anyc].mean() * 100, **{f"share_{e}": (at == i).mean() * 100 for i, e in enumerate(EXP)}))
    w = F["A3h_w"]; run = w >= 0.05; run[:, 0] = True
    rows.append(dict(seed=seed, fold=fold, arm="A3h_run_share", acc=np.nan, acc_flag=np.nan, f1=np.nan, cost=np.nan, n_flag=int(flag.sum()), **{f"share_{e}": run[:, i].mean() * 100 for i, e in enumerate(EXP)}))
    rows.append(dict(seed=seed, fold=fold, arm="BESTIDX", acc=np.nan, acc_flag=np.nan, f1=np.nan, cost=np.nan, n_flag=int(flag.sum()), best_idx=kb))
D = pd.DataFrame(rows); D.to_csv(a.run / "urban/t2b_per_seed_fold.csv", index=False)
Fd = D.groupby(["fold", "arm"]).mean(numeric_only=True).drop(columns="seed").reset_index(); Fd.to_csv(a.run / "urban/t2b_per_fold.csv", index=False)
folds = sorted(Fd.fold.unique()); K = len(folds); ratio = float(np.mean(list(ntr_ratio.values())))
M = lambda arm, col: Fd[Fd.arm == arm].set_index("fold").loc[folds, col].values.astype(float)
rng = np.random.default_rng(0)
def nb(d):
    s2 = d.var(ddof=1); se = np.sqrt((1 / K + ratio) * s2); t = d.mean() / se if se > 0 else 0.0; p = 2 * stats.t.sf(abs(t), K - 1); h = stats.t.ppf(.975, K - 1) * se
    bs = np.sort(d[rng.integers(0, K, (10000, K))].mean(1)); return dict(mean=float(d.mean()), ci_lo=float(d.mean() - h), ci_hi=float(d.mean() + h), p=float(p), boot_lo=float(bs[249]), boot_hi=float(bs[9749]),
                                                                          dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0, folds=[round(float(v), 2) for v in d])
CMP = {"best_single": "BEST", "always_average": "A4"}; res = []
for arm in ARMS + ["A3h_all", "AVG2", "A4", "E1", "E2c", "E2b", "E2a"]:
    for cn, cm in CMP.items():
        if arm == cm: continue
        o, fl = M(arm, "acc") - M(cm, "acc"), M(arm, "acc_flag") - M(cm, "acc_flag"); no, nf = nb(o), nb(fl)
        res.append(dict(arm=arm, comparator=cn, overall_mean=no["mean"], overall_ci_lo=no["ci_lo"], overall_ci_hi=no["ci_hi"], overall_boot_lo=no["boot_lo"], overall_boot_hi=no["boot_hi"], overall_p=no["p"], overall_dz=no["dz"],
                        folds_ge_0_5=int((o >= .5).sum()), flag_mean=nf["mean"], flag_ci_lo=nf["ci_lo"], flag_ci_hi=nf["ci_hi"], flag_boot_lo=nf["boot_lo"], flag_boot_hi=nf["boot_hi"], flag_p=nf["p"], folds_flag_ge_1=int((fl >= 1).sum()),
                        cost_arm=float(M(arm, "cost").mean()), cost_comp=float(M(cm, "cost").mean()), cost_ok=bool(M(arm, "cost").mean() <= M(cm, "cost").mean() * (1 + 1e-9)),
                        macro_f1_diff=float((M(arm, "f1") - M(cm, "f1")).mean()), overall_fold_diffs=no["folds"], flag_fold_diffs=nf["folds"]))
C = pd.DataFrame(res); C.to_csv(a.run / "urban/t2b_contrasts.csv", index=False)
# Holm over the six arms, arm p = larger of its two comparator p-values
parm = {k: max(C[(C.arm == k) & (C.comparator == c)].overall_p.iloc[0] for c in CMP) for k in ARMS}
order = sorted(ARMS, key=lambda k: parm[k]); holm, run_max = {}, 0.0
for i, k in enumerate(order): run_max = max(run_max, min(1.0, (len(ARMS) - i) * parm[k])); holm[k] = run_max
verdict = {"folds": K, "nb_ratio": ratio, "arms": {}}
for k in ARMS:
    r = {c: C[(C.arm == k) & (C.comparator == c)].iloc[0] for c in CMP}
    chk = {}
    for c, x in r.items():
        chk[c] = dict(cost_ok=bool(x.cost_ok), overall_mean_ge_0_5=bool(x.overall_mean >= .5), folds_overall_ge_7=bool(x.folds_ge_0_5 >= 7), flag_mean_ge_1=bool(x.flag_mean >= 1.0), folds_flag_ge_7=bool(x.folds_flag_ge_1 >= 7))
    ok = all(all(v.values()) for v in chk.values()) and holm[k] < 0.05
    cap = {}
    for sub, col in (("overall", "acc"), ("flagged", "acc_flag")):
        b, o4 = M("BEST", col).mean(), M("ORACLE4", col).mean(); cap[sub] = float((M(k, col).mean() - b) / (o4 - b)) if o4 > b else None
    verdict["arms"][k] = dict(checks=chk, p_raw=parm[k], p_holm=holm[k], passes=bool(ok), oracle_fraction=cap, acc=float(M(k, "acc").mean()), acc_flag=float(M(k, "acc_flag").mean()), cost=float(M(k, "cost").mean()))
best_gate = max(GATES, key=lambda k: M(k, "acc").mean()); gain = float(M(best_gate, "acc").mean() - M("A4", "acc").mean())
verdict["best_gate_arm"] = best_gate; verdict["best_gate_minus_always_average"] = gain; verdict["stop_rule_fires"] = bool(gain < 0.3)
verdict["any_arm_passes"] = bool(any(v["passes"] for v in verdict["arms"].values()))
verdict["verdict"] = "PASS" if verdict["any_arm_passes"] else ("STOP" if verdict["stop_rule_fires"] else "FAIL")
verdict["means"] = {k: dict(acc=float(M(k, "acc").mean()), acc_flag=float(M(k, "acc_flag").mean()), f1=float(np.nanmean(M(k, "f1"))), cost=float(np.nanmean(M(k, "cost")))) for k in
                    ["E1", "E2c", "E2b", "E2a", "BEST", "AVG2", "A4", "A3h_all"] + ARMS + ["ORACLE4", "ORACLE_E1E2a"]}
write_json(a.run / "urban/t2b_verdict.json", verdict); print(json.dumps({k: verdict[k] for k in ["verdict", "best_gate_arm", "best_gate_minus_always_average", "stop_rule_fires"]}, indent=1))
