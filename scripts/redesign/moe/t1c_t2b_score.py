"""T1c gated-mixture scoring (ANALYSIS_PLAN_T1C.md section 7). Reads test labels; run once after t1c_t2b_fit.py, only for the gate-passing conditions.
python t1c_t2b_score.py --run DIR"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1c_common import *
from common import f1_macro, write_json
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); a = ap.parse_args()
ARMS = ["A1", "A2h", "A2f", "A3h", "A3f", "A5"]; GATES = ["A1", "A2h", "A2f", "A3h", "A3f"]; CMP = {"best_single": "BEST", "always_average": "A4"}
all_tested = [c for b in CONDITIONS for c in CONDITIONS[b]]; gate = {}
for bed in BEDS: gate.update(json.load(open(a.run / bed / "strength_verdict.json"))["gate_pass"])
rng = np.random.default_rng(0)
def nb(d, ratio):
    K = len(d); s2 = d.var(ddof=1); se = np.sqrt((1 / K + ratio) * s2); t = d.mean() / se if se > 0 else 0.0; p = float(2 * stats.t.sf(abs(t), K - 1)); h = stats.t.ppf(.975, K - 1) * se
    bs = np.sort(d[rng.integers(0, K, (10000, K))].mean(1))
    return dict(mean=float(d.mean()), ci_lo=float(d.mean() - h), ci_hi=float(d.mean() + h), p=p, boot_lo=float(bs[249]), boot_hi=float(bs[9749]), dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0)
rows, ratios, contrast_rows, V = [], {}, [], {"conditions": {}}
for bed in BEDS:
    run_conds = [c for c in CONDITIONS[bed] if gate[c]]
    if not run_conds: continue
    D = load_bed(bed); y = D["y"]; K = BEDS[bed]["ncls"]; EX2 = EXPERTS2[bed]; EXP = ["E1"] + EX2; G = len(EXP)
    for fpath in sorted(glob.glob(str(a.run / bed / "t2b_fit/s*_f*.npz"))):
        stem = Path(fpath).stem; Fz = np.load(fpath); S = np.load(a.run / bed / f"raw_strength/{stem}.npz"); te, tr = S["te"], S["tr"]; yte = y[te]; cond = S["cond"] if bed == "tau" else None
        seed, fold = int(S["seed"]), int(S["fold"]); COST = Fz["cost"]; kb = int(Fz["kb"]); tau1 = float(S["tau1"])
        for cn in run_conds:
            g = groups(bed, cn, te, D["assign"], cond); yr = gather_y(yte, g); ratios[(cn, seed, fold)] = (len(g[0][1]) if bed == "tau" else len(te)) / len(tr)
            Z = [gather({c: S[f"z1|{c}"] for c, _ in g}, g).astype(np.float64)] + [gather({c: S[f"UA_{e}|{c}"] for c, _ in g}, g).astype(np.float64) for e in EX2]
            P = [softmax(z) for z in Z]; flag = pmargin(P[0]) < tau1; n = len(yr)
            out = {EXP[i]: (P[i], np.full(n, COST[i])) for i in range(G)}; out["BEST"] = (P[kb], np.full(n, COST[kb]))
            out["A4"] = (Fz[f"{cn}|A4_p"], Fz[f"{cn}|A4_cost"]); out["AVG2"] = (Fz[f"{cn}|AVG2_p"], np.full(n, COST[0] + COST[-1]))
            for k in ARMS: out[k] = (Fz[f"{cn}|{k}_p"], Fz[f"{cn}|{k}_cost"])
            out["A3h_all"] = (Fz[f"{cn}|A3h_p_all"], Fz[f"{cn}|A3h_cost_all"])
            correct = np.stack([p.argmax(1) == yr for p in P], 1); anyc = correct.any(1)
            for k, (p, c) in out.items():
                pr = p.argmax(1); ok = pr == yr
                rows.append(dict(bed=bed, cond=cn, seed=seed, fold=fold, arm=k, acc=ok.mean() * 100, acc_flag=ok[flag].mean() * 100, f1=f1_macro(yr, pr, K) * 100, cost=c.mean(), n_flag=int(flag.sum())))
            for nm, v in [("ORACLE_ALL", anyc), ("ORACLE_E1E2a", correct[:, 0] | correct[:, -1])]:
                rows.append(dict(bed=bed, cond=cn, seed=seed, fold=fold, arm=nm, acc=v.mean() * 100, acc_flag=v[flag].mean() * 100, f1=np.nan, cost=np.nan, n_flag=int(flag.sum())))
            at = Fz[f"{cn}|A2h_action"]; ch = correct[np.arange(n), at]
            rows.append(dict(bed=bed, cond=cn, seed=seed, fold=fold, arm="A2h_gate_diag", acc=ch.mean() * 100, acc_flag=ch[flag].mean() * 100, f1=np.nan, cost=np.nan, n_flag=int(flag.sum()), **{f"share_{e}": (at == i).mean() * 100 for i, e in enumerate(EXP)}))
            w = Fz[f"{cn}|A3h_w"]; run = w >= 0.05; run[:, 0] = True
            rows.append(dict(bed=bed, cond=cn, seed=seed, fold=fold, arm="A3h_run_share", acc=np.nan, acc_flag=np.nan, f1=np.nan, cost=np.nan, n_flag=int(flag.sum()), **{f"share_{e}": run[:, i].mean() * 100 for i, e in enumerate(EXP)}))
            rows.append(dict(bed=bed, cond=cn, seed=seed, fold=fold, arm="BESTIDX", acc=np.nan, acc_flag=np.nan, f1=np.nan, cost=np.nan, n_flag=int(flag.sum()), best_idx=kb))
Dd = pd.DataFrame(rows); (a.run / "t2b").mkdir(exist_ok=True); Dd.to_csv(a.run / "t2b/t2b_per_seed_fold.csv", index=False)
Fd = Dd.groupby(["cond", "fold", "arm"]).mean(numeric_only=True).drop(columns="seed").reset_index(); Fd.to_csv(a.run / "t2b/t2b_per_fold.csv", index=False)
singles = lambda bed: ["E1"] + EXPERTS2[bed]; cres = []; adj_min = {}
for cn in [c for c in all_tested if gate[c]]:
    bed = BED_OF[cn]; Fc = Fd[Fd.cond == cn]; folds = sorted(Fc.fold.unique()); K_ = len(folds); ratio = float(np.mean([v for k, v in ratios.items() if k[0] == cn])); maj = K_ // 2 + 1; old = 7 if K_ == 10 else 4
    M = lambda arm, col: Fc[Fc.arm == arm].set_index("fold").loc[folds, col].values.astype(float)
    res = []
    for arm in ARMS + ["A3h_all", "AVG2", "A4"] + singles(bed):
        for cmn, cm in CMP.items():
            if arm == cm: continue
            o, fl = M(arm, "acc") - M(cm, "acc"), M(arm, "acc_flag") - M(cm, "acc_flag"); no, nf = nb(o, ratio), nb(fl, ratio)
            res.append(dict(cond=cn, arm=arm, comparator=cmn, overall_mean=no["mean"], overall_ci_lo=no["ci_lo"], overall_ci_hi=no["ci_hi"], overall_p=no["p"], overall_dz=no["dz"], folds_ge_0_5=int((o >= .5).sum()),
                            flag_mean=nf["mean"], flag_ci_lo=nf["ci_lo"], flag_ci_hi=nf["ci_hi"], flag_p=nf["p"], folds_flag_ge_1=int((fl >= 1).sum()), cost_arm=float(M(arm, "cost").mean()), cost_comp=float(M(cm, "cost").mean()),
                            cost_ok=bool(M(arm, "cost").mean() <= M(cm, "cost").mean() * (1 + 1e-9)), macro_f1_diff=float((M(arm, "f1") - M(cm, "f1")).mean()), majority_needed=maj, old_count_needed=old,
                            overall_fold_diffs=[round(float(v), 2) for v in o], flag_fold_diffs=[round(float(v), 2) for v in fl]))
    C = pd.DataFrame(res); cres.append(C)
    parm = {k: max(C[(C.arm == k) & (C.comparator == c)].overall_p.iloc[0] for c in CMP) for k in ARMS}
    order = sorted(ARMS, key=lambda k: parm[k]); holm, run_max = {}, 0.0
    for i, k in enumerate(order): run_max = max(run_max, min(1.0, (len(ARMS) - i) * parm[k])); holm[k] = run_max
    adj_min[cn] = min(holm.values())
    V["conditions"][cn] = dict(parm=parm, holm_arms=holm, arms={}, K=K_, majority_needed=maj)
    for k in ARMS:
        r = {c: C[(C.arm == k) & (C.comparator == c)].iloc[0] for c in CMP}
        chk = {c: dict(cost_ok=bool(x.cost_ok), overall_mean_ge_0_5=bool(x.overall_mean >= .5), folds_overall_majority=bool(x.folds_ge_0_5 >= maj), flag_mean_ge_1=bool(x.flag_mean >= 1.0), folds_flag_majority=bool(x.folds_flag_ge_1 >= maj),
                         folds_overall_ge_old=bool(x.folds_ge_0_5 >= old), folds_flag_ge_old=bool(x.folds_flag_ge_1 >= old)) for c, x in r.items()}
        must = ["cost_ok", "overall_mean_ge_0_5", "folds_overall_majority", "flag_mean_ge_1", "folds_flag_majority"]
        V["conditions"][cn]["arms"][k] = dict(checks=chk, structural_ok=bool(all(all(v[m] for m in must) for v in chk.values())), p_raw=parm[k], p_holm_arms=holm[k],
                                              acc=float(M(k, "acc").mean()), acc_flag=float(M(k, "acc_flag").mean()), cost=float(M(k, "cost").mean()))
    best_gate = max(GATES, key=lambda k: M(k, "acc").mean()); gain = float(M(best_gate, "acc").mean() - M("A4", "acc").mean())
    V["conditions"][cn].update(best_gate_arm=best_gate, best_gate_minus_always_average=gain, stop_rule_fires=bool(gain < 0.3),
                               means={k: dict(acc=float(M(k, "acc").mean()), acc_flag=float(M(k, "acc_flag").mean()), f1=float(np.nanmean(M(k, "f1"))), cost=float(np.nanmean(M(k, "cost")))) for k in
                                      singles(bed) + ["BEST", "AVG2", "A4", "A3h_all"] + ARMS + ["ORACLE_ALL", "ORACLE_E1E2a"]})
# Holm over the T1c conditions applied to each condition's smallest arm-adjusted p (failed-gate conditions carry 1)
P5 = {c: adj_min.get(c, 1.0) for c in all_tested}; order = sorted(P5, key=lambda c: P5[c]); holm5, run_max = {}, 0.0
for i, c in enumerate(order): run_max = max(run_max, min(1.0, (len(P5) - i) * P5[c])); holm5[c] = run_max
for cn in [c for c in all_tested if gate[c]]:
    v = V["conditions"][cn]; v["min_arm_holm_p"] = P5[cn]; v["p_holm_over_conditions"] = holm5[cn]
    for k, ar in v["arms"].items(): ar["passes"] = bool(ar["structural_ok"] and ar["p_holm_arms"] < 0.05 and holm5[cn] < 0.05)
    v["any_arm_passes"] = any(ar["passes"] for ar in v["arms"].values()); v["verdict"] = "PASS" if v["any_arm_passes"] else ("STOP" if v["stop_rule_fires"] else "FAIL")
V["gate_pass"] = gate; V["holm_over_conditions"] = holm5
if cres: pd.concat(cres).to_csv(a.run / "t2b/t2b_contrasts.csv", index=False)
write_json(a.run / "t2b/t2b_verdict.json", V)
for cn, v in V["conditions"].items(): print(cn, v["verdict"], "best gate", v["best_gate_arm"], "%.2f vs always-average" % v["best_gate_minus_always_average"], {k: round(ar["acc"], 2) for k, ar in v["arms"].items()}, "A4 %.2f" % v["means"]["A4"]["acc"])
