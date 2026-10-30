"""T1 analysis: specialist minus uniform twin on the clips E1 flags, per fold, with the Nadeau-Bengio test. ANALYSIS_PLAN.md B6 to B8.
    python s3_t1.py --run DIR"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "idea6/stage2"))
from common import *
from heads import macro_f1
from idea5b_common import holm

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); ap.add_argument("--boot", type=int, default=10000); a = ap.parse_args()
d = load_table(); y = d.label.values.astype(np.int64)
ARMS_E2 = ["U", "M0", "R05", "R10", "R20", "F", "T", "UA", "M0A", "R10A"]
ARMS_E3 = ["E3U", "E3M0", "E3R10", "E3F"]
PAIRS = [("R10", "U", "primary"), ("M0", "U", "E2 B"), ("R05", "U", "E2 B"), ("R20", "U", "E2 B"), ("F", "U", "E2 B"), ("T", "U", "E2 B"),
         ("M0A", "UA", "E2 A"), ("R10A", "UA", "E2 A"), ("E3M0", "E3U", "E3"), ("E3R10", "E3U", "E3"), ("E3F", "E3U", "E3")]
rows = []
for f in sorted(glob.glob(str(a.run / "t1/raw/s*_f*.npz"))):
    R = np.load(f); seed, fold = [int(t[1:]) for t in Path(f).stem.split("_")]
    te = R["te"]; yt = y[te]; flag = R["flag"]; flag3 = R["flag3"] if "flag3" in R.files else None
    e1 = R["z1_te"].argmax(1)
    for arm in ARMS_E2 + ARMS_E3:
        if f"{arm}_te" not in R.files: continue
        fl = flag3 if arm.startswith("E3") else flag
        pr = R[f"{arm}_te"].argmax(1)
        rows.append(dict(seed=seed, fold=fold, arm=arm, n_te=len(te), n_flag=int(fl.sum()), flag_share=float(fl.mean()),
                         acc_flag=float((pr[fl] == yt[fl]).mean()), f1_flag=macro_f1(yt[fl], pr[fl]), acc_unflag=float((pr[~fl] == yt[~fl]).mean()),
                         acc_all=float((pr == yt).mean()), cascade_acc=float((np.where(fl, pr, e1) == yt).mean()), epochs=int(R[f"{arm}_epochs"])))
    rows.append(dict(seed=seed, fold=fold, arm="E1", n_te=len(te), n_flag=int(flag.sum()), flag_share=float(flag.mean()), acc_flag=float((e1[flag] == yt[flag]).mean()),
                     f1_flag=macro_f1(yt[flag], e1[flag]), acc_unflag=float((e1[~flag] == yt[~flag]).mean()), acc_all=float((e1 == yt).mean()), cascade_acc=float((e1 == yt).mean()), epochs=int(R["e1_epochs"])))
A = pd.DataFrame(rows).sort_values(["arm", "seed", "fold"]); A.to_csv(a.run / "t1/t1_arm_metrics_per_seed_fold.csv", index=False)
AF = A.groupby(["arm", "fold"]).mean(numeric_only=True).drop(columns="seed").reset_index(); AF.to_csv(a.run / "t1/t1_arm_metrics_per_fold.csv", index=False)
summ_arm = AF.groupby("arm").mean(numeric_only=True).drop(columns="fold"); summ_arm["n_folds"] = AF.groupby("arm").fold.nunique(); summ_arm.to_csv(a.run / "t1/t1_arm_metrics_mean.csv")
n_te = AF[AF.arm == "E1"].set_index("fold").n_te; n_all = len(d)
ratio = float(np.mean(n_te / (n_all - n_te)))
rng = np.random.default_rng(12345)
out = []
for arm, twin, fam in PAIRS:
    if arm not in set(A.arm) or twin not in set(A.arm): continue
    for metric in ("acc_flag", "f1_flag", "acc_unflag", "acc_all", "cascade_acc"):
        sa, sb = AF[AF.arm == arm].set_index("fold")[metric], AF[AF.arm == twin].set_index("fold")[metric]
        fm = ((sa - sb) * 100).dropna(); k = len(fm)
        if k < 3: continue
        sd = fm.std(ddof=1)
        t_nb = fm.mean() / np.sqrt((1 / k + ratio) * sd ** 2) if sd > 0 else np.inf
        p_nb = float(2 * stats.t.sf(abs(t_nb), k - 1)); tcrit = stats.t.ppf(0.975, k - 1); se = np.sqrt((1 / k + ratio) * sd ** 2)
        bs = np.sort(rng.choice(fm.to_numpy(), (a.boot, k)).mean(1))
        out.append(dict(family=fam, arm=arm, twin=twin, metric=metric, n_folds=k, mean_contrast_pts=fm.mean(), ci95_nb_lo=fm.mean() - tcrit * se, ci95_nb_hi=fm.mean() + tcrit * se,
                        ci95_boot_lo=bs[int(0.025 * a.boot)], ci95_boot_hi=bs[int(0.975 * a.boot)], dz=fm.mean() / sd if sd > 0 else np.nan, t_nb=t_nb, p_nb_two_sided=p_nb,
                        p_plain_t=float(stats.ttest_1samp(fm, 0).pvalue), folds_ge_0_5=int((fm >= 0.5).sum()), folds_ge_1_0=int((fm >= 1.0).sum()), folds_positive=int((fm > 0).sum()),
                        fold_contrasts=json.dumps([round(v, 3) for v in fm.values])))
C = pd.DataFrame(out)
C["holm_p_nb"] = np.nan
for fam in ("E2 B", "E2 A", "E3"):
    pass
expl = C[(C.metric == "acc_flag") & (C.family.isin(["E2 B", "E2 A"]))]
C.loc[expl.index, "holm_p_nb"] = holm(expl.p_nb_two_sided.to_numpy())
e3 = C[(C.metric == "acc_flag") & (C.family == "E3")]
C.loc[e3.index, "holm_p_nb"] = holm(e3.p_nb_two_sided.to_numpy())
C.to_csv(a.run / "t1/t1_contrasts.csv", index=False)
pr = C[(C.arm == "R10") & (C.metric == "acc_flag")].iloc[0]
v = dict(primary_arm="R10 (epsilon 0.1, input B) minus U", n_folds=int(pr.n_folds), mean_contrast_pts=float(pr.mean_contrast_pts), ci95_nb=[float(pr.ci95_nb_lo), float(pr.ci95_nb_hi)],
         ci95_boot=[float(pr.ci95_boot_lo), float(pr.ci95_boot_hi)], dz=float(pr.dz), p_nb_two_sided=float(pr.p_nb_two_sided), folds_ge_0_5=int(pr.folds_ge_0_5), folds_ge_1_0=int(pr.folds_ge_1_0),
         nb_ratio=ratio, flag_share_mean=float(AF[AF.arm == "E1"].flag_share.mean()))
v["proof"] = bool(pr.n_folds == 10 and pr.folds_ge_1_0 >= 7 and pr.p_nb_two_sided < 0.05 and pr.mean_contrast_pts > 0)
v["stop"] = bool(pr.mean_contrast_pts < 0.3)
v["d1_go_contrast_part"] = bool(pr.n_folds == 10 and pr.folds_ge_0_5 >= 7)
t1a = json.loads((a.run / "t1a/t1a_verdict.json").read_text()) if (a.run / "t1a/t1a_verdict.json").exists() else None
v["t1a_mean_test_spearman"] = t1a["test"]["mean_spearman"] if t1a else None
v["d1_go"] = bool(v["d1_go_contrast_part"] and t1a and t1a["test"]["mean_spearman"] > 0.3)
v["verdict"] = "PROOF" if v["proof"] else ("STOP" if v["stop"] else "INCONCLUSIVE")
v["rules"] = {"proof": ">= +1.0 pt in >= 7 of 10 folds and NB two-sided p < 0.05 with positive mean (T1 row)", "stop": "mean contrast < 0.3 pt (T1 row, D1 prediction)",
              "d1_go": ">= +0.5 pt in >= 7 of 10 folds and T1a mean test-fold Spearman > 0.3 (D1 stage 1)"}
write_json(a.run / "t1/t1_verdict.json", v)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(summ_arm.round(4).to_string()); print(C[C.metric == "acc_flag"].drop(columns=["fold_contrasts", "metric"]).round(4).to_string()); print(json.dumps(v, indent=1, default=float))
