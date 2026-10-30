"""T1a and T1 on TAU 2020 Mobile (ANALYSIS_PLAN.md Amendment 1). Reads <run>/tau/raw/s*_f*.npz. Primary condition: shift (other devices);
clean (device A) alongside. K = 5 folds, "7 of 10" read as "4 of 5".
    python s7_tau_analysis.py --run DIR"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "idea6/stage2"))
from common import write_json, softmax, pmargin
from heads import macro_f1
from idea5b_common import holm

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); ap.add_argument("--boot", type=int, default=10000); a = ap.parse_args()
TAU = Path.home() / "Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv"
d = pd.read_csv(TAU, usecols=["folder", "name", "class_id"]); y = d.class_id.values.astype(np.int64)
T = a.run / "tau"; (T / "analysis").mkdir(exist_ok=True)
def lmargin(z):
    s = np.sort(z, 1); return s[:, -1] - s[:, -2]
ARMS = ["U", "M0", "R05", "R10", "R20", "F", "T", "UA", "M0A", "R10A"]
PAIRS = [("R10", "U", "primary"), ("M0", "U", "B"), ("R05", "U", "B"), ("R20", "U", "B"), ("F", "U", "B"), ("T", "U", "B"), ("M0A", "UA", "A"), ("R10A", "UA", "A")]
t1a, arms = [], []
for f in sorted(glob.glob(str(T / "raw/s*_f*.npz"))):
    R = np.load(f); seed, fold = [int(t[1:]) for t in Path(f).stem.split("_")]
    tr, te, cond = R["tr"], R["te"], R["cond"]; yt = y[te]
    z1, z2 = R["z1_oof"].astype(np.float64), R["z2_oof"].astype(np.float64); p1, p2 = softmax(z1), softmax(z2); m1, m2 = pmargin(p1), pmargin(p2); w2 = p2.argmax(1) != y[tr]
    t1a.append(dict(seed=seed, fold=fold, split="train", n=len(tr), spearman=spearmanr(m1, m2)[0], auroc=roc_auc_score(w2, 1 - m1), auroc_logit_margin=roc_auc_score(w2, -lmargin(z1)),
                    auroc_d=roc_auc_score(w2, R["e1_d"]), auroc_negAUM=roc_auc_score(w2, -R["e1_aum"]), auroc_el2n=roc_auc_score(w2, R["e1_el2n"]), auroc_forget=roc_auc_score(w2, R["e1_forget"]),
                    spearman_d1_d2=spearmanr(R["e1_d"], R["e2_d"])[0], err_E1=float((p1.argmax(1) != y[tr]).mean()), err_E2=float(w2.mean())))
    z2t = R["UA_te"].astype(np.float64); m1t, m2t = R["m1_te"], pmargin(softmax(z2t)); z1t = R["z1_te"].astype(np.float64)
    for cn, cname in ((1, "test_shift"), (0, "test_clean")):
        k = cond == cn; w = z2t[k].argmax(1) != yt[k]
        t1a.append(dict(seed=seed, fold=fold, split=cname, n=int(k.sum()), spearman=spearmanr(m1t[k], m2t[k])[0], auroc=roc_auc_score(w, 1 - m1t[k]),
                        auroc_logit_margin=roc_auc_score(w, -lmargin(z1t[k])), err_E1=float((z1t[k].argmax(1) != yt[k]).mean()), err_E2=float(w.mean())))
    e1 = R["z1_te"].argmax(1); flag = R["flag"]
    for cn, cname in ((1, "shift"), (0, "clean")):
        k = cond == cn; fl = flag & k; nf = ~flag & k
        for arm in ARMS + ["E1"]:
            pr = e1 if arm == "E1" else R[f"{arm}_te"].argmax(1)
            arms.append(dict(seed=seed, fold=fold, cond=cname, arm=arm, n_cond=int(k.sum()), n_train=len(tr), n_flag=int(fl.sum()), flag_share=float(fl.sum() / k.sum()),
                             acc_flag=float((pr[fl] == yt[fl]).mean()), f1_flag=macro_f1(yt[fl], pr[fl]), acc_unflag=float((pr[nf] == yt[nf]).mean()), acc_all=float((pr[k] == yt[k]).mean()),
                             cascade_acc=float((np.where(flag, pr, e1)[k] == yt[k]).mean())))
P = pd.DataFrame(t1a).sort_values(["split", "seed", "fold"]); P.to_csv(T / "analysis/t1a_per_seed_fold.csv", index=False)
Fd = P.groupby(["split", "fold"]).mean(numeric_only=True).drop(columns="seed").reset_index(); Fd.to_csv(T / "analysis/t1a_per_fold.csv", index=False)
v = {}
for s in ("train", "test_shift", "test_clean"):
    g = Fd[Fd.split == s]; ok = (g.spearman >= 0.3) & (g.auroc >= 0.6)
    v[s] = dict(n_folds=int(len(g)), mean_spearman=float(g.spearman.mean()), min_spearman=float(g.spearman.min()), mean_auroc=float(g.auroc.mean()), min_auroc=float(g.auroc.min()),
                folds_both=int(ok.sum()), mean_auroc_logit_margin=float(g.auroc_logit_margin.mean()), mean_err_E1=float(g.err_E1.mean()), mean_err_E2=float(g.err_E2.mean()))
    if s == "train": v[s].update({k: float(g[k].mean()) for k in ("auroc_d", "auroc_negAUM", "auroc_el2n", "auroc_forget", "spearman_d1_d2")})
nf = Fd[Fd.split == "train"].fold.nunique()
v["verdict_shift"] = "PASS" if (nf == 5 and v["train"]["folds_both"] >= 4 and v["test_shift"]["folds_both"] >= 4) else ("STOP" if v["train"]["mean_spearman"] < 0.15 and v["test_shift"]["mean_spearman"] < 0.15 else "INCONCLUSIVE")
v["rule"] = "PASS: Spearman >= 0.3 and AUROC >= 0.6 in >= 4 of 5 folds on training folds and on shift test rows; STOP: mean Spearman < 0.15 on both"
write_json(T / "analysis/t1a_verdict.json", v)
A = pd.DataFrame(arms); A.to_csv(T / "analysis/arm_metrics_per_seed_fold.csv", index=False)
AF = A.groupby(["cond", "arm", "fold"]).mean(numeric_only=True).drop(columns="seed").reset_index(); AF.to_csv(T / "analysis/arm_metrics_per_fold.csv", index=False)
AF.groupby(["cond", "arm"]).mean(numeric_only=True).drop(columns="fold").to_csv(T / "analysis/arm_metrics_mean.csv")
rng = np.random.default_rng(12345); out = []
for cname in ("shift", "clean"):
    for arm, twin, fam in PAIRS:
        for metric in ("acc_flag", "f1_flag", "acc_unflag", "acc_all", "cascade_acc"):
            sa = AF[(AF.cond == cname) & (AF.arm == arm)].set_index("fold")[metric]; sb = AF[(AF.cond == cname) & (AF.arm == twin)].set_index("fold")[metric]
            fm = ((sa - sb) * 100).dropna(); k = len(fm)
            if k < 3: continue
            ref = AF[(AF.cond == cname) & (AF.arm == "E1")].set_index("fold"); ratio = float((ref.n_cond / ref.n_train).mean()); sd = fm.std(ddof=1)
            se = np.sqrt((1 / k + ratio) * sd ** 2); t_nb = fm.mean() / se if sd > 0 else np.inf; tc = stats.t.ppf(0.975, k - 1)
            bs = np.sort(rng.choice(fm.to_numpy(), (a.boot, k)).mean(1))
            out.append(dict(cond=cname, family=fam, arm=arm, twin=twin, metric=metric, n_folds=k, mean_contrast_pts=fm.mean(), ci95_nb_lo=fm.mean() - tc * se, ci95_nb_hi=fm.mean() + tc * se,
                            ci95_boot_lo=bs[int(0.025 * a.boot)], ci95_boot_hi=bs[int(0.975 * a.boot)], dz=fm.mean() / sd if sd > 0 else np.nan, t_nb=t_nb, p_nb_two_sided=float(2 * stats.t.sf(abs(t_nb), k - 1)),
                            nb_ratio=ratio, folds_ge_0_5=int((fm >= 0.5).sum()), folds_ge_1_0=int((fm >= 1.0).sum()), folds_positive=int((fm > 0).sum()), fold_contrasts=json.dumps([round(x, 3) for x in fm.values])))
C = pd.DataFrame(out); C["holm_p_nb"] = np.nan
for cname in ("shift", "clean"):
    ex = C[(C.cond == cname) & (C.metric == "acc_flag") & (C.family != "primary")]; C.loc[ex.index, "holm_p_nb"] = holm(ex.p_nb_two_sided.to_numpy())
C.to_csv(T / "analysis/t1_contrasts.csv", index=False)
vv = {"rule": "proof: >= +1.0 pt in >= 4 of 5 folds, NB two-sided p < 0.05, positive mean; stop: mean contrast < 0.3 pt; both on flagged shift clips, primary arm R10 (input B) minus U"}
for cname in ("shift", "clean"):
    pr = C[(C.cond == cname) & (C.arm == "R10") & (C.metric == "acc_flag")].iloc[0]
    vv[cname] = dict(mean_contrast_pts=float(pr.mean_contrast_pts), ci95_nb=[float(pr.ci95_nb_lo), float(pr.ci95_nb_hi)], ci95_boot=[float(pr.ci95_boot_lo), float(pr.ci95_boot_hi)], dz=float(pr.dz),
                     p_nb_two_sided=float(pr.p_nb_two_sided), folds_ge_0_5=int(pr.folds_ge_0_5), folds_ge_1_0=int(pr.folds_ge_1_0), n_folds=int(pr.n_folds), nb_ratio=float(pr.nb_ratio),
                     flag_share=float(AF[(AF.cond == cname) & (AF.arm == "E1")].flag_share.mean()),
                     proof=bool(pr.n_folds == 5 and pr.folds_ge_1_0 >= 4 and pr.p_nb_two_sided < 0.05 and pr.mean_contrast_pts > 0), stop=bool(pr.mean_contrast_pts < 0.3))
vv["verdict_shift"] = "PROOF" if vv["shift"]["proof"] else ("STOP" if vv["shift"]["stop"] else "INCONCLUSIVE")
write_json(T / "analysis/t1_verdict.json", vv)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(Fd.round(3).to_string()); print(json.dumps(v, indent=1)); print(AF.groupby(["cond", "arm"]).mean(numeric_only=True).drop(columns="fold").round(4).to_string())
print(C[C.metric == "acc_flag"].drop(columns=["fold_contrasts", "metric"]).round(4).to_string()); print(json.dumps(vv, indent=1))
