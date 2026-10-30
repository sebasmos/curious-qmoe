"""T7 scoring (ANALYSIS_PLAN_T7.md sections 6 to 8). The only T7 script that reads test labels; run once after the head jobs. Writes to <run>/urban:
arms_per_seed_fold.csv, arms_per_fold.csv, arms_summary.csv, contrasts.csv, plane.csv, rules.json.
    python t7_score.py --run RUN"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_table, softmax, pmargin, f1_macro
import t7_mask as M

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); a = ap.parse_args()
out = a.run / "urban"; out.mkdir(exist_ok=True)
y = load_table().label.values.astype(np.int64)
MASKS = [f"{s}{r}" for r in (25, 50) for s in ("energy", "random", "crop")]
ARMS = ["F", "LIBF"] + [f"{h}_{m}" for h in ("CE", "KD", "LIB") for m in MASKS]


def phi(e1, e2):
    if e1.std() == 0 or e2.std() == 0: return np.nan
    return float(np.corrcoef(e1, e2)[0, 1])


rows = []
for f in sorted(glob.glob(str(a.run / "raw/s*_f*.npz"))):
    if f.endswith("_refs.npz"): continue
    R = np.load(f); te = R["te"]; yt = y[te]; tau1 = float(R["tau1"])
    for dom, suf in (("clean", "_te"), ("field", "_fld")):
        z1 = R["E1_te" if dom == "clean" else "E1_fld"].astype(np.float64)
        flag = (R["flag"] if dom == "clean" else pmargin(softmax(z1)) < tau1)
        p1 = z1.argmax(1); c1 = p1 == yt
        for arm in ["E1"] + ARMS:
            z = z1 if arm == "E1" else R[arm + suf].astype(np.float64)
            p = z.argmax(1); c = p == yt
            r = dict(seed=int(R["seed"]), fold=int(R["fold"]), domain=dom, arm=arm, n=len(te), n_flag=int(flag.sum()),
                     acc=c.mean(), f1=f1_macro(yt, p, 10), acc_flag=c[flag].mean(),
                     oracle_flag=(c | c1)[flag].mean(), oracle_all=(c | c1).mean(),
                     phi_flag=phi(~c1[flag], ~c[flag]), phi_all=phi(~c1, ~c),
                     corrects_e1_errors_flag=(c[flag & ~c1].mean() if (flag & ~c1).any() else np.nan),
                     alpha=float(R[arm + "_alpha"]) if arm.startswith("KD_") else np.nan)
            rows.append(r)
D = pd.DataFrame(rows); D.to_csv(out / "arms_per_seed_fold.csv", index=False)
num = [c for c in D.columns if c not in ("seed", "fold", "domain", "arm")]
Fd = D.groupby(["domain", "arm", "fold"])[num].mean().reset_index(); Fd.to_csv(out / "arms_per_fold.csv", index=False)
ratio = float(np.mean([(g.n.iloc[0]) / (8732 - g.n.iloc[0]) for _, g in Fd[(Fd.arm == "E1") & (Fd.domain == "clean")].groupby("fold")]))


def nb(d):
    d = np.asarray(d, float); K = len(d); se = np.sqrt((1 / K + ratio) * d.var(ddof=1)); t = stats.t.ppf(0.975, K - 1)
    return float(d.mean()), float(d.mean() - t * se), float(d.mean() + t * se), int((d > 0).sum()), int((d < 0).sum())


def piv(dom, col):
    g = Fd[Fd.domain == dom].pivot(index="fold", columns="arm", values=col); return g


# costs
CF = M.F_COST_STORED; CE1 = M.E1_COST
E1_FRONT = 8665193472.0
cost = {"E1": CE1, "F": CF, "LIBF": CF}
cost_nf = {"E1": CE1 - E1_FRONT, "F": CF - M.FBANK_BITOPS_US8K, "LIBF": CF - M.FBANK_BITOPS_US8K}
for h in ("CE", "KD", "LIB"):
    for m in MASKS:
        sel, rho = m.rstrip("0123456789"), int(m.lstrip("abcdefghijklmnopqrstuvwxyz"))
        cost[f"{h}_{m}"] = M.priced_cost(sel, rho); cost_nf[f"{h}_{m}"] = M.priced_cost(sel, rho, front=False)

summ = []
for dom in ("clean", "field"):
    A, Fl, O, Ph, F1 = piv(dom, "acc"), piv(dom, "acc_flag"), piv(dom, "oracle_flag"), piv(dom, "phi_flag"), piv(dom, "f1")
    for arm in ["E1"] + ARMS:
        r = dict(domain=dom, arm=arm, acc=A[arm].mean() * 100, f1=F1[arm].mean() * 100, acc_flag=Fl[arm].mean() * 100, oracle_flag_with_E1=O[arm].mean() * 100,
                 phi_flag_with_E1=Ph[arm].mean(), phi_all_with_E1=piv(dom, "phi_all")[arm].mean(), corrects_e1_flag_errors=piv(dom, "corrects_e1_errors_flag")[arm].mean() * 100,
                 cost=cost[arm], cost_ratio_to_F=cost[arm] / CF, cost_ratio_to_E1=cost[arm] / CE1)
        for ref in ("F", "E1"):
            if arm != ref:
                m_, lo, hi, up, dn = nb((A[arm] - A[ref]) * 100); r.update({f"acc_minus_{ref}": m_, f"acc_minus_{ref}_lo": lo, f"acc_minus_{ref}_hi": hi, f"folds_up_vs_{ref}": up, f"folds_down_vs_{ref}": dn})
        if arm not in ("E1", "F", "LIBF"):
            m_, lo, hi, up, dn = nb((O[arm] - O["F"]) * 100); r.update(oracle_gain_vs_F=m_, oracle_gain_lo=lo, oracle_gain_hi=hi, oracle_gain_folds_up=up)
            r["phi_flag_folds_below_F"] = int((Ph[arm] < Ph["F"]).sum()); r["phi_flag_minus_F"] = float((Ph[arm] - Ph["F"]).mean())
            lam = (cost[arm] - CE1) / (CF - CE1)
            gap = (A[arm] - (A["E1"] + lam * (A["F"] - A["E1"]))) * 100
            lam_nf = (cost_nf[arm] - cost_nf["E1"]) / (cost_nf["F"] - cost_nf["E1"])
            gap_nf = (A[arm] - (A["E1"] + lam_nf * (A["F"] - A["E1"]))) * 100
            ll = (np.log10(cost[arm]) - np.log10(CE1)) / (np.log10(CF) - np.log10(CE1)); gap_log = (A[arm] - (A["E1"] + ll * (A["F"] - A["E1"]))) * 100
            F1p = F1; gap_f1 = (F1p[arm] - (F1p["E1"] + lam * (F1p["F"] - F1p["E1"]))) * 100
            for nm, g in (("gap_line", gap), ("gap_line_nofront", gap_nf), ("gap_line_logcost", gap_log), ("gap_line_f1", gap_f1)):
                m_, lo, hi, up, dn = nb(g); r.update({nm: m_, nm + "_lo": lo, nm + "_hi": hi, nm + "_folds_up": up})
        summ.append(r)
S = pd.DataFrame(summ); S.to_csv(out / "arms_summary.csv", index=False)

# rules, primary arm
pa = "CE_energy25"; s = S[(S.domain == "clean") & (S.arm == pa)].iloc[0]
a1 = bool(-s.acc_minus_F <= 2.0); a2 = bool(s.gap_line > 0)
b1 = bool(s.oracle_gain_vs_F >= 1.0); b2 = bool(s.phi_flag_minus_F < 0 and s.phi_flag_folds_below_F >= 7)
stop = bool((-s.acc_minus_F > 4.0) or (not a2))
verdict = "PROOF" if (a1 and a2 and (b1 or b2)) else ("STOP" if stop else "INCONCLUSIVE")
rules = dict(primary_arm=pa, acc_F=float(S[(S.domain == "clean") & (S.arm == "F")].acc.iloc[0]), acc_E1=float(S[(S.domain == "clean") & (S.arm == "E1")].acc.iloc[0]), acc_M=float(s.acc),
             gap_F_minus_M=float(-s.acc_minus_F), a1_within_2=a1, line_gap_points=float(s.gap_line), line_gap_ci=[float(s.gap_line_lo), float(s.gap_line_hi)], a2_above_line=a2,
             line_gap_nofront=float(s.gap_line_nofront), line_gap_logcost=float(s.gap_line_logcost), line_gap_f1=float(s.gap_line_f1),
             oracle_gain_vs_F=float(s.oracle_gain_vs_F), oracle_gain_ci=[float(s.oracle_gain_lo), float(s.oracle_gain_hi)], b1_oracle_ge_1=b1,
             phi_M=float(s.phi_flag_with_E1), phi_F=float(S[(S.domain == "clean") & (S.arm == "F")].phi_flag_with_E1.iloc[0]), phi_folds_below_F=int(s.phi_flag_folds_below_F), b2_phi=b2,
             stop_gap_gt_4=bool(-s.acc_minus_F > 4.0), stop_below_line=bool(not a2), verdict=verdict)
(out / "rules.json").write_text(json.dumps(rules, indent=1)); print(json.dumps(rules, indent=1))
