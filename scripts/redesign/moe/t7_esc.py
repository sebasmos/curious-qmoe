"""T7 secondary: ESC-50 clean (ANALYSIS_PLAN_T7.md section 10 and amendment 2). Heads (library recipe, macro F1 on the grouped inner validation) for E1, F and the six masked arms, then scoring.
A cell's heads are all fitted before its test labels are used, and nothing after that step selects anything. Writes <run>/esc50/{arms_per_seed_fold,arms_per_fold,arms_summary}.csv and rules.json.
    python t7_esc.py --run RUN [--seeds 0 1 2]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2]))
import heads as H
H.NCLS = 50
_mf = H.macro_f1; H.macro_f1 = lambda y, p, k=50: _mf(y, p, k)
from common import Q, S3B, softmax, pmargin, f1_macro
from QWave.datasets import inner_split, recording_groups
import t7_mask as M

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args(); torch.set_num_threads(a.threads)
out = a.run / "esc50"; out.mkdir(exist_ok=True)
S2D = Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"; S3C = Q / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1"
d = pd.read_csv(Path.home() / "Documents/DATASETS/ESC-50/ast_768/esc-50.csv", usecols=["folder", "name", "label", "class_id"])
y = d.label.values.astype(np.int64); fold_of = d.folder.values; N = len(d); K = 50
rec = recording_groups(d["name"].to_numpy(), 1)
X1 = np.load(S3B / "emb_cnn/esc50/mn10_w8a8__clean.npy").astype(np.float32)
XF = np.load(S2D / "emb_ast_short/esc50/w4g128__clean.npy").astype(np.float32)
MASKS = [f"{s}{r}" for r in (25, 50) for s in ("energy", "random", "crop")]
XM = {m: np.load(a.run / f"emb/esc50/w4g128__{m}__clean.npy").astype(np.float32) for m in MASKS}
for X in [X1, XF, *XM.values()]: assert X.shape[0] == N and np.isfinite(X).all()
costs = json.loads((S3C / "analysis/esc50/costs.json").read_text())
FB, HD, CF, CE1 = costs["fbank_bitops_mean"], costs["heads"]["768"]["esc"]["bitops"], costs["ast"]["w4g128@512+esc"], costs["cheap"]["cnn:mn10:w8a8:esc"]
E1_FRONT = json.loads((S3B / "emb_cnn/esc50/mn10_w8a8__clean.json").read_text())["cost_per_clip"]["frontend_bitops_mean"]
L = 512
assert abs(M.priced_cost(None, 100, L, fbank=FB, head=HD) / CF - 1) < 1e-6, "R3 (ESC-50) failed"


def phi(e1, e2):
    return np.nan if e1.std() == 0 or e2.std() == 0 else float(np.corrcoef(e1, e2)[0, 1])


rows = []
for seed in a.seeds:
    for fold in range(1, 6):
        t0 = time.time()
        tr = np.flatnonzero(fold_of != fold); te = np.flatnonzero(fold_of == fold)
        fit_g, iv_g = inner_split(tr, y, 0.1, seed, rec)
        def fh(X):
            h = H.fit_head(X[fit_g], y[fit_g], np.ones(len(fit_g)), seed, X[iv_g], y[iv_g], metric="f1"); return h, h.logits(X[te])
        h1, z1 = fh(X1); zs = {"F": fh(XF)[1], **{f"LIB_{m}": fh(XM[m])[1] for m in MASKS}}
        m1_iv = pmargin(softmax(h1.logits(X1[iv_g]))); u = np.unique(m1_iv); c = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
        rates = (m1_iv[None, :] < c[:, None]).mean(1); tau1 = c[np.lexsort((rates, np.abs(rates - 0.30)))[0]]
        flag = pmargin(softmax(z1)) < tau1                      # the cells' heads are all fitted; labels are read from here on
        yt = y[te]; p1 = z1.argmax(1); c1 = p1 == yt
        for arm, z in {"E1": z1, **zs}.items():
            p = z.argmax(1); cc = p == yt
            rows.append(dict(seed=seed, fold=fold, arm=arm, n=len(te), n_flag=int(flag.sum()), acc=cc.mean(), f1=f1_macro(yt, p, K), acc_flag=cc[flag].mean(), oracle_flag=(cc | c1)[flag].mean(),
                             phi_flag=phi(~c1[flag], ~cc[flag]), phi_all=phi(~c1, ~cc), corrects_e1_errors_flag=cc[flag & ~c1].mean() if (flag & ~c1).any() else np.nan))
        print(f"s{seed} f{fold}: {time.time()-t0:.0f}s flagged {flag.mean():.3f}", flush=True)
D = pd.DataFrame(rows); D.to_csv(out / "arms_per_seed_fold.csv", index=False)
num = [c for c in D.columns if c not in ("seed", "fold", "arm")]
Fd = D.groupby(["arm", "fold"])[num].mean().reset_index(); Fd.to_csv(out / "arms_per_fold.csv", index=False)
ratio = float(np.mean(Fd[Fd.arm == "E1"].n / (N - Fd[Fd.arm == "E1"].n)))


def nb(x):
    x = np.asarray(x, float); k = len(x); se = np.sqrt((1 / k + ratio) * x.var(ddof=1)); t = stats.t.ppf(0.975, k - 1)
    return float(x.mean()), float(x.mean() - t * se), float(x.mean() + t * se), int((x > 0).sum()), int((x < 0).sum())


def piv(col): return Fd.pivot(index="fold", columns="arm", values=col)


A, Fl, O, Ph, F1 = piv("acc"), piv("acc_flag"), piv("oracle_flag"), piv("phi_flag"), piv("f1")
cost = {"E1": CE1, "F": CF}; cost_nf = {"E1": CE1 - E1_FRONT, "F": CF - FB}
for m in MASKS:
    sel, rho = m.rstrip("0123456789"), int(m.lstrip("abcdefghijklmnopqrstuvwxyz"))
    cost[f"LIB_{m}"] = M.priced_cost(sel, rho, L, fbank=FB, head=HD); cost_nf[f"LIB_{m}"] = M.priced_cost(sel, rho, L, fbank=FB, head=HD, front=False)
summ = []
for arm in ["E1", "F"] + [f"LIB_{m}" for m in MASKS]:
    r = dict(arm=arm, acc=A[arm].mean() * 100, f1=F1[arm].mean() * 100, acc_flag=Fl[arm].mean() * 100, oracle_flag_with_E1=O[arm].mean() * 100, phi_flag_with_E1=Ph[arm].mean(),
             phi_all_with_E1=piv("phi_all")[arm].mean(), cost=cost[arm], cost_ratio_to_F=cost[arm] / CF, cost_ratio_to_E1=cost[arm] / CE1)
    for ref in ("F", "E1"):
        if arm != ref:
            m_, lo, hi, up, dn = nb((A[arm] - A[ref]) * 100); r.update({f"acc_minus_{ref}": m_, f"acc_minus_{ref}_lo": lo, f"acc_minus_{ref}_hi": hi, f"folds_up_vs_{ref}": up})
    if arm.startswith("LIB_"):
        m_, lo, hi, up, dn = nb((O[arm] - O["F"]) * 100); r.update(oracle_gain_vs_F=m_, oracle_gain_lo=lo, oracle_gain_hi=hi, oracle_gain_folds_up=up)
        r["phi_flag_folds_below_F"] = int((Ph[arm] < Ph["F"]).sum()); r["phi_flag_minus_F"] = float((Ph[arm] - Ph["F"]).mean())
        lam = (cost[arm] - CE1) / (CF - CE1); lam_nf = (cost_nf[arm] - cost_nf["E1"]) / (cost_nf["F"] - cost_nf["E1"])
        for nm, g in (("gap_line", (A[arm] - (A["E1"] + lam * (A["F"] - A["E1"]))) * 100), ("gap_line_nofront", (A[arm] - (A["E1"] + lam_nf * (A["F"] - A["E1"]))) * 100),
                      ("gap_line_f1", (F1[arm] - (F1["E1"] + lam * (F1["F"] - F1["E1"]))) * 100)):
            m_, lo, hi, up, dn = nb(g); r.update({nm: m_, nm + "_lo": lo, nm + "_hi": hi, nm + "_folds_up": up})
    summ.append(r)
S = pd.DataFrame(summ); S.to_csv(out / "arms_summary.csv", index=False)
s = S[S.arm == "LIB_energy25"].iloc[0]; Fr = S[S.arm == "F"].iloc[0]
rules = dict(primary_arm="LIB_energy25 (ESC-50, secondary, no verdict)", acc_F=float(Fr.acc), acc_E1=float(S[S.arm == "E1"].acc.iloc[0]), acc_M=float(s.acc), gap_F_minus_M=float(-s.acc_minus_F),
             a1_within_2=bool(-s.acc_minus_F <= 2.0), line_gap_points=float(s.gap_line), a2_above_line=bool(s.gap_line > 0), line_gap_nofront=float(s.gap_line_nofront), line_gap_f1=float(s.gap_line_f1),
             oracle_gain_vs_F=float(s.oracle_gain_vs_F), b1_oracle_ge_1=bool(s.oracle_gain_vs_F >= 1.0), phi_M=float(s.phi_flag_with_E1), phi_F=float(Fr.phi_flag_with_E1),
             phi_folds_below_F=int(s.phi_flag_folds_below_F), b2_phi=bool(s.phi_flag_minus_F < 0 and s.phi_flag_folds_below_F >= 4),
             note="folds 5: the 7-of-10 fold count of b2 is scaled to 4 of 5")
(out / "rules.json").write_text(json.dumps(rules, indent=1)); print(json.dumps(rules, indent=1))
