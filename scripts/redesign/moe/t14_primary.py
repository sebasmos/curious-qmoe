"""T14 primary analysis (ANALYSIS_PLAN_T14.md sections 3, 4, 6, 7). Reads the pair file of t14_pair.py, the stored static.csv, best_single.json. python t14_primary.py --r RESULTS_DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t14_pair import PAIR, S3C, FINAL, CORR, NOISE, TASKS
import t14_eval as E, t14_stats as S
from t14_primary_data import static, single, FMAP
ap = argparse.ArgumentParser(); ap.add_argument("--r", required=True, type=Path); a = ap.parse_args(); R = a.r
BS = json.load(open(R / "inputs/best_single.json")); BV = json.load(open(R / "inputs/best_single_v2.json")); PF = pd.read_csv(R / "pair/pair_cell_fold.csv")
def ser_pair(task, col, cells=None):
    g = PF[PF.task == task]; g = g[g.cell.isin(cells)] if cells else g
    return g[["cell", "seed", "fold"]].assign(f1=g[col].to_numpy())
def pair_cost(task, cell_rate):  # cheap + rate * ast (stage 3c convention), checked against cascades.csv.gz
    c = json.load(open(S3C / f"analysis/{task}/costs.json")); return c["cheap"][PAIR[task]["cn"]], c["ast"][PAIR[task]["an"].replace("ast:", "")]
rows_s, rows_h, tests = [], [], []
cmp_rows = []
verdict = {}
for task in PAIR:
    T = TASKS[task]; nbr = 1.0 / (T["nf"] - 1); P = PAIR[task]; cells = BS[task]["cells"]
    cc, ca = pair_cost(task, None)
    rate = PF[PF.task == task].groupby("cell").rate.mean(); cost_cell = cc + rate * ca
    lim = float(cost_cell.loc[cells].max())
    bclean, bfield, bshift = (BV[task][f"main:{f}"]["best"] for f in ("clean", "field", "shift"))
    allsc = BV[task]["main:shift"]["all_scores"]; best_sens = sorted([s for s, (v, c) in allsc.items() if c <= lim * (1 + 1e-12)], key=lambda s: (-allsc[s][0], s))[0]
    fp32 = {"esc50": "ast:fp32@512+esc", "esc5": "ast:fp32+esc", "urban": "ast:fp32+esc"}[task]; w4 = {"esc50": "ast:w4g128@512+esc", "esc5": "ast:w4g128+esc", "urban": "ast:w4g128+esc"}[task]
    comps = {"best_single_clean": (single(task, bclean), "primary"), "best_single_field": (single(task, bfield), "primary"), "best_single_shift": (single(task, bshift), "primary"),
             "best_single_v1_cleaniv": (single(task, BS[task]["best_single"]), "sensitivity"), "best_single_shift_max_cost": (single(task, best_sens), "sensitivity"),
             "always_average": (ser_pair(task, "avg"), "secondary"), "cheap_expert": (ser_pair(task, "cheap"), "secondary"), "strong_expert": (ser_pair(task, "strong"), "secondary"),
             "ast_fp32_same_frames": (single(task, fp32), "secondary"), "ast_4bit_same_frames": (single(task, w4), "secondary")}
    if task == "esc50": comps["best_single_additive14_shift"] = (single(task, BV[task]["additive14:shift"]["best"]), "sensitivity")
    pair = ser_pair(task, "pair")
    verdict[task] = dict(best_single_clean=bclean, best_single_field=bfield, best_single_shift=bshift, best_single_v1=BS[task]["best_single"], best_single_sens=best_sens, cost_limit_sens=lim, pair_cost_by_cell={k: float(v) for k, v in cost_cell.items()}, cells=cells)
    for cname, (ref, role) in comps.items():
        if ref.empty: rows_s.append(dict(task=task, comparator=cname, role=role, note="not stored")); continue
        for cond in ("clean", "field"):
            d = E.descr(ref, cond); d0 = E.descr(pair, cond); s = E.sampling(pair, ref, cond)
            rows_s.append(dict(task=task, comparator=cname, role=role, cond=cond, pair_mean=d0["mean"], pair_sd_fold=d0["sd_fold"], pair_sd_seed=d0["sd_seed"], pair_sd_inter=d0["sd_inter"],
                               cmp_mean=d["mean"], cmp_sd_fold=d["sd_fold"], cmp_sd_seed=d["sd_seed"], cmp_sd_inter=d["sd_inter"], **(s or {})))
        sh = E.shift(pair, ref, nbr)
        rows_h.append(dict(task=task, comparator=cname, role=role, **(sh or dict(note="fewer than 3 shared cells"))))
        # comparator on the additive 14 cells for esc50 sensitivity
    if task == "esc50":
        a14 = BS[task]["cells_additive14"]; ref = comps["best_single_additive14_shift"][0]; sh = E.shift(pair[pair.cell.isin(a14)], ref[ref.cell.isin(a14)], nbr)
        rows_h.append(dict(task=task, comparator="best_single_additive14_shift_on_14_cells", role="sensitivity", **sh))
    # per-system descriptive row for the pair and every comparator (their own measures)
S_df, H_df = pd.DataFrame(rows_s), pd.DataFrame(rows_h)
# primary tests and Holm over the 12
prim = []
for task in PAIR:
    for cond in ("clean", "field"):
        r = S_df[(S_df.task == task) & (S_df.comparator == f"best_single_{cond}") & (S_df.cond == cond)].iloc[0]; prim.append(dict(task=task, notion=1, test=f"BF variance {cond}", p=r.bf_p, effect=r.ratio_fold, lo=r.ratio_fold_lo, hi=r.ratio_fold_hi))
    h = H_df[(H_df.task == task) & (H_df.comparator == "best_single_shift")].iloc[0]
    prim.append(dict(task=task, notion=2, test="worst-case F1 difference", p=h.dworst_p, effect=h.dworst, lo=h.dworst_lo, hi=h.dworst_hi))
    prim.append(dict(task=task, notion=2, test="average drop difference", p=h.ddrop_p, effect=h.ddrop, lo=h.ddrop_lo, hi=h.ddrop_hi))
Pm = pd.DataFrame(prim); Pm["holm12"] = S.holm(Pm.p.to_numpy()); Pm["holm12_x15"] = np.minimum(1, Pm.holm12 * 15)
for n in (1, 2):
    m = Pm.notion == n; Pm.loc[m, "holm_within_notion"] = S.holm(Pm.loc[m, "p"].to_numpy())
# verdicts
v1, v2 = {}, {}
for task in PAIR:
    ok1 = all(S_df[(S_df.task == task) & (S_df.comparator == f"best_single_{c}") & (S_df.cond == c)].sampling_label.iloc[0] == "more stable" for c in ("clean", "field")); v1[task] = bool(ok1)
    q = Pm[(Pm.task == task) & (Pm.notion == 2)]; dw = q[q.test.str.startswith("worst")].iloc[0]; dd = q[q.test.str.startswith("average")].iloc[0]
    v2[task] = bool(dw.effect >= 0.5 and dd.effect >= 0.5 and dw.holm12 < 0.05 and dd.holm12 < 0.05)
verdict["notion1_task_holds"] = v1; verdict["notion1_holds"] = sum(v1.values()) >= 2; verdict["notion2_task_holds"] = v2; verdict["notion2_holds"] = sum(v2.values()) >= 2
out = R / "tables"; out.mkdir(exist_ok=True)
S_df.to_csv(out / "primary_sampling.csv", index=False); H_df.to_csv(out / "primary_shift.csv", index=False); Pm.to_csv(out / "primary_tests.csv", index=False)
json.dump(verdict, open(out / "primary_verdict.json", "w"), indent=1, default=str)
pd.set_option("display.width", 250); print(Pm.round(4).to_string()); print(json.dumps({k: v for k, v in verdict.items() if k.startswith("notion")})); print(S_df[S_df.comparator.isin(["best_single_clean","best_single_field"])].round(3).to_string())
