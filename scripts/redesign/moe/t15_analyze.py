"""T15 and T18 analysis (ANALYSIS_PLAN_T15.md section 8, ANALYSIS_PLAN_T18.md section 4): fold-level contrasts, Nadeau-Bengio, Holm (family 15), hull, error correlations,
escalation, verdicts. Reads cell_fold_acc.csv and phi_fold.csv of t15_score.py and the pre-selections.   python t15_analyze.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t15_common import *
OUT = R / "tables"; OUT.mkdir(exist_ok=True)
import glob
S = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "score_parts/*/cell_fold_acc.csv")))]); PH = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "score_parts/*/phi_fold.csv")))])
DONE = sorted(S.task.unique()); print("tasks scored:", DONE)
SV = json.load(open(R / "precond/survivors.json")); P15 = json.load(open(R / "inputs/t15_preselected.json")); P18 = json.load(open(R / "inputs/t18_preselected.json"))
cols = [c for c in ["acc", "f1", "n", "cost", "esc1", "esc2", "oracle"] if c in S.columns]
FA = S.groupby(["task", "cell", "kind", "system", "fold"])[cols].mean().reset_index()      # seeds averaged inside a fold
AF = {k: g.sort_values("fold") for k, g in FA.groupby(["task", "cell", "kind", "system"])}
def arr(task, cell, kind, system, col="acc"):
    g = AF.get((task, cell, kind, system)); return None if g is None or col not in g or g[col].isna().all() else g[col].to_numpy()
def ratio_of(task, pool):
    if task == "tau":
        nA = AF[("tau", "clean", "single", "mn10")]["n"].to_numpy(); ns = AF[("tau", "shift", "single", "mn10")]["n"].to_numpy(); return float(np.mean(ns / (nA.sum() - nA)))
    n = AF[(task, pool[0], "single", "mn10")]["n"].to_numpy(); return float(np.mean(n / (n.sum() - n)))
MAJ = lambda K: K // 2 + 1
def pooled(task, pool, fn): return np.mean([fn(c) for c in pool], 0)
def hull_vec(task, cell, cost, univ):
    """fold accuracies of the time-sharing hull of the singles at `cost` in `cell` (Stage 2 hull_pair on the cell-mean accuracy)."""
    CT = costs(task); av = [e for e in univ if arr(task, cell, "single", e) is not None]
    sp = pd.DataFrame([dict(system=e, f1=arr(task, cell, "single", e).mean(), cost=CT[e]) for e in av]); v, lo, hi, lam = hull_value(sp, cost)
    return (arr(task, cell, "single", lo) if lam == 0 else (1 - lam) * arr(task, cell, "single", lo) + lam * arr(task, cell, "single", hi)), (lo, hi, lam)
def univ_of(task): return [e for e in ["mn10", "mn20", "astfp32", "astw4", "astshort"] if not (task == "tau" and e == "astshort")] + SV["views"][task] + ([SV["lr"]] if SV.get("lr") else [])
def pool_of(task): return POOL[task]
rows15, rows18, extra = [], [], {}
for task in DONE:
    pool = pool_of(task); K = TASKS[task]["nf"]; ratio = ratio_of(task, pool); CT = costs(task); univ = [e for e in univ_of(task) if all(arr(task, c, "single", e) is not None for c in pool)]
    sel = P15[task]["mixture"]; cost = P15[task]["cost"]; bs = P15[task]["best_single"]
    mix = pooled(task, pool, lambda c: arr(task, c, "mix", sel)); best = pooled(task, pool, lambda c: arr(task, c, "single", bs))
    hl = pooled(task, pool, lambda c: hull_vec(task, c, cost, univ)[0]); pair = pooled(task, pool, lambda c: arr(task, c, "mix", cname((E1, E2))))
    dB, dH, dP = nb(mix - best, ratio), nb(mix - hl, ratio), nb(mix - pair, ratio)
    members = sel.split("+"); newm = [e for e in members if is_new(e)]
    # phi clause
    def phi_vec(a_, b_, cells):
        g = PH[(PH.task == task) & (PH.cell.isin(cells)) & (((PH.a == a_) & (PH.b == b_)) | ((PH.a == b_) & (PH.b == a_)))]
        return None if g.empty else g.groupby(["cell", "fold"]).phi.mean().groupby("fold").mean().to_numpy()
    ph_ast = phi_vec(E1, E2, pool); clause = {}
    for v in newm:
        pv = phi_vec(E1, v, pool); clause[v] = dict(phi_view=float(np.nanmean(pv)), phi_ast=float(np.nanmean(ph_ast)), folds_lower=int((pv < ph_ast).sum()), ok=bool((pv < ph_ast).sum() >= MAJ(K)))
    phi_ok = bool(newm) and all(c["ok"] for c in clause.values())
    rows15.append(dict(task=task, mixture=sel, cost=cost, new_views=",".join(newm), mix_acc=mix.mean(), best_single=bs, best_single_cost=CT[bs], best_single_acc=best.mean(), hull_acc=hl.mean(), pair_acc=pair.mean(),
                       d_best=dB["mean"], d_best_lo=dB["lo"], d_best_hi=dB["hi"], p_best=dB["p"], folds_best_ge_1=int(((mix - best) >= 1).sum()), d_hull=dH["mean"], d_hull_lo=dH["lo"], d_hull_hi=dH["hi"], p_hull=dH["p"],
                       folds_hull_ge_1=int(((mix - hl) >= 1).sum()), d_pair=dP["mean"], p_pair=dP["p"], n_folds=K, p_max=max(dB["p"], dH["p"]), phi_clause=phi_ok, phi_detail=json.dumps(clause),
                       a_mean=bool(dB["mean"] >= 1 and dH["mean"] >= 1), b_folds=bool(((mix - best) >= 1).sum() >= MAJ(K) and ((mix - hl) >= 1).sum() >= MAJ(K)),
                       fold_d_best=np.round(mix - best, 2).tolist(), fold_d_hull=np.round(mix - hl, 2).tolist()))
    # ---------- T18 ----------
    if task in P18 and SV.get("lr"):
        o = P18[task]; q = f"q{o['q1']}_{o['q2']}"; tiers = o["tiers"]; hc = pooled(task, pool, lambda c: arr(task, c, "hier", q)); co = np.mean([arr(task, c, "hier", q, "cost").mean() for c in pool])
        a3 = pooled(task, pool, lambda c: arr(task, c, "avg3", "+".join(tiers))); hl = pooled(task, pool, lambda c: hull_vec(task, c, float(arr(task, c, "hier", q, "cost").mean()), univ)[0])
        ivs = P15[task]["iv_singles"]; cand = [e for e in univ if CT[e] <= co * (1 + 1e-12) and e in ivs]; bs18 = max(cand, key=lambda e: ivs[e]) if cand else None
        bst = pooled(task, pool, lambda c: arr(task, c, "single", bs18)) if bs18 else None
        dH, d3 = nb(hc - hl, ratio), nb(hc - a3, ratio); dB = nb(hc - bst, ratio) if bst is not None else dict(mean=np.nan, lo=np.nan, hi=np.nan, p=1.0)
        e1p = pooled(task, pool, lambda c: arr(task, c, "hier", q, "esc1")); e2p = pooled(task, pool, lambda c: arr(task, c, "hier", q, "esc2"))
        e1c, e2c = arr(task, "clean", "hier", q, "esc1"), arr(task, "clean", "hier", q, "esc2"); r1, r2 = nb(e1p - e1c, ratio), nb(e2p - e2c, ratio)
        esc_ok = bool((e1p - e1c > 0).sum() >= MAJ(K) and (e2p - e2c > 0).sum() >= MAJ(K) and r1["p"] < .05 and r2["p"] < .05)
        pr = pooled(task, pool, lambda c: arr(task, c, "mix", cname((E1, E2))))
        rows18.append(dict(task=task, setting=q, tiers="+".join(tiers), cost=co, hier_acc=hc.mean(), hull_acc=hl.mean(), best_single=bs18, best_single_acc=(bst.mean() if bst is not None else np.nan), avg3_acc=a3.mean(), avg3_cost=sum(CT[t] for t in tiers),
                           pair_acc=pr.mean(), d_hull=dH["mean"], d_hull_lo=dH["lo"], d_hull_hi=dH["hi"], p_hull=dH["p"], folds_hull_ge_1=int(((hc - hl) >= 1).sum()), d_best=dB["mean"], p_best=dB["p"],
                           folds_best_ge_1=int(((hc - bst) >= 1).sum()) if bst is not None else 0, d_avg3=d3["mean"], d_avg3_lo=d3["lo"], d_avg3_hi=d3["hi"], p_avg3=d3["p"], folds_avg3_ge_1=int(((hc - a3) >= 1).sum()),
                           esc1_clean=e1c.mean(), esc1_shift=e1p.mean(), esc1_rise=r1["mean"], p_esc1=r1["p"], esc2_clean=e2c.mean(), esc2_shift=e2p.mean(), esc2_rise=r2["mean"], p_esc2=r2["p"], escalation_rises=esc_ok, n_folds=K,
                           p_max=max(dH["p"], dB["p"], d3["p"]), a_mean=bool(dH["mean"] >= 1 and dB["mean"] >= 1 and d3["mean"] >= 1),
                           b_folds=bool(((hc - hl) >= 1).sum() >= MAJ(K) and (bst is not None and ((hc - bst) >= 1).sum() >= MAJ(K)) and ((hc - a3) >= 1).sum() >= MAJ(K)),
                           best_setting_vs_pair=max(float(np.mean([pooled(task, pool, lambda c: arr(task, c, "hier", f"q{q1}_{q2}")).mean() for _ in [0]])) - pr.mean() for q1 in (0.1, 0.3, 0.5) for q2 in (0.1, 0.3, 0.5))))
P = pd.DataFrame(rows15)
P["p_holm_family"] = holm_family(P.p_max.to_numpy(), FAMILY); P["p_holm_within3"] = holm_family(P.p_max.to_numpy(), len(P))
P["passes"] = P.a_mean & P.b_folds & P.phi_clause & (P.p_holm_family < .05); P["passes_within3"] = P.a_mean & P.b_folds & P.phi_clause & (P.p_holm_within3 < .05)
P.to_csv(OUT / "t15_primary.csv", index=False)
Sa = bool((P.d_pair <= 0.5).all()); Sb = not any(SV["views"][t] for t in DONE)
v15 = "STOP (no view survives)" if Sb else ("PASS" if P.passes.sum() >= 2 else ("STOP (S-a)" if Sa else "NO PASS"))
J15 = dict(verdict=v15, n_tasks_pass=int(P.passes.sum()), n_tasks_pass_within3=int(P.passes_within3.sum()), passes_within_T15_only=bool(P.passes_within3.sum() >= 2 and P.passes.sum() < 2), S_a=Sa, S_b=Sb, tasks=P.drop(columns=["phi_detail"]).to_dict("records"))
json.dump(J15, open(OUT / "t15_verdict.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
if rows18:
    Q = pd.DataFrame(rows18); Q["p_holm_family"] = holm_family(Q.p_max.to_numpy(), FAMILY); Q["p_holm_within3"] = holm_family(Q.p_max.to_numpy(), len(Q))
    Q["passes"] = Q.a_mean & Q.b_folds & Q.escalation_rises & (Q.p_holm_family < .05); Q["passes_within3"] = Q.a_mean & Q.b_folds & Q.escalation_rises & (Q.p_holm_within3 < .05); Q.to_csv(OUT / "t18_primary.csv", index=False)
    Sb18 = bool((Q.best_setting_vs_pair <= 0.5).all())
    v18 = "PASS" if Q.passes.sum() >= 2 else ("STOP (S-b)" if Sb18 else "NO PASS")
    J18 = dict(verdict=v18, n_tasks_pass=int(Q.passes.sum()), n_tasks_pass_within3=int(Q.passes_within3.sum()), passes_within_T18_only=bool(Q.passes_within3.sum() >= 2 and Q.passes.sum() < 2), S_b=Sb18, tasks=Q.to_dict("records"))
else:
    J18 = dict(verdict="STOP (precondition: no low-resolution candidate qualifies)", tasks=[]); Q = pd.DataFrame()
json.dump(J18, open(OUT / "t18_verdict.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
# ---------- descriptive tables: every system on every cell, every mixture pooled, phi, oracles ----------
LC = FA.groupby(["task", "cell", "kind", "system"])[[c for c in ["acc", "f1", "cost", "esc1", "esc2", "oracle"] if c in FA.columns]].mean().reset_index(); LC.to_csv(OUT / "all_systems_by_cell.csv", index=False)
pm = []
for task in DONE:
    pool = pool_of(task); ratio = ratio_of(task, pool); univ = [e for e in univ_of(task) if all(arr(task, c, "single", e) is not None for c in pool)]; bs = P15[task]["best_single"]; best = pooled(task, pool, lambda c: arr(task, c, "single", bs))
    for sys_ in sorted({k[3] for k in AF if k[0] == task and k[2] == "mix"}):
        if any(arr(task, c, "mix", sys_) is None for c in pool): continue
        m = pooled(task, pool, lambda c: arr(task, c, "mix", sys_)); cost = mean_cost(task, sys_.split("+")); hl = pooled(task, pool, lambda c: hull_vec(task, c, cost, univ)[0])
        orc = np.mean([arr(task, c, "mix", sys_, "oracle").mean() for c in pool]); cl = arr(task, "clean", "mix", sys_)
        pm.append(dict(task=task, mixture=sys_, size=len(sys_.split("+")), cost=cost, pooled_shift_acc=m.mean(), clean_acc=cl.mean() if cl is not None else np.nan, hull_acc=hl.mean(), d_hull=nb(m - hl, ratio)["mean"], p_hull=nb(m - hl, ratio)["p"],
                       oracle_acc=orc, has_new=any(is_new(e) for e in sys_.split("+"))))
pd.DataFrame(pm).to_csv(OUT / "all_mixtures_pooled_shift.csv", index=False)
ph = PH.groupby(["task", "cell", "a", "b"]).phi.mean().reset_index(); ph.to_csv(OUT / "phi_by_cell.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print(P.drop(columns=["phi_detail", "fold_d_best", "fold_d_hull"]).to_string()); print(json.dumps({k: v for k, v in J15.items() if k != "tasks"}, indent=1))
if len(Q): print(Q.to_string()); print(json.dumps({k: v for k, v in J18.items() if k != "tasks"}, indent=1))
