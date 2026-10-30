"""Reviewer extras, task A: the always-average static ensemble next to the stage 3c cascades and the time-sharing hull.

For each stage 3c pick (cheap tier + AST tier, rule avg) the static ensemble averages both tiers' softmax outputs on every
clip. Its cost is the cheap tier plus the AST tier (both run on every clip). It joins the static systems of the hull set
"new" of stage 3c (Stage 2, 3b and 3c statics). The hull is rebuilt with Stage 2's hull_pair at the cascade's mean cost,
the cascade-minus-hull differences are tested at the fold level (seeds averaged inside folds, two-sided t over folds,
Nadeau-Bengio) with Holm over the three tasks, as in analyze_3c.py. Without the ensemble the script must reproduce the
hull_diff and p values stored in <s3c>/analysis/<task>/selected.csv (checked, written to repro_check.csv).

    python a_ensemble_hull.py --s3c S3C --s3b S3B --s2 S2 --i5 I5 --out OUT
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT, ROOT / "scripts/redesign/idea6/stage2", ROOT / "scripts/redesign/idea6/stage3b", ROOT / "scripts/redesign/idea6/stage3c"):
    sys.path.insert(0, str(p))
import analyze as A  # noqa: E402
from analyze_3b import TASKS  # noqa: E402
from idea5b_common import mix_expected_f1, holm  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("s3c", "s3b", "s2", "i5", "out"):
    ap.add_argument("--" + k, required=True, type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

# designs: (label, task, [cheap_src, bb, head, tier, rule, budget], primary)
def posthoc_urban():
    d = pd.read_csv(a.s3c / "analysis/urban/selected.csv")
    r = d[(d.family == "S4") & (d.condition == "field")].iloc[0]
    return [r.cheap_src, r.cheap_bb, str(r.cheap), r.expensive, r.rule, r.budget]

DESIGNS = []
for t in ("esc50", "esc5", "urban"):
    DESIGNS.append((f"{t} P3 (pre-registered)", t, [str(v) for v in json.load(open(a.s3c / f"analysis/{t}/picks.json"))["P3"]["design"]], True))
DESIGNS.append(("urban S4 post hoc (AST-Q8 400 frames)", "urban", posthoc_urban(), False))

CNN_NEW = {"mn05", "mn20", "dymn04"}
cache = {}

def sdir(system, task, seed):
    """(ScoreDir, config) of one static system."""
    key = (system, task, seed)
    if key in cache:
        return cache[key]
    T = TASKS[task]
    SL = T["short"]
    parts = system.split(":")
    if parts[0] == "eff":
        _, bb, k = parts
        r = (A.ScoreDir(a.s2 / "scores" / task / f"effp_{task}_s{seed}" / k), bb)
    elif parts[0] == "cnn":
        _, m, bb, k = parts
        root = (a.s3c if m in CNN_NEW else a.s3b) / "scores_cnn" / task
        r = (A.ScoreDir(root / f"{m}_{task}_s{seed}" / k), f"{m}_{bb}")
    else:
        n, h = system[4:].split("+")
        if "@" in n:
            base, L = n.split("@")
            assert int(L) == SL, system
            r = (A.ScoreDir(a.s2 / "scores_ast_short" / task / f"astshort_{task}_s{seed}" / h), base)
        else:
            root = a.i5 / "scores" / task
            r = (A.ScoreDir(root / f"ast{task}_s{seed}" / h), n)
    cache[key] = r
    return r

def logits(system, task, seed, cond):
    d, cfg = sdir(system, task, seed)
    return d.get(cfg, cond)

def prob(system, task, seed, cond):
    r = logits(system, task, seed, cond)
    return r, A.softmax(r["z"])

def cheap_name(src, bb, k):
    return f"eff:{bb}:{k}" if src == "eff" else f"cnn:{src}:{bb}:{k}"

rows, repro = [], []
for label, task, des, primary in DESIGNS:
    T = TASKS[task]; K, seeds = T["K"], T["seeds"]
    src, bb, k, tier, rule, bud = des
    assert rule == "avg"
    cn, an = cheap_name(src, bb, k), f"ast:{tier}+esc"
    costs = json.load(open(a.s3c / f"analysis/{task}/costs.json"))
    c_cheap = costs["cheap"][cn]; c_ast = costs["ast"][tier + "+esc"]
    c_ens = c_cheap + c_ast
    S = pd.read_csv(a.s3c / f"analysis/{task}/static.csv")
    S = S[S.set.isin(["stage3b", "new"])]
    CA = pd.read_csv(a.s3c / f"analysis/{task}/cascades.csv.gz")
    sel = pd.read_csv(a.s3c / f"analysis/{task}/selected.csv")
    ref0 = logits("eff:fp32:esc", task, seeds[0], "clean")
    folds = sorted(set(ref0["fold"]))
    n_te = np.array([np.sum(ref0["fold"] == f) for f in folds])
    A.NB_RATIO["task"] = float(np.mean(n_te / (len(ref0["y"]) - n_te)))
    for cond in ("clean", "field"):
        g = CA[(CA.cheap_src == src) & (CA.cheap_bb == bb) & (CA.cheap.astype(str) == k) & (CA.expensive == tier) &
               (CA.rule == rule) & (CA.budget == bud) & (CA.condition == cond)].sort_values(["seed", "fold"])
        keys = list(zip(g.seed, g.fold)); assert len(keys) == len(seeds) * len(folds)
        casc_f1 = g.f1.to_numpy(); cost = float(g.cost.mean())
        # ensemble predictions and F1
        pred_ens, ys, ens_f1 = {}, {}, []
        for s in seeds:
            rc, pc = prob(cn, task, s, cond); ra, pa = prob(an, task, s, cond)
            for f in folds:
                te = np.flatnonzero(rc["fold"] == f)
                pr = ((pc[te] + pa[te]) / 2).argmax(1)
                pred_ens[(s, f)] = pr.astype(np.int64); ys[(s, f)] = rc["y"][te]
        ens_f1 = np.array([A.f1_fast(ys[kk], pred_ens[kk], K) for kk in keys])
        sp0 = S[S.condition == cond].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
        sp1 = pd.concat([sp0, pd.DataFrame([{"system": "ENSEMBLE", "f1": float(ens_f1.mean()), "cost": c_ens}])], ignore_index=True)
        s_rec = {sy: dict(zip(zip(gg.seed, gg.fold), gg.f1)) for sy, gg in S[S.condition == cond].groupby("system")}

        def pred_of(system, kk):
            if system == "ENSEMBLE":
                return pred_ens[kk]
            r = logits(system, task, kk[0], cond)
            te = np.flatnonzero(r["fold"] == kk[1])
            return r["z"][te].argmax(1).astype(np.int64)

        def hull(spx):
            v, lo, hi, lam = A.hull_pair(spx, cost)
            out = []
            for kk in keys:
                if lam == 0.0:
                    out.append(ens_f1[keys.index(kk)] if lo == "ENSEMBLE" else s_rec[lo][kk])
                else:
                    out.append(float(mix_expected_f1(ys[kk], pred_of(lo, kk), pred_of(hi, kk), lam, K)[0]))
            return v, lo, hi, lam, np.array(out)

        res = {}
        for name, spx in (("without", sp0), ("with", sp1)):
            v, lo, hi, lam, h = hull(spx)
            t = A.tests(casc_f1, h, keys)
            res[name] = dict(hull_lo=lo, hull_hi=hi, lam=lam, hull_f1=float(h.mean()), diff=t["diff"], fold_min=t["diff_fold_min"],
                             fold_max=t["diff_fold_max"], p_t=t["p_t_fold"], p_nb=t["p_nb"], hull_arr=h)
        # reproduction of the stored 3c hull (family P3 or S4)
        fam = "P3" if primary else "S4"
        st = sel[(sel.family == fam) & (sel.condition == cond)].iloc[0]
        repro.append(dict(design=label, condition=cond, stored_hull_diff=st.hull_diff, mine=res["without"]["diff"],
                          stored_p_t=st.hull_p_t_fold, mine_p_t=res["without"]["p_t"], stored_p_nb=st.hull_p_nb, mine_p_nb=res["without"]["p_nb"],
                          stored_cost=st.cost, mine_cost=cost, same_pair=(st.hull_lo, st.hull_hi) == (res["without"]["hull_lo"], res["without"]["hull_hi"])))
        # ensemble against cascade and against the AST tier alone / cheap alone
        t_ec = A.tests(casc_f1, ens_f1, keys)
        ast_f1 = np.array([s_rec[an][kk] for kk in keys]); ch_f1 = np.array([s_rec[cn][kk] for kk in keys])
        for name in ("without", "with"):
            r_ = res[name]
            rows.append(dict(design=label, task=task, primary=primary, condition=cond, hull_set=name, cascade_f1=float(casc_f1.mean()),
                             cascade_cost=cost, cascade_rate=float(g.rate.mean()), ens_f1=float(ens_f1.mean()), ens_cost=c_ens,
                             cheap_f1=float(ch_f1.mean()), cheap_cost=c_cheap, ast_f1=float(ast_f1.mean()), ast_cost=c_ast,
                             cascade_minus_ens=t_ec["diff"], p_t_casc_ens=t_ec["p_t_fold"],
                             hull_lo=r_["hull_lo"], hull_hi=r_["hull_hi"], lam=r_["lam"], hull_f1=r_["hull_f1"], diff=r_["diff"],
                             fold_min=r_["fold_min"], fold_max=r_["fold_max"], p_t=r_["p_t"], p_nb=r_["p_nb"]))
        print(label, cond, {n: round(res[n]["diff"] * 100, 2) for n in res}, res["with"]["hull_lo"], res["with"]["hull_hi"], flush=True)

R = pd.DataFrame(rows)
# Holm over the three tasks (primary designs; and the variant with the post hoc urban design in place of the pre-registered one)
R["holm_t"] = np.nan; R["holm_nb"] = np.nan; R["holm_family"] = ""
for fam, labels in (("primary", [d[0] for d in DESIGNS[:3]]), ("urban post hoc in place", [DESIGNS[0][0], DESIGNS[1][0], DESIGNS[3][0]])):
    for cond in ("clean", "field"):
        for hs in ("without", "with"):
            m = (R.design.isin(labels)) & (R.condition == cond) & (R.hull_set == hs)
            sub = R[m]
            if fam == "primary":
                R.loc[m, "holm_t"] = holm(sub.p_t.to_numpy()); R.loc[m, "holm_nb"] = holm(sub.p_nb.to_numpy()); R.loc[m, "holm_family"] = fam
            else:
                h_t, h_n = holm(sub.p_t.to_numpy()), holm(sub.p_nb.to_numpy())
                R.loc[m & R.design.str.startswith("urban S4"), "holm_t"] = h_t[list(sub.design).index(DESIGNS[3][0])]
                R.loc[m & R.design.str.startswith("urban S4"), "holm_nb"] = h_n[list(sub.design).index(DESIGNS[3][0])]
                R.loc[m & R.design.str.startswith("urban S4"), "holm_family"] = fam
R.to_csv(a.out / "a_ensemble_hull.csv", index=False)
pd.DataFrame(repro).to_csv(a.out / "a_repro_check.csv", index=False)
print(pd.DataFrame(repro).to_string())
