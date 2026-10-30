"""Check 1, 6, 7 and 10: recompute selection, H1 and H2 from per-clip logits with the auditor's own code,
then the nested variant, an enlarged hull, selection-rule sensitivity and extra budgets.

    python -B recompute.py esc50|esc5|urban

Writes outputs/recompute_<task>/: selection.csv (every design's clean inner-validation gain), tests.json
(H1, H2, nested, enlarged hull, sensitivity, cost views), fold_diffs.json, static_field.csv.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AST_HEADS, BACKBONES, BUDGETS, CELLS, FULL_TIERS, KINDS, OUT, R, RULES, S2, SHORT_CFGS,  # noqa
                    TASKS, Run, f1m, fold_means, fold_tests, hull_pair, margin, mix_f1_exact, mix_f1_mc, sm,
                    threshold)

task = sys.argv[1]
T = TASKS[task]
K, seeds, SL = T["K"], T["seeds"], T["SL"]
od = OUT / f"recompute_{task}"
od.mkdir(parents=True, exist_ok=True)
t0 = time.time()
EXTRA_B = {"b60": 0.6, "b70": 0.7, "b80": 0.8, "b90": 0.9}  # exploratory, check 10 only

# ── costs (auditor's recount, check 4) ─────────────────────────────────────
C = json.loads((OUT / "costs_audit.json").read_text())
claimed = json.loads((R / f"analysis/{task}/costs.json").read_text())
emb = T["emb"]
mel = C["mel_mean_bitops"][emb]
fb = C["fbank"][emb]
eff_heads = {k: v["bitops"] for k, v in claimed["eff_heads"].items()}  # heads taken from the claim (< 0.002 T)
ast_heads = {k: v["bitops"] for k, v in claimed["ast_heads"].items()}
EB = C["effnet_bitops"]


def cheap_cost(bb, k, view="priced"):
    if bb == "fp32":
        b = EB["fp32"] if view != "all" else EB["fp32_plus_all"]
    else:
        b = {"priced": EB["w8a8_priced"], "unpriced": EB["w8a8_unpriced"], "all": EB["w8a8_priced_plus_all"]}[view]
    return mel + b + eff_heads[k]


def ast_cost(name, h, view="priced"):
    cfg, _, L_ = name.partition("@")
    L_ = int(L_) if L_ else 1024
    key = f"{cfg}@{L_}"
    if key in C["ast_priced"]:
        b = C["ast_priced" if view != "unpriced" else "ast_unpriced"][key]
    else:  # configurations only in the 1024-frame static set; same formula
        from costs_audit import ast_bitops
        b = ast_bitops(cfg, L_, view != "unpriced")
    if view == "all":
        b += C["ast_elementwise_ops"][str(L_)]["fp32" if cfg == "fp32" else "quant"] * 1024
    return fb + b + ast_heads[h]


# ── runs ───────────────────────────────────────────────────────────────────
eff = {(s, k): Run(R / "scores" / task / T["eff"].format(s) / k) for s in seeds for k in KINDS}
full = {(s, h): Run(S2 / "scores" / task / T["ast"].format(s) / h) for s in seeds for h in AST_HEADS}
short = {(s, h): Run(R / "scores_ast_short" / task / T["short"].format(s) / h) for s in seeds for h in AST_HEADS}
src = {}
for p in sorted((S2 / "scores" / task / T["ast"].format(seeds[0]) / "esc").glob("*__clean.npz")):
    src[p.name.split("__")[0]] = (full, p.name.split("__")[0])
for c in SHORT_CFGS:
    src[f"{c}@{SL}"] = (short, c)


def ast_has(name, cond):
    d, base = src[name]
    return all(r.has(base, cond) for r in d.values())


def ast_get(name, s, h, cond):
    d, base = src[name]
    return d[(s, h)].get(base, cond)


clean_ast = [n for n in src if ast_has(n, "clean")]
field_ast = [n for n in src if ast_has(n, "field")]
tiers = [t for t in FULL_TIERS + [f"{c}@{SL}" for c in SHORT_CFGS] if t in field_ast]
assert len(tiers) == 13, tiers
designs = [(bb, k, e, r, b) for bb in BACKBONES for k in KINDS for e in tiers for r in RULES for b in BUDGETS]
print(task, "clean AST statics", len(clean_ast), "field AST statics", len(field_ast), "tiers", len(tiers),
      "designs", len(designs), flush=True)

# ── per (seed, fold) work ──────────────────────────────────────────────────
ref0 = eff[(seeds[0], "esc")].get("fp32", "clean")
folds = sorted(ref0["iv"])
n_all = len(ref0["y"])
iv_static = {}      # (cond, system, s, f) -> (f1, preds)
iv_design = []      # rows
test_static = {}    # (cond, system, s, f) -> (f1, preds)
test_design = {}    # (cond, design, s, f) -> (f1, rate)
yte, yiv = {}, {}
rng = np.random.default_rng(777)
U_iv = {}
checks = {"rows_equal": True, "n_files_checked": 0}

for s in seeds:
    ref = eff[(s, "esc")].get("fp32", "clean")
    for f in folds:
        te = np.flatnonzero(ref["fold"] == f)
        ivi = ref["iv"][f][0]
        y, yv = ref["y"][te], ref["y"][ivi]
        yte[(s, f)], yiv[(s, f)] = y, yv
        U_iv[(s, f)] = rng.random((400, len(ivi)))
        for cond in ("clean", "field"):
            # cheap systems
            cp, civ = {}, {}
            for bb in BACKBONES:
                for k in KINDS:
                    r = eff[(s, k)].get(bb, cond)
                    if not (np.array_equal(r["y"], ref["y"]) and np.array_equal(r["fold"], ref["fold"])
                            and np.array_equal(r["iv"][f][0], ivi)):
                        checks["rows_equal"] = False
                    checks["n_files_checked"] += 1
                    cp[(bb, k)] = sm(r["z"][te])
                    civ[(bb, k)] = sm(r["iv"][f][1])
            ap, aiv = {}, {}
            names = clean_ast if cond == "clean" else field_ast
            for n in names:
                for h in AST_HEADS:
                    r = ast_get(n, s, h, cond)
                    if not (np.array_equal(r["y"], ref["y"]) and np.array_equal(r["fold"], ref["fold"])
                            and np.array_equal(r["iv"][f][0], ivi)):
                        checks["rows_equal"] = False
                    checks["n_files_checked"] += 1
                    ap[(n, h)] = sm(r["z"][te])
                    aiv[(n, h)] = sm(r["iv"][f][1])
            # statics (single systems), test and inner validation
            for (bb, k), p in cp.items():
                pr = p.argmax(1)
                test_static[(cond, f"eff:{bb}:{k}", s, f)] = (f1m(y, pr, K), pr)
                pv = civ[(bb, k)].argmax(1)
                iv_static[(cond, f"eff:{bb}:{k}", s, f)] = (f1m(yv, pv, K), pv)
            for (n, h), p in ap.items():
                pr = p.argmax(1)
                test_static[(cond, f"ast:{n}+{h}", s, f)] = (f1m(y, pr, K), pr)
                pv = aiv[(n, h)].argmax(1)
                iv_static[(cond, f"ast:{n}+{h}", s, f)] = (f1m(yv, pv, K), pv)
            # check 6: ensembles as extra statics (test rows only)
            for bb in BACKBONES:
                for r_ in (2, 3, 4):
                    for sub in itertools.combinations(KINDS, r_):
                        pr = np.mean([cp[(bb, k)] for k in sub], 0).argmax(1)
                        test_static[(cond, f"ens:{bb}:{'+'.join(sub)}", s, f)] = (f1m(y, pr, K), pr)
            for n in names:
                pr = (ap[(n, "esc")] + ap[(n, "4")]).argmax(1)
                test_static[(cond, f"astens:{n}", s, f)] = (f1m(y, pr, K), pr)
                for bb in BACKBONES:
                    for k in KINDS:
                        if n in tiers:
                            pr = (cp[(bb, k)] + ap[(n, "esc")]).argmax(1)
                            test_static[(cond, f"both:{bb}:{k}:{n}", s, f)] = (f1m(y, pr, K), pr)
            # thresholds from CLEAN inner validation (same for both conditions)
            if cond == "clean":
                thr = {}
                for (bb, k), p in civ.items():
                    m = margin(p)
                    for b, rate in {**BUDGETS, **EXTRA_B}.items():
                        thr[(bb, k, b)] = threshold(m, rate)
                    ac = p.argmax(1)
                    for b in BUDGETS:
                        esc = m < thr[(bb, k, b)]
                        for e in tiers:
                            pe = aiv[(e, "esc")]
                            for rr in RULES:
                                de = pe.argmax(1) if rr == "ast" else (p + pe).argmax(1)
                                iv_design.append(dict(cheap_bb=bb, cheap=k, expensive=e, rule=rr, budget=b, seed=s,
                                                      fold=f, iv_rate=float(esc.mean()),
                                                      f1_iv=f1m(yv, np.where(esc, de, ac), K)))
            # field inner validation of each design (nested variant needs the design's iv cost only; skip)
            # test rows of every design
            for (bb, k), p in cp.items():
                m, ac = margin(p), p.argmax(1)
                for b in list(BUDGETS) + list(EXTRA_B):
                    esc = m < thr[(bb, k, b)]
                    rate = float(esc.mean())
                    for e in tiers:
                        pe = ap[(e, "esc")]
                        for rr in RULES:
                            if b in EXTRA_B and not (bb == "w8a8"):
                                continue
                            de = pe.argmax(1) if rr == "ast" else (p + pe).argmax(1)
                            test_design[(cond, (bb, k, e, rr, b), s, f)] = (f1m(y, np.where(esc, de, ac), K), rate)
    for d in list(eff.values()) + list(full.values()) + list(short.values()):
        d.clear()
    print(task, f"seed {s} done {time.time() - t0:.0f}s", flush=True)

json.dump(checks, open(od / "row_checks.json", "w"))
IVD = pd.DataFrame(iv_design)
n_te = np.array([len(yte[(seeds[0], f)]) for f in folds])
NB = float(np.mean(n_te / (n_all - n_te)))


def scost(name, view="priced"):
    kind, rest = name.split(":", 1)
    if kind == "eff":
        bb, k = rest.split(":")
        return cheap_cost(bb, k, view)
    if kind == "ast":
        n, h = rest.split("+")
        return ast_cost(n, h, view)
    if kind == "ens":
        bb, ks = rest.split(":")
        ks = ks.split("+")
        return cheap_cost(bb, ks[0], view) + sum(eff_heads[k] for k in ks[1:])
    if kind == "astens":
        return ast_cost(rest, "esc", view) + ast_heads["4"]
    if kind == "both":
        bb, k, n = rest.split(":")
        return cheap_cost(bb, k, view) + ast_cost(n, "esc", view)
    raise KeyError(name)


def dcost(design, rate, view="priced"):
    bb, k, e, _, _ = design
    return cheap_cost(bb, k, view) + rate * ast_cost(e, "esc", view)


# ── selection on clean inner validation ────────────────────────────────────
def iv_hull_value(lo, hi, lam, s, f, cond="clean"):
    flo, plo = iv_static[(cond, lo, s, f)]
    if lam == 0.0:
        return flo
    phi = iv_static[(cond, hi, s, f)][1]
    yv = yiv[(s, f)]
    if len(np.unique(yv)) == K:
        return mix_f1_exact(yv, plo, phi, lam, K)
    return mix_f1_mc(yv, plo, phi, lam, K, U_iv[(s, f)])


def select(view="priced", subset=None, keys=None):
    """Gain of every design over the clean inner-validation hull; keys restricts the (seed, fold) rows
    (nested variant: one fold). Returns a DataFrame sorted best first."""
    keys = keys or [(s, f) for s in seeds for f in folds]
    sysn = sorted({k[1] for k in iv_static if k[0] == "clean"})
    f1s = [np.mean([iv_static[("clean", n, s, f)][0] for s, f in keys]) for n in sysn]
    cs = [scost(n, view) for n in sysn]
    rows = []
    sub = IVD[IVD.apply(lambda r: (r.seed, r.fold) in set(keys), axis=1)] if len(keys) < len(seeds) * len(folds) else IVD
    for (bb, k, e, rr, b), g in sub.groupby(["cheap_bb", "cheap", "expensive", "rule", "budget"], sort=False):
        if subset and not subset((bb, k, e, rr, b)):
            continue
        g = g.sort_values(["seed", "fold"])
        civ_ = float(np.mean([dcost((bb, k, e, rr, b), x, view) for x in g.iv_rate]))
        v, lo, hi, lam = hull_pair(sysn, f1s, cs, civ_)
        hv = [iv_hull_value(lo, hi, lam, s, f) for s, f in zip(g.seed, g.fold)]
        rows.append(dict(cheap_bb=bb, cheap=k, expensive=e, rule=rr, budget=b, f1_iv=g.f1_iv.mean(),
                         iv_rate=g.iv_rate.mean(), cost_iv=civ_, hull_lo=lo, hull_hi=hi, lam=lam,
                         hull_iv=float(np.mean(hv)), gain_iv=float(np.mean(g.f1_iv.to_numpy() - np.array(hv)))))
    return pd.DataFrame(rows).sort_values(["gain_iv", "cost_iv"], ascending=[False, True]).reset_index(drop=True)


SEL = select("priced")
SEL.to_csv(od / "selection.csv", index=False)
print(task, "selection done", f"{time.time() - t0:.0f}s", SEL.iloc[:3][["cheap_bb", "cheap", "expensive", "rule",
                                                                          "budget", "gain_iv"]].to_string(), flush=True)


# ── test comparisons ───────────────────────────────────────────────────────
def static_table(cond, view="priced", extra=False):
    names = sorted({k[1] for k in test_static if k[0] == cond})
    if not extra:
        names = [n for n in names if n.split(":")[0] in ("eff", "ast")]
    f1s = [np.mean([test_static[(cond, n, s, f)][0] for s in seeds for f in folds]) for n in names]
    return names, f1s, [scost(n, view) for n in names]


def hull_test(cond, lo, hi, lam, s, f):
    if lam == 0.0:
        return test_static[(cond, lo, s, f)][0]
    return mix_f1_exact(yte[(s, f)], test_static[(cond, lo, s, f)][1], test_static[(cond, hi, s, f)][1], lam, K)


def compare(design, cond="field", view="priced", extra=False, pair_from=None):
    keys = [(s, f) for s in seeds for f in folds]
    x = np.array([test_design[(cond, design, s, f)][0] for s, f in keys])
    rates = np.array([test_design[(cond, design, s, f)][1] for s, f in keys])
    cost = float(np.mean([dcost(design, r_, view) for r_ in rates]))
    names, f1s, cs = static_table(cond, view, extra)
    v, lo, hi, lam = hull_pair(names, f1s, cs, cost)
    h = np.array([hull_test(cond, lo, hi, lam, s, f) for s, f in keys])
    fm = fold_means(x - h, [f for _, f in keys])
    out = dict(design=" ".join(design), cond=cond, view=view, extra_statics=extra, f1=float(x.mean()), cost_T=cost / 1e12,
               rate=float(rates.mean()), hull=f"{lo} / {hi} ({lam:.2f})", hull_f1=float(h.mean()),
               diff_pts=100 * float((x - h).mean()), fold_diffs_pts=[round(100 * v_, 3) for v_ in fm],
               **{k_: v_ for k_, v_ in fold_tests(fm, NB).items() if k_ != "mean"})
    return out, fm


res = {"task": task, "nb_ratio": NB, "n_designs": len(designs)}
claim_pick = json.loads((R / f"analysis/{task}/picks.json").read_text())["H1"]["design"]
pick = tuple(SEL.iloc[0][["cheap_bb", "cheap", "expensive", "rule", "budget"]])
res["pick_mine"] = list(pick)
res["pick_claim"] = claim_pick
res["pick_gain_iv"], res["runner_up_gain_iv"] = float(SEL.iloc[0].gain_iv), float(SEL.iloc[1].gain_iv)
res["pick_rank_of_claim"] = int(SEL.index[(SEL.cheap_bb == claim_pick[0]) & (SEL.cheap == claim_pick[1])
                                         & (SEL.expensive == claim_pick[2]) & (SEL.rule == claim_pick[3])
                                         & (SEL.budget == claim_pick[4])][0]) + 1
fold_store = {}
res["H1"], fold_store["H1"] = compare(pick, "field")
res["H2"], fold_store["H2"] = compare(pick, "clean")
# H3: FP32 cheap backbone only; H4: W8A8 only (own selections)
for fam, bb_ in (("H3", "fp32"), ("H4", "w8a8")):
    sel_ = SEL[SEL.cheap_bb == bb_].reset_index(drop=True)
    d_ = tuple(sel_.iloc[0][["cheap_bb", "cheap", "expensive", "rule", "budget"]])
    res[fam], fold_store[fam] = compare(d_, "field")
# E3 unpriced view (own selection)
SEL_U = select("unpriced")
d_ = tuple(SEL_U.iloc[0][["cheap_bb", "cheap", "expensive", "rule", "budget"]])
res["E3_unpriced"], fold_store["E3"] = compare(d_, "field", "unpriced")
print(task, "H1-H4 done", f"{time.time() - t0:.0f}s", flush=True)

# check 6: enlarged hull (cheap-head ensembles, AST two-head ensembles, cheap+AST mean on every clip)
res["H1_enlarged_hull"], fold_store["H1_enlarged"] = compare(pick, "field", extra=True)
res["H2_enlarged_hull"], _ = compare(pick, "clean", extra=True)
names, f1s, cs = static_table("field", extra=True)
st = pd.DataFrame(dict(system=names, f1=f1s, cost_T=np.array(cs) / 1e12)).sort_values("cost_T")
st.to_csv(od / "static_field.csv", index=False)
names, f1s, cs = static_table("clean", extra=True)
pd.DataFrame(dict(system=names, f1=f1s, cost_T=np.array(cs) / 1e12)).sort_values("cost_T").to_csv(
    od / "static_clean.csv", index=False)

# check 4 sensitivity: costs with every elementwise op and activation quantization at 32 x 32 on both tiers
res["H1_cost_all_elementwise"], fold_store["H1_all"] = compare(pick, "field", "all")


# break-even: extra cost added to every W8A8 cheap system until H1's mean difference reaches 0
def h1_with_delta(delta):
    keys = [(s, f) for s in seeds for f in folds]
    x = np.array([test_design[("field", pick, s, f)][0] for s, f in keys])
    rates = np.array([test_design[("field", pick, s, f)][1] for s, f in keys])
    cost = float(np.mean([dcost(pick, r_) for r_ in rates])) + delta
    names, f1s, cs = static_table("field")
    cs = [c_ + (delta if n.startswith("eff:w8a8") else 0.0) for n, c_ in zip(names, cs)]
    v, lo, hi, lam = hull_pair(names, f1s, cs, cost)
    h = np.array([hull_test("field", lo, hi, lam, s, f) for s, f in keys])
    fm = fold_means(x - h, [f for _, f in keys])
    return 100 * fm.mean(), fold_tests(fm, NB)["p_t"], f"{lo}/{hi}({lam:.2f})"


res["delta_cheap_cost"] = {f"{d_:.2f}T": h1_with_delta(d_ * 1e12) for d_ in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)}

# ── check 1: nested variants ───────────────────────────────────────────────
nested = {"strict": [], "design_only": []}
picks_nested = {}
for f in folds:
    keys_f = [(s, f) for s in seeds]
    sel_f = select("priced", keys=keys_f)
    d_f = tuple(sel_f.iloc[0][["cheap_bb", "cheap", "expensive", "rule", "budget"]])
    picks_nested[f] = " ".join(d_f)
    x = np.array([test_design[("field", d_f, s, f)][0] for s in seeds])
    rates = np.array([test_design[("field", d_f, s, f)][1] for s in seeds])
    cost = float(np.mean([dcost(d_f, r_) for r_ in rates]))
    # strict: hull pair from this fold's own field inner-validation rows (mean over seeds)
    sysn = sorted({k[1] for k in iv_static if k[0] == "field"})
    f1v = [np.mean([iv_static[("field", n, s, f)][0] for s in seeds]) for n in sysn]
    _, lo, hi, lam = hull_pair(sysn, f1v, [scost(n) for n in sysn], cost)
    h = np.array([hull_test("field", lo, hi, lam, s, f) for s in seeds])
    nested["strict"].append(float((x - h).mean()))
    # design per fold, hull pair as the claim (all folds' test means at this cost)
    names, f1s, cs = static_table("field")
    _, lo2, hi2, lam2 = hull_pair(names, f1s, cs, cost)
    h2 = np.array([hull_test("field", lo2, hi2, lam2, s, f) for s in seeds])
    nested["design_only"].append(float((x - h2).mean()))
res["nested"] = {k: dict(diff_pts=100 * float(np.mean(v)), fold_diffs_pts=[round(100 * x_, 3) for x_ in v],
                         **{a: b for a, b in fold_tests(np.array(v), NB).items() if a != "mean"}) for k, v in nested.items()}
res["nested"]["picks"] = picks_nested
fold_store["nested_strict"] = np.array(nested["strict"])
fold_store["nested_design_only"] = np.array(nested["design_only"])

# ── check 7: selection-rule sensitivity ────────────────────────────────────
sens = {}
for rank in (1, 2):
    d_ = tuple(SEL.iloc[rank][["cheap_bb", "cheap", "expensive", "rule", "budget"]])
    sens[f"rank{rank + 1}"], fold_store[f"rank{rank + 1}"] = compare(d_, "field")
for b in BUDGETS:
    d_ = pick[:4] + (b,)
    sens[f"fixed_{b}"], fold_store[f"fixed_{b}"] = compare(d_, "field")
sens["same_tier_other_rule"], _ = compare(pick[:3] + ({"ast": "avg", "avg": "ast"}[pick[3]],) + pick[4:], "field")
res["sensitivity"] = sens
SEL.assign(task=task).to_csv(od / "selection.csv", index=False)

# ── check 10: extra budgets (exploratory) and the static short AST ─────────
ex = {}
for b in EXTRA_B:
    d_ = pick[:4] + (b,)
    if ("field", d_, seeds[0], folds[0]) in test_design:
        o, _ = compare(d_, "field")
        ex[b] = {k_: o[k_] for k_ in ("f1", "cost_T", "rate", "diff_pts", "p_t")}
res["extra_budgets_h1_tiers"] = ex
st_f = pd.read_csv(od / "static_field.csv")
res["static_short"] = st_f[st_f.system.str.contains(f"@{SL}")].to_dict("records")
# every design on the field mixture against the hull (for cross-task selection and rank checks)
e2 = []
for d_ in designs:
    o, fm_ = compare(d_, "field")
    e2.append(dict(cheap_bb=d_[0], cheap=d_[1], expensive=d_[2], rule=d_[3], budget=d_[4], f1=o["f1"],
                   cost_T=o["cost_T"], diff_pts=o["diff_pts"], p_t=o["p_t"], p_nb=o["p_nb"],
                   fold_diffs=json.dumps(o["fold_diffs_pts"])))
E2 = pd.DataFrame(e2).merge(SEL[["cheap_bb", "cheap", "expensive", "rule", "budget", "gain_iv", "cost_iv"]])
E2.to_csv(od / "e2_field.csv", index=False)
res["spearman_gain_iv_vs_test_diff"] = float(E2[["gain_iv", "diff_pts"]].corr("spearman").iloc[0, 1])
res["seconds"] = time.time() - t0
json.dump(res, open(od / "tests.json", "w"), indent=1, default=float)
np.savez(od / "fold_diffs.npz", **{k: np.asarray(v) for k, v in fold_store.items()})
print(json.dumps({k: res[k] for k in ("pick_mine", "pick_claim", "pick_rank_of_claim", "H1", "H2", "nested")},
                 indent=1, default=float))
print(task, "done", f"{time.time() - t0:.0f}s", flush=True)
