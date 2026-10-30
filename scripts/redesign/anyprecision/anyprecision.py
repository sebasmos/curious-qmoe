"""T9, D4 any-precision kill check, one task, as fixed in scripts/redesign/anyprecision/ANALYSIS_PLAN.md.

A three-way margin gate over stored 3, 4 and 8-bit AST systems (3-bit on every clip, 4-bit on the clips whose 3-bit margin is
below t1, 8-bit on those whose 4-bit margin is below t2), thresholds from clean inner-validation margins, priced with option C's
priced bit-ops view, against the time-sharing hull of option C's static set plus the setting's always-average points.

  --phase select   inner-validation rows only: 240 designs, the pick (pick.json), the inner-validation static check
  --phase test     reads pick.json, scores the pick on the clean and field test folds, all designs descriptively

Loading of the static systems follows option C's analyze_c.py (same score folders, same names, same field assembly).

    python anyprecision.py --task esc50 --phase select --r R --s3c S3C --unp N --s3b S3B --s2 S2 --i5 I5 --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IDEA6 = HERE.parent / "idea6"
ROOT = HERE.parents[2]
for p in (ROOT, IDEA6 / "stage2", IDEA6 / "stage3b", IDEA6):
    sys.path.insert(0, str(p))

import analyze as A  # noqa: E402  Stage 2's helpers
from analyze_3b import AST_HEADS, CASC_CONDS, CHEAP_CONDS, FULL_TIERS, KINDS, NEW_CFGS, S2_SHORT, TASKS  # noqa: E402
from idea5b_common import bitops as ast_bitops  # noqa: E402
from idea5b_common import mix_expected_f1  # noqa: E402

CNN_S3B = {"mn04": 384, "mn10": 960}
CNN_NEW = {"mn05": 480, "mn20": 1920, "dymn04": 384}
NEW_TR = ["n3g64", "w3g32"]
NEW_OWN = ["w3g64", "n3g64", "w3g32"]
BUDGETS = {"b05": 0.05, "b10": 0.10, "b20": 0.20, "b30": 0.30, "b50": 0.50}
B2 = [0.2, 0.3, 0.5, 0.7]
RULES = ["tier", "avg"]
PRICE = 1024.0
NESTED_UP = {"n3g64": "w4g64", "n4g128": "w5g128", "n4": "w5", "n3": "w4", "n3g128": "w4g128", "n2g128": "w3g128"}   # k + 1 planes
COMBOS = {"34": (0, 1), "38": (0, 2), "48": (1, 2), "348": (0, 1, 2)}


def settings_for(SL):
    return {
        "F-direct": ("w3g64", "w4g128", "w8g64"),
        "F-nested": ("n3g64", "n4g128", "w8g64"),
        "R-direct-tr": (f"w3g64@{SL}", f"w4g128@{SL}", f"w8g64@{SL}"),
        "R-direct-own": (f"w3g64o@{SL}", f"w4g128@{SL}", f"w8g64@{SL}"),
        "R-nested-tr": (f"n3g64@{SL}", f"n4@{SL}", f"w8g64@{SL}"),
        "R-nested-own": (f"n3g64o@{SL}", f"n4@{SL}", f"w8g64@{SL}"),
    }


def cheap_name(src, bb, k):
    return f"eff:{bb}:{k}" if src == "eff" else f"cnn:{src}:{bb}:{k}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=["esc50", "esc5", "urban"])
    ap.add_argument("--phase", required=True, choices=["select", "test"])
    for k in ("r", "s3c", "unp", "s3b", "s2", "i5", "out"):
        ap.add_argument("--" + k, required=True, type=Path)
    a = ap.parse_args(argv)
    t0 = time.time()
    task = a.task
    T = TASKS[task]
    K, seeds, SL, ds = T["K"], T["seeds"], T["short"], T["ds"]
    urban = task == "urban"
    esc = ds == "esc50"
    test = a.phase == "test"
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    SET = settings_for(SL)

    # ---- sources, as analyze_c.py ------------------------------------------------------------------------------
    sources = {}

    def add(src, root, tag, cfg, dim, hull_set="stage3c", kinds=KINDS):
        dirs = {(s, k): A.ScoreDir(root / tag.format(s) / k) for s in seeds for k in kinds}
        have = [k for k in kinds if all(dirs[(s, k)].files for s in seeds)]
        if not have:
            raise SystemExit(f"{task}: source {src} has no scores under {root}")
        sources[src] = {"dirs": dirs, "cfg": cfg, "kinds": have, "set": hull_set}

    add("eff", a.s2 / "scores" / task, f"effp_{task}_s{{}}", {"fp32": "fp32", "w8a8": "w8a8"}, 1536)
    for m in CNN_S3B:
        name = "mn10pad" if urban and m == "mn10" else m
        add(name, a.s3b / "scores_cnn" / task, f"{m}_{task}_s{{}}", {bb: f"{m}_{bb}" for bb in ("fp32", "w8a8")}, 0,
            hull_set="excluded" if name == "mn10pad" else "stage3c")
    if urban:
        add("mn10unpr", a.unp / "scores_unpr" / "urban", "mn10unp_urban_s{}", {bb: f"mn10_{bb}" for bb in ("fp32", "w8a8")}, 0)
    for m in CNN_NEW:
        add(m, a.s3c / "scores_cnn" / task, f"{m}_{task}_s{{}}", {bb: f"{m}_{bb}" for bb in ("fp32", "w8a8")}, 0)

    full_root = a.i5 / "scores" / task
    groups = {
        "full": ({(s, h): A.ScoreDir(full_root / f"ast{task}_s{s}" / h) for s in seeds for h in AST_HEADS}, None, ""),
        "s2short": ({(s, h): A.ScoreDir(a.s2 / "scores_ast_short" / task / f"astshort_{task}_s{s}" / h)
                     for s in seeds for h in AST_HEADS}, S2_SHORT, f"@{SL}"),
        "a256": ({(s, h): A.ScoreDir(a.s3b / "scores_ast256" / task / f"ast256_{task}_s{s}" / h)
                  for s in seeds for h in AST_HEADS}, NEW_CFGS, "@256"),
        "a384": ({(s, h): A.ScoreDir(a.s3b / "scores_ast384" / task / f"ast384_{task}_s{s}" / h)
                  for s in seeds for h in AST_HEADS}, NEW_CFGS, "@384"),
    }
    if esc:
        groups["m2"] = ({(s, h): A.ScoreDir(a.s3b / "scores_astm2" / task / f"astm2_{task}_s{s}" / h)
                         for s in seeds for h in AST_HEADS}, NEW_CFGS, "@256x2")
    groups["tr"] = ({(s, h): A.ScoreDir(a.r / "scores_tr" / task / f"astshort_{task}_s{s}" / h)
                     for s in seeds for h in AST_HEADS}, NEW_TR, f"@{SL}")
    src_ast, ast_set = {}, {}
    for g, (dirs, cfgs, sfx) in groups.items():
        if cfgs is None:
            cfgs = sorted({f.split("__")[0] for f in dirs[(seeds[0], "esc")].files})
        for c in cfgs:
            src_ast[c + sfx] = (dirs, c, g)
            ast_set[c + sfx] = "new" if g == "tr" else "stage3c"
    for c in NEW_OWN:
        dirs = {(s, h): A.ScoreDir(a.r / "scores_own" / task / f"own_{c}_{task}_s{s}" / h) for s in seeds for h in AST_HEADS}
        src_ast[f"{c}o@{SL}"] = (dirs, c, "own")
        ast_set[f"{c}o@{SL}"] = "new"

    def ast_has(name, cond):
        dirs, base, _ = src_ast[name]
        return all(d.has(base, cond) for d in dirs.values())

    def ast_get(name, s, h, cond):
        dirs, base, _ = src_ast[name]
        return dirs[(s, h)].get(base, cond)

    def cheap_get(src, bb, s, k, cond):
        return sources[src]["dirs"][(s, k)].get(sources[src]["cfg"][bb], cond)

    def cheap_has(src, bb, cond):
        S_ = sources[src]
        return all(S_["dirs"][(s, k)].has(S_["cfg"][bb], cond) for s in seeds for k in S_["kinds"])

    conds = CHEAP_CONDS + ["field"]
    cheap_all = [(x, bb) for x in sources for bb in sources[x]["cfg"]]
    cheap_conds = {cb: [c for c in conds if cheap_has(*cb, c)] for cb in cheap_all}
    ast_conds = {n: [c for c in conds if ast_has(n, c)] for n in src_ast}
    tiers_needed = sorted({t for v in SET.values() for t in v})
    for t in tiers_needed:
        if t not in src_ast or not all(c in ast_conds[t] for c in CASC_CONDS):
            raise SystemExit(f"{task}: tier {t} lacks clean or field scores")
    print(f"{task} {a.phase}: {len(cheap_all)} cheap, {len(src_ast)} AST systems", flush=True)

    # ---- costs: option C's stored priced costs, nested tiers of the settings re-priced at k + 1 planes -----------
    stored = json.loads((a.r / "analysis" / task / "costs.json").read_text())
    assert stored["price"] == PRICE
    cost_of = {}
    for k_, v in stored["cheap"].items():
        cost_of[k_] = v
    for k_, v in stored["ast"].items():
        cost_of["ast:" + k_] = v

    def tier_bitops(base, L, price):
        if L is None:
            return ast_bitops(base, price)
        P_, T_ = A.short_tokens(L)
        return A.ast_bitops_tok(base, price, T_, P_)

    def nested_delta(tier):
        """Extra priced bit-ops of the nested child needing k + 1 planes (0 for a direct tier)."""
        base, _, suf = tier.partition("@")
        if base.endswith("o"):
            base = base[:-1]
        if base not in NESTED_UP:
            return 0.0
        mult = 2 if suf.endswith("x2") else 1
        L = int(suf.replace("x2", "")) if suf else None
        return mult * (tier_bitops(NESTED_UP[base], L, PRICE) - tier_bitops(base, L, PRICE))

    def tcost(tier, plain=False):
        return cost_of[f"ast:{tier}+esc"] + (0.0 if plain else nested_delta(tier))

    def sys_cost(name):
        return cost_of[name]

    def sys_set(name):
        if name.startswith("ast:"):
            return ast_set[name[4:].split("+")[0]]
        if name.startswith("eff:"):
            return "stage3c"
        return sources[name.split(":")[1]]["set"]

    # ---- one pass over seeds and folds -------------------------------------------------------------------------
    ref0 = cheap_get("eff", "fp32", seeds[0], "esc", "clean")
    folds = sorted(ref0["iv"])
    n_te = np.array([np.sum(ref0["fold"] == f) for f in folds])
    A.NB_RATIO["task"] = float(np.mean(n_te / (len(ref0["y"]) - n_te)))
    keys = [(s, f) for s in seeds for f in folds]
    static, ivrows = [], []
    iv_static, preds, ys, yiv, tp_iv, tp_te = {}, {}, {}, {}, {}, {}   # tier probabilities: iv (clean), test (clean, field)
    for s in seeds:
        ref = cheap_get("eff", "fp32", s, "esc", "clean")
        y_all = ref["y"]
        for fold in folds:
            te = np.flatnonzero(ref["fold"] == fold)
            iv = ref["iv"][fold][0]
            key = (s, fold)
            yiv[key] = y_all[iv]
            if test:
                ys[key] = y_all[te]
            for x, bb in cheap_all:
                for k in sources[x]["kinds"]:
                    r_ = cheap_get(x, bb, s, k, "clean")
                    if not (np.array_equal(r_["iv"][fold][0], iv) and (r_["y"] == y_all).all()):
                        raise SystemExit(f"{x}:{bb}:{k}: rows differ")
                    nm = cheap_name(x, bb, k)
                    pr = A.softmax(r_["iv"][fold][1]).argmax(1)
                    iv_static[(nm, s, fold)] = (A.f1_fast(yiv[key], pr, K), pr)
                    ivrows.append((nm, s, fold, iv_static[(nm, s, fold)][0]))
            for n in src_ast:
                for h in AST_HEADS:
                    r_ = ast_get(n, s, h, "clean")
                    if not (np.array_equal(r_["iv"][fold][0], iv) and (r_["y"] == y_all).all()):
                        raise SystemExit(f"{n}+{h}: rows differ")
                    p = A.softmax(r_["iv"][fold][1])
                    nm = f"ast:{n}+{h}"
                    pr = p.argmax(1)
                    iv_static[(nm, s, fold)] = (A.f1_fast(yiv[key], pr, K), pr)
                    ivrows.append((nm, s, fold, iv_static[(nm, s, fold)][0]))
                    if h == "esc" and n in tiers_needed:
                        tp_iv[(n, s, fold)] = p
            if test:
                for c in CASC_CONDS:
                    for x, bb in cheap_all:
                        if c not in cheap_conds[(x, bb)]:
                            continue
                        for k in sources[x]["kinds"]:
                            pr = cheap_get(x, bb, s, k, c)["z"][te].argmax(1)
                            nm = cheap_name(x, bb, k)
                            preds[(nm, c, s, fold)] = pr.astype(np.int16)
                            static.append((nm, c, s, fold, A.f1_fast(ys[key], pr, K)))
                    for n in src_ast:
                        if c not in ast_conds[n]:
                            continue
                        for h in AST_HEADS:
                            p = A.softmax(ast_get(n, s, h, c)["z"][te])
                            pr = p.argmax(1)
                            nm = f"ast:{n}+{h}"
                            preds[(nm, c, s, fold)] = pr.astype(np.int16)
                            static.append((nm, c, s, fold, A.f1_fast(ys[key], pr, K)))
                            if h == "esc" and n in tiers_needed:
                                tp_te[(n, c, s, fold)] = p.astype(np.float32)
        for S_ in sources.values():
            for d in S_["dirs"].values():
                d.cache.clear()
        for n, (dirs, _, _) in src_ast.items():
            for d in dirs.values():
                d.cache.clear()
        print(f"{task} seed {s} loaded {time.time() - t0:.0f}s", flush=True)

    # ---- the inner-validation static check against option C's static_iv.csv ----------------------------------
    IV = pd.DataFrame(ivrows, columns=["system", "seed", "fold", "f1"])
    ref_iv = pd.read_csv(a.r / "analysis" / task / "static_iv.csv").set_index("system")
    mine = IV.groupby("system").f1.mean()
    diff = (mine - ref_iv.loc[mine.index, "f1"]).abs()
    if diff.max() > 1e-9 or set(mine.index) != set(ref_iv.index):
        raise SystemExit(f"{task}: inner-validation statics differ from option C (max {diff.max():.2e}); "
                         f"missing {set(ref_iv.index) ^ set(mine.index)}")
    print(f"{task}: inner-validation statics equal option C's static_iv.csv (max diff {diff.max():.1e}, {len(mine)} systems)", flush=True)
    base_sys = [n for n in mine.index if sys_set(n) in ("stage3c", "new")]

    # ---- gate ----------------------------------------------------------------------------------------------------
    def thresholds(setting, b1, b2, key):
        t3, t4, _ = SET[setting]
        s, f = key
        return (A.thr_rate(A.margin(tp_iv[(t3, s, f)]), b1), A.thr_rate(A.margin(tp_iv[(t4, s, f)]), b2))

    def gate(p3, p4, p8, t1, t2, rule):
        e1 = A.margin(p3) < t1
        e2 = e1 & (A.margin(p4) < t2)
        if rule == "tier":
            pred = np.where(e2, p8.argmax(1), np.where(e1, p4.argmax(1), p3.argmax(1)))
        else:
            pred = np.where(e2, ((p3 + p4 + p8) / 3).argmax(1), np.where(e1, ((p3 + p4) / 2).argmax(1), p3.argmax(1)))
        return pred, e1, e2

    def ens_pred(setting, combo, probs):
        ts = SET[setting]
        return np.mean([probs[ts[i]] for i in COMBOS[combo]], axis=0).argmax(1)

    def ens_cost(setting, combo, plain=False):
        return float(sum(tcost(SET[setting][i], plain) for i in COMBOS[combo]))

    DES = [(st, r, b1, b2) for st in SET for r in RULES for b1 in BUDGETS for b2 in B2]

    # ---- selection on clean inner validation ---------------------------------------------------------------------
    if a.phase == "select":
        st_iv = IV[IV.system.isin(base_sys)].groupby("system").f1.mean()
        sp_base = pd.DataFrame({"system": st_iv.index, "f1": st_iv.values, "cost": [sys_cost(n) for n in st_iv.index]})
        rng = np.random.default_rng(A.RNG_IV)
        rows = []
        # inner-validation predictions of the setting's always-average points
        for st in SET:
            for cb in COMBOS:
                nm = f"ens:{st}:{cb}"
                for key in keys:
                    probs = {t: tp_iv[(t, *key)] for t in SET[st]}
                    pr = ens_pred(st, cb, probs)
                    iv_static[(nm, *key)] = (A.f1_fast(yiv[key], pr, K), pr)
        for st, r, bn, b2 in DES:
            t3, t4, t8 = SET[st]
            f1s, r1s, r2s = [], [], []
            for key in keys:
                t1, t2 = thresholds(st, BUDGETS[bn], b2, key)
                pred, e1, e2 = gate(tp_iv[(t3, *key)], tp_iv[(t4, *key)], tp_iv[(t8, *key)], t1, t2, r)
                f1s.append(A.f1_fast(yiv[key], pred, K))
                r1s.append(e1.mean())
                r2s.append(e2.mean())
            c_iv = float(np.mean([tcost(t3) + a1 * tcost(t4) + a2 * tcost(t8) for a1, a2 in zip(r1s, r2s)]))
            ens_rows = [{"system": f"ens:{st}:{cb}", "f1": float(np.mean([iv_static[(f"ens:{st}:{cb}", *k)][0] for k in keys])),
                         "cost": ens_cost(st, cb)} for cb in COMBOS]
            sp = pd.concat([sp_base, pd.DataFrame(ens_rows)], ignore_index=True)
            v, lo, hi, lam = A.hull_pair(sp, c_iv)
            hull = []
            for key in keys:
                plo = iv_static[(lo, *key)][1]
                if lam == 0.0:
                    hull.append(iv_static[(lo, *key)][0])
                    continue
                phi = iv_static[(hi, *key)][1]
                draws = rng.random((A.DRAWS, len(plo))) < lam
                hull.append(float(A.f1_batch(yiv[key], np.where(draws, phi[None, :], plo[None, :]), K).mean()))
            hull = np.array(hull)
            rows.append({"setting": st, "rule": r, "b1": bn, "b2": b2, "f1_iv": float(np.mean(f1s)), "rate1_iv": float(np.mean(r1s)),
                         "rate2_iv": float(np.mean(r2s)), "cost_iv": c_iv, "hull_lo": lo, "hull_hi": hi, "lam": lam,
                         "hull_f1_iv": float(hull.mean()), "gain_iv": float((np.array(f1s) - hull).mean())})
        SO = pd.DataFrame(rows).sort_values(["gain_iv", "cost_iv"], ascending=[False, True])
        SO.to_csv(out / "selection.csv", index=False)
        top = SO.iloc[0]
        pick = {"task": task, "design": [top.setting, top.rule, top.b1, float(top.b2)], "gain_iv": float(top.gain_iv),
                "cost_iv": float(top.cost_iv), "rate1_iv": float(top.rate1_iv), "rate2_iv": float(top.rate2_iv),
                "runner_up": [str(v) for v in SO.iloc[1][["setting", "rule", "b1", "b2"]]],
                "runner_up_gain_iv": float(SO.iloc[1].gain_iv), "n_designs": int(len(SO)),
                "best_per_setting": {st: float(SO[SO.setting == st].gain_iv.max()) for st in SET}}
        (out / "pick.json").write_text(json.dumps(pick, indent=1))
        print(f"{task}: pick {pick['design']} gain_iv {pick['gain_iv']:+.4f} cost {pick['cost_iv'] / 1e12:.3f} T "
              f"({time.time() - t0:.0f}s)", flush=True)
        return

    # ---- test phase ----------------------------------------------------------------------------------------------
    S = pd.DataFrame(static, columns=["system", "condition", "seed", "fold", "f1"])
    ref_st = pd.read_csv(a.r / "analysis" / task / "static.csv")
    ref_st = ref_st[ref_st.condition.isin(CASC_CONDS)]
    m_ = S.merge(ref_st[["system", "condition", "seed", "fold", "f1"]], on=["system", "condition", "seed", "fold"], how="outer",
                 suffixes=("", "_ref"), indicator=True)
    if (m_["_merge"] != "both").any() or (m_.f1 - m_.f1_ref).abs().max() > 1e-9:
        bad = m_[(m_["_merge"] != "both") | ((m_.f1 - m_.f1_ref).abs() > 1e-9)]
        raise SystemExit(f"{task}: test statics differ from option C's static.csv:\n{bad.head()}")
    print(f"{task}: test statics equal option C's static.csv ({len(S)} rows)", flush=True)
    S["set"] = S.system.map(sys_set)
    S = S[S["set"].isin(["stage3c", "new"])]
    s_rec = {(n, c): dict(zip(zip(g.seed, g.fold), g.f1)) for (n, c), g in S.groupby(["system", "condition"])}
    sp_stat = {c: g.groupby("system").f1.mean() for c, g in S.groupby("condition")}

    pick = json.loads((out / "pick.json").read_text())
    st_pick, r_pick, bn_pick, b2_pick = pick["design"]
    b2_pick = float(b2_pick)

    def pred_of(name, c, key, setting):
        if name.startswith("ens:"):
            _, st, cb = name.split(":")
            return ens_pred(st, cb, {t: tp_te[(t, c, *key)].astype(np.float64) for t in SET[st]}).astype(np.int64)
        return preds[(name, c, *key)].astype(np.int64)

    def evaluate(design, c, plain=False, nested_static_up=False, detail=False):
        st, r, bn, b2 = design
        t3, t4, t8 = SET[st]
        f1s, r1s, r2s, gates = [], [], [], {}
        for key in keys:
            t1, t2 = thresholds(st, BUDGETS[bn], b2, key)
            pred, e1, e2 = gate(*(tp_te[(t, c, *key)].astype(np.float64) for t in (t3, t4, t8)), t1, t2, r)
            f1s.append(A.f1_fast(ys[key], pred, K))
            r1s.append(e1.mean())
            r2s.append(e2.mean())
            if detail:
                gates[key] = (e1, e2)
        xv = np.array(f1s)
        cost = float(np.mean([tcost(t3, plain) + a1 * tcost(t4, plain) + a2 * tcost(t8, plain) for a1, a2 in zip(r1s, r2s)]))
        # hull set: option C's statics for this condition plus the setting's always-average points
        sp = sp_stat[c]
        names = list(sp.index)
        costs_ = {n: sys_cost(n) for n in names}
        if nested_static_up:
            for n in names:
                if n.startswith("ast:"):
                    costs_[n] += nested_delta(n[4:].split("+")[0])
        f1m = {n: float(sp[n]) for n in names}
        recs = dict((n, s_rec[(n, c)]) for n in names)
        for cb in COMBOS:
            nm = f"ens:{st}:{cb}"
            vals = {k_: A.f1_fast(ys[k_], pred_of(nm, c, k_, st), K) for k_ in keys}
            recs[nm] = vals
            f1m[nm] = float(np.mean(list(vals.values())))
            costs_[nm] = ens_cost(st, cb, plain)
        spd = pd.DataFrame({"system": list(f1m), "f1": [f1m[n] for n in f1m], "cost": [costs_[n] for n in f1m]})
        v, lo, hi, lam = A.hull_pair(spd, cost)
        hull = np.array([recs[lo][k_] if lam == 0.0 else float(mix_expected_f1(ys[k_], pred_of(lo, c, k_, st), pred_of(hi, c, k_, st), lam, K)[0])
                         for k_ in keys])
        row = {"setting": st, "rule": r, "b1": bn, "b2": b2, "condition": c, "f1": float(xv.mean()), "cost": cost,
               "rate1": float(np.mean(r1s)), "rate2": float(np.mean(r2s)), "hull_lo": lo, "hull_hi": hi, "lam": lam,
               "hull_f1_interp": v, "hull_f1": float(hull.mean())}
        row.update({f"hull_{k_}": v_ for k_, v_ in A.tests(xv, hull, keys).items()})
        if detail:
            return row, gates
        return row

    # the pre-registered pick
    res = []
    gates_pick = {}
    for c in CASC_CONDS:
        row, g_ = evaluate((st_pick, r_pick, bn_pick, b2_pick), c, detail=True)
        row["family"] = "H1" if c == "field" else "H2"
        res.append(row)
        gates_pick[c] = g_
    # sensitivities of the pick (descriptive)
    for c in CASC_CONDS:
        row = evaluate((st_pick, r_pick, bn_pick, b2_pick), c, plain=True)
        row["family"] = "sens_plain_nested_price"
        res.append(row)
        row = evaluate((st_pick, r_pick, bn_pick, b2_pick), c, nested_static_up=True)
        row["family"] = "sens_nested_statics_k_plus_1"
        res.append(row)
    pd.DataFrame(res).to_csv(out / "selected.csv", index=False)
    print(f"{task}: pick scored {time.time() - t0:.0f}s", flush=True)

    # diagnostics of the pick: where clips end, accuracy of each precision on each assigned tier
    t3, t4, t8 = SET[st_pick]
    dg = []
    for c in CASC_CONDS:
        tot = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0], 2: [0, 0, 0, 0]}   # n, correct3, correct4, correct8
        for key in keys:
            e1, e2 = gates_pick[c][key]
            grp = np.where(e2, 2, np.where(e1, 1, 0))
            ps = [tp_te[(t, c, *key)].argmax(1) for t in (t3, t4, t8)]
            for g_ in range(3):
                m = grp == g_
                tot[g_][0] += int(m.sum())
                for j in range(3):
                    tot[g_][1 + j] += int((ps[j][m] == ys[key][m]).sum())
        n_all = sum(v[0] for v in tot.values())
        for g_, v in tot.items():
            dg.append({"condition": c, "ends_at": ["3-bit", "4-bit", "8-bit"][g_], "share": v[0] / n_all, "n": v[0],
                       "acc_3bit": v[1] / max(v[0], 1), "acc_4bit": v[2] / max(v[0], 1), "acc_8bit": v[3] / max(v[0], 1)})
    pd.DataFrame(dg).to_csv(out / "tier_diagnostics.csv", index=False)

    # every design, descriptive
    allrows = []
    for d in DES:
        for c in CASC_CONDS:
            allrows.append(evaluate(d, c))
    AL = pd.DataFrame(allrows)
    AL.to_csv(out / "all_designs.csv", index=False)
    meta = {"task": task, "K": K, "seeds": seeds, "folds": folds, "nb_ratio": A.NB_RATIO["task"], "pick": pick,
            "settings": SET, "n_hull_systems": {c: int(len(v)) for c, v in sp_stat.items()}, "seconds": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{task}: test phase done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
