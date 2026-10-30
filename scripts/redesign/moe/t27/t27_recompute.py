"""T27 own recompute (ANALYSIS_PLAN.md, Recompute): (i) S2 minus the always-on MN10+MN20+AST average and (ii) S2 minus
the widened hull extended with that average, from the raw score files only. Uses T26's independent recompute
(t26_recompute.py: own softmax, margin, threshold, sklearn macro F1, linear envelope, Monte Carlo time-sharing, own
NB test, own loading of the score files) and none of the analysis helpers apart from the per-clip cost arrays. S1 and
S2 budgets are fixed at T26's (read from its verdict files); no reselection.

    python t27_recompute.py --r R27
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t26"))
import t26_recompute as RC  # noqa: E402

R26 = Path.home() / "Documents/qmoe-results/2026-10-29_0912_moe-t26_code-6ecea17"
TRI = "mn10+mn20+ast"
_sp, _sc = RC.static_preds, RC.static_cost
BUDGET = {}


def static_preds(P):
    pr = _sp(P)
    pr[TRI] = (P["mn10"] + P["mn20"] + P["ast"]).argmax(1)
    return pr


def static_cost(c):
    out = _sc(c)
    out[TRI] = c["mn10"] + c["mn20"] + c["ast"]
    return out


def fixed_choose(IV, keys):
    """T26's budgets, thresholds by RC.cut_at / RC.s2_cuts on clean inner validation (the recompute's own rule)."""
    r1, (r2, r3) = BUDGET["s1"], BUDGET["s2"]
    cuts = {"s1": {k: RC.cut_at(RC.marg(IV[k][1]["mn10"]), r1) for k in keys},
            "s2": {k: RC.s2_cuts(IV[k][1], r2, r3) for k in keys}}
    return {"s1": (r1,), "s2": (r2, r3)}, cuts, None


def evaluate(TE, keys, cuts, K, ratio, conds, rng_seed=7):
    """Per bed: S2, the three-model average, (i) and (ii); also S2 minus the 17-point hull for reference."""
    rng = np.random.default_rng(rng_seed)
    idx = pd.MultiIndex.from_tuples(keys, names=["seed", "fold"])
    out = {}
    for cond in conds:
        rec, pts = {}, {}
        for k in keys:
            y, P, c = TE[(cond, k)]
            pr, cs = static_preds(P), static_cost(c)
            rec[k] = (y, pr, RC.systems(P, c, {"s2": cuts["s2"][k]})["s2"])
            for nm in pr:
                a = pts.setdefault(nm, [[], []])
                a[0].append(cs[nm].mean())
                a[1].append(RC.f1(y, pr[nm]))
        pts = {nm: (np.mean(v[0]), np.mean(v[1])) for nm, v in pts.items()}
        xs = np.array([RC.f1(rec[k][0], rec[k][2][0]) for k in keys])
        Cx = float(np.mean([rec[k][2][1].mean() for k in keys]))
        tri = np.array([RC.f1(rec[k][0], rec[k][1][TRI]) for k in keys])
        fold_mean = lambda v: pd.Series(v, index=idx).groupby(level="fold").mean().to_numpy()  # noqa: E731
        r = {"s2_f1": float(xs.mean()), "s2_cost_T": Cx / 1e12, "tri_f1": pts[TRI][1], "tri_cost_T": pts[TRI][0] / 1e12,
             "i": RC.nb(fold_mean(xs - tri), ratio)}
        for tag, P_ in (("ii", pts), ("hull17", {nm: v for nm, v in pts.items() if nm != TRI})):
            _, lo, hi, lam = RC.envelope(P_, Cx)
            hs = []
            for k in keys:
                y, pr, _ = rec[k]
                if lam == 0:
                    hs.append(RC.f1(y, pr[lo]))
                else:
                    dr = rng.random((RC.MC, len(y))) < lam
                    hs.append(float(RC.f1_many(y, np.where(dr, pr[hi][None, :], pr[lo][None, :]), K).mean()))
            r[tag] = {"hull": f"{lo}|{hi}|{lam:.4f}", "hull_f1": float(np.mean(hs)), **RC.nb(fold_mean(xs - np.array(hs)), ratio)}
        out[f"s2|{cond}"] = r
    return out


RC.static_preds, RC.static_cost, RC.choose, RC.evaluate = static_preds, static_cost, fixed_choose, evaluate


def compare(rec, R):
    """Recompute minus analysis on (i) and (ii) differences and on the F1s; the plan's bar is 0.01 F1 points."""
    rows = []
    for d, name in (("d1", "fsdkaggle2018"), ("d3", "fsdnoisy18k")):
        p = R / "tables" / d / "results.json"
        if not p.exists():
            return None
        an = {r["cond"]: r for r in json.loads(p.read_text())["rows"]}
        for cond in ("field", "clean"):
            x, a = rec[name][f"s2|{cond}"], an[cond]
            for lab, xv, av in (("s2_f1", x["s2_f1"], a["s2_f1"]), ("tri_f1", x["tri_f1"], a["tri_f1"]),
                                ("i_d", x["i"]["d"], a["i_d"]), ("ii_d", x["ii"]["d"], a["ii_d"]),
                                ("ii_hull_f1", x["ii"]["hull_f1"], a["ii_hull_f1"]), ("hull17_d", x["hull17"]["d"], a["hull17_d"]),
                                ("i_p", x["i"]["p"], a["i_p_nb"]), ("ii_p", x["ii"]["p"], a["ii_p_nb"])):
                rows.append({"dataset": name, "cond": cond, "quantity": lab, "recompute": xv, "analysis": av, "abs_diff": abs(xv - av)})
        for cond in ("field", "clean"):
            x, a = rec[name][f"s2|{cond}"], an[cond]
            for lab, xv, av in (("ii_hull", x["ii"]["hull"], a["ii_hull"]),):
                rows.append({"dataset": name, "cond": cond, "quantity": lab, "recompute": xv, "analysis": av, "abs_diff": float(xv.split("|")[:2] != av.split("|")[:2])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    a = ap.parse_args()
    rec = {}
    for name, part, vpath in (("fsdkaggle2018", "a", "tables/a/verdict.json"), ("fsdnoisy18k", "c", "tables/c/d3/verdict.json")):
        V26 = json.loads((R26 / vpath).read_text())
        BUDGET["s1"] = float(V26["budgets"]["s1"])
        BUDGET["s2"] = tuple(float(x) for x in V26["budgets"]["s2"].split("|"))
        res, _ = RC.part_a(R26) if part == "a" else RC.part_c(R26, "d3", [0, 1, 2], False)
        rec[name] = res
        print(json.dumps({name: res}, indent=1, default=str), flush=True)
    (a.r / "tables" / "recompute.json").write_text(json.dumps(rec, indent=1, default=str))
    C = compare(rec, a.r)
    if C is not None:
        C.to_csv(a.r / "tables" / "recompute_vs_analysis.csv", index=False)
        num = C[~C.quantity.isin(["ii_hull", "i_p", "ii_p"])]
        worst = float(num.abs_diff.max())
        agree = {"worst_abs_diff_f1_points": worst, "bar": 0.01, "agree": bool(worst <= 0.01),
                 "hull_endpoints_same": bool((C[C.quantity == "ii_hull"].abs_diff == 0).all()),
                 "worst_p_diff": float(C[C.quantity.isin(["i_p", "ii_p"])].abs_diff.max())}
        (a.r / "tables" / "recompute_agreement.json").write_text(json.dumps(agree, indent=1))
        print(C.round(5).to_string())
        print(json.dumps(agree, indent=1))


if __name__ == "__main__":
    main()
