"""Idea 6 option B, S4: the pick's cheap backbone as a real-int8 model (ONNX Runtime), scored with the head's
scaler refit on int8 features (score_int8_optb.py) and, for the record, with the FP32-fitted scaler (way a).

(a) Non-inferiority within 1 point of the int8 static against the W8A8 static with the same head, on clean and on
    shift (family of two, Holm within; the Nadeau-Bengio p alongside).
(b) The pick with the int8 cheap tier (thresholds re-set on the int8 model's own clean inner validation at the
    budget's rate) against the hull of every static plus the int8 statics of the scored head kinds at the W8A8
    price, on clean and shift (raw p).
(c) Humility of the int8 tier: escalation rate per condition with the clean thresholds, gate AUROC for "cheap wrong".

Reads analysis/tau/{picks.json, costs.json, static.csv} and the score files; writes analysis/tau/int8_static.csv,
int8_cascade.csv, int8_escalation.csv, int8_summary.json.

    python scripts/redesign/idea6/optionB/analyze_int8_optb.py --r R --meta CSV --model mn10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT, HERE.parent / "stage2", HERE.parent / "stage3b", HERE):
    sys.path.insert(0, str(p))

import analyze as A  # noqa: E402
from analyze_3b import ni_test  # noqa: E402
from analyze_optb import CONDS, DEVICES, K, SEEDS, Src  # noqa: E402
from idea5b_common import holm, mix_expected_f1  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--r", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--model", required=True)
    a = ap.parse_args(argv)
    out = a.r / "analysis" / "tau"
    picks = json.loads((out / "picks.json").read_text())
    m, k, e, r, b = picks["pick"]["design"]
    if m != a.model:
        raise SystemExit(f"the pick's cheap backbone is {m}, not {a.model}")
    costs = json.loads((out / "costs.json").read_text())
    S = pd.read_csv(out / "static.csv")
    meta_an = json.loads((out / "meta.json").read_text())
    nb_ratio = meta_an["nb_ratio"]
    rate_b = A.BUDGETS[b]
    kinds = sorted({p.name for s in SEEDS for p in (a.r / "scores_int8" / f"{m}_tau_s{s}").iterdir() if p.is_dir()})
    if k not in kinds:
        raise SystemExit(f"int8 scores for the pick's head kind {k} are missing")
    i8 = {(s, kk): Src(a.r / "scores_int8" / f"{m}_tau_s{s}" / kk) for s in SEEDS for kk in kinds}
    w8 = {(s, kk): Src(a.r / "scores" / f"{m}_tau_s{s}" / kk) for s in SEEDS for kk in kinds}
    ast = {s: Src(a.r / "scores" / f"ast_tau_s{s}" / "esc") for s in SEEDS}
    ref = w8[(SEEDS[0], k)].get(f"{m}_w8a8")
    folds = sorted(ref["iv"])
    y_u, fold_u, dev_u = ref["y"], ref["fold"], ref["device"]
    masks = {}
    for f in folds:
        rf = np.flatnonzero(fold_u == f)
        d = dev_u[rf]
        masks[f] = {"clean": rf[d == "a"], "shift": rf[d != "a"], **{f"dev:{x}": rf[d == x] for x in DEVICES}}
    cost_i8 = {kk: costs["cheap"][f"cnn:{m}:w8a8:{kk}"] for kk in kinds}
    cost_e = costs["ast"][f"ast:{e}+esc"]

    static, casc, esc_rows, au, preds, ys = [], [], [], {}, {}, {}
    for s in SEEDS:
        for f in folds:
            # thresholds on the int8 model's own clean inner validation (and the W8A8 ones for the record)
            thr = {}
            for way in ("int8", "int8a"):
                p = A.softmax(i8[(s, k)].get(f"{m}_{way}")["iv"][f][1])
                if not np.array_equal(i8[(s, k)].get(f"{m}_{way}")["iv"][f][0], w8[(s, k)].get(f"{m}_w8a8")["iv"][f][0]):
                    raise SystemExit("inner-validation rows differ")
                thr[way] = A.thr_rate(A.margin(p), rate_b)
            for c in CONDS:
                rows = masks[f][c]
                y = y_u[rows]
                ys[(s, f, c)] = y
                for kk in kinds:
                    for way, src, cfg in (("w8a8", w8[(s, kk)], f"{m}_w8a8"), ("int8", i8[(s, kk)], f"{m}_int8"), ("int8a", i8[(s, kk)], f"{m}_int8a")):
                        rr = src.get(cfg)
                        if not (np.array_equal(rr["y"], y_u) and np.array_equal(rr["fold"], fold_u)):
                            raise SystemExit(f"{src.d}/{cfg}: rows differ")
                        p = A.softmax(rr["z"][rows])
                        pr = p.argmax(1)
                        preds[(way, kk, c, s, f)] = pr
                        static.append({"way": way, "cheap": kk, "condition": c, "seed": s, "fold": f, "n": len(rows),
                                       "f1": A.f1_fast(y, pr, K), "cost": cost_i8[kk]})
                        if kk == k and way != "w8a8":
                            mg = A.margin(p)
                            escm = mg < thr[way]
                            esc_rows.append({"way": way, "condition": c, "seed": s, "fold": f, "rate": float(escm.mean()),
                                             "error": float((pr != y).mean())})
                            a_ = au.setdefault((way, c, s), {"score": [], "wrong": []})
                            a_["score"].append(1 - mg)
                            a_["wrong"].append(pr != y)
                            pe = A.softmax(ast[s].get(e)["z"][rows])
                            q = pe if r == "ast" else (p + pe) / 2
                            pred = np.where(escm, q.argmax(1), pr)
                            casc.append({"way": way, "condition": c, "seed": s, "fold": f, "f1": A.f1_fast(y, pred, K),
                                         "rate": float(escm.mean()), "cost": cost_i8[k] + float(escm.mean()) * cost_e})
    ST = pd.DataFrame(static)
    ST.to_csv(out / "int8_static.csv", index=False)
    CA = pd.DataFrame(casc)
    CA.to_csv(out / "int8_cascade.csv", index=False)
    pd.DataFrame(esc_rows).to_csv(out / "int8_escalation.csv", index=False)
    keys = [(s, f) for s in SEEDS for f in folds]

    def series(df, **sel):
        g = df
        for kk, v in sel.items():
            g = g[g[kk] == v]
        g = g.set_index(["seed", "fold"])
        return np.array([g.loc[kk, "f1"] for kk in keys])

    summary = {"pick": picks["pick"]["design"], "model": m, "kinds_scored": kinds, "ni": {}, "cascade": {}, "escalation": {}, "auroc": {}}
    ni_rows = []
    for way in ("int8", "int8a"):
        for c in ("clean", "shift"):
            x, z = series(ST, way=way, cheap=k, condition=c), series(ST, way="w8a8", cheap=k, condition=c)
            t_ = ni_test(x, z, keys, nb_ratio[c])
            ni_rows.append({"way": way, "condition": c, "f1_int8": float(x.mean()), "f1_w8a8": float(z.mean()), **t_})
    NI = pd.DataFrame(ni_rows)
    for way in ("int8", "int8a"):
        sel = NI.way == way
        NI.loc[sel, "ni_p_holm"] = holm(NI.loc[sel, "ni_p"].to_numpy())
    NI["verdict"] = np.where((NI.way == "int8") & (NI.ni_p_holm < 0.05), "HOLDS", np.where(NI.way == "int8", "FAILS", "record"))
    NI.to_csv(out / "int8_ni.csv", index=False)
    summary["ni"] = NI.to_dict("records")
    # the pick with the int8 tier against the hull plus the int8 statics (at the W8A8 price)
    S_all = S.copy()
    add = ST[ST.way == "int8"].assign(system=lambda d: "cnn:" + m + ":int8:" + d.cheap)[["system", "condition", "seed", "fold", "f1", "cost"]]
    S_all = pd.concat([S_all[["system", "condition", "seed", "fold", "f1", "cost"]], add], ignore_index=True)
    srec = {(n, c): dict(zip(zip(g.seed, g.fold), g.f1)) for (n, c), g in S_all.groupby(["system", "condition"])}
    an = pd.read_csv(out / "static.csv")
    cas_rows = []
    for way in ("int8", "int8a"):
        for c in ("clean", "shift"):
            g = CA[(CA.way == way) & (CA.condition == c)].set_index(["seed", "fold"])
            x = np.array([g.loc[kk, "f1"] for kk in keys])
            cost = float(g.cost.mean())
            sp = S_all[S_all.condition == c].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
            v_interp, lo, hi, lam = A.hull_pair(sp, cost)
            hull = []
            for kk in keys:
                if lam == 0.0:
                    hull.append(srec[(lo, c)][kk])
                    continue
                plo = preds[("int8", lo.split(":")[3], c, *kk)] if ":int8:" in lo else None
                phi = preds[("int8", hi.split(":")[3], c, *kk)] if ":int8:" in hi else None
                # statics of the main analysis are re-derived from their score files when needed
                if plo is None:
                    plo = static_pred(a.r, lo, c, kk, masks)
                if phi is None:
                    phi = static_pred(a.r, hi, c, kk, masks)
                hull.append(float(mix_expected_f1(ys[(*kk, c)], plo.astype(np.int64), phi.astype(np.int64), lam, K)[0]))
            t_ = A.tests(x, np.array(hull), keys, nb_ratio[c])
            cas_rows.append({"way": way, "condition": c, "f1": float(x.mean()), "cost": cost, "rate": float(g.rate.mean()),
                             "hull_lo": lo, "hull_hi": hi, "lam": lam, "hull_f1": float(np.mean(hull)), **t_})
    CR = pd.DataFrame(cas_rows)
    CR.to_csv(out / "int8_cascade_vs_hull.csv", index=False)
    summary["cascade"] = CR.to_dict("records")
    E = pd.DataFrame(esc_rows)
    summary["escalation"] = E.groupby(["way", "condition"]).rate.mean().unstack().to_dict()
    rows = []
    for (way, c, s), a_ in au.items():
        sc, w = np.concatenate(a_["score"]), np.concatenate(a_["wrong"])
        rows.append({"way": way, "condition": c, "seed": s, "auroc_wrong": roc_auc_score(w, sc) if 0 < w.sum() < len(w) else np.nan})
    AU = pd.DataFrame(rows)
    AU.to_csv(out / "int8_auroc.csv", index=False)
    summary["auroc"] = AU.groupby(["way", "condition"]).auroc_wrong.mean().unstack().to_dict()
    side = json.loads((a.r / "emb_int8" / "tau" / f"{m}_int8__clean.json").read_text())
    summary["embedding_error"] = {kk: v for kk, v in side.items() if kk.startswith("rel_err")}
    build = json.loads((a.r / "models" / "tau" / f"{m}_build.json").read_text())
    summary["int8_graph_nodes"] = build["folds"][str(min(folds))].get("int8_optimized_nodes")
    summary["calibration_method"] = build["method"]
    (out / "int8_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(NI[["way", "condition", "f1_int8", "f1_w8a8", "ni_diff", "ni_p", "ni_p_holm", "verdict"]].to_string())
    print(CR[["way", "condition", "f1", "hull_f1", "diff", "p_t_fold"]].to_string())


_SP = {}


def static_pred(r, name, c, kk, masks):
    """Test predictions of a static of the main analysis (cnn:<m>:<bb>:<k> or ast:<cfg>+<h>) on condition c."""
    s, f = kk
    if name.startswith("ast:"):
        cfg, h = name[4:].split("+")
        key = (r / "scores" / f"ast_tau_s{s}" / h, cfg)
    else:
        _, m, bb, kk_ = name.split(":")
        key = (r / "scores" / f"{m}_tau_s{s}" / kk_, f"{m}_{bb}")
    if key not in _SP:
        _SP[key] = Src(key[0]).get(key[1])["z"]
    return A.softmax(_SP[key][masks[f][c]]).argmax(1)


if __name__ == "__main__":
    main()
