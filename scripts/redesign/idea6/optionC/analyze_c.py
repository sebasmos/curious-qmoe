"""Idea 6 option C analysis of one task, as fixed in scripts/redesign/idea6/optionC/ANALYSIS_PLAN.md (dfb54d5):
Stage 3c's analysis (analyze_3c.py, helpers imported from Stage 2 and 3b) with the 3-bit AST tiers at the clips'
real length as new expensive tiers (transferred and own heads), the cheap family fixed to mn10 and mn20 W8A8
(the exact-length mn10 with retrained heads on UrbanSound8K), a --price switch for the unpriced cost view (S4),
the three-tier cascade (E1) and the reproduction of Stage 3c's pick (E3).

Cheap sources (Stage 2's ScoreDir layout):
  eff            --s2/scores/<task>/effp_<task>_s<seed>/<kind>/<fp32|w8a8>__<cond>.npz
  mn04, mn10     --s3b/scores_cnn/<task>/...        (urban: mn10 is loaded as mn10pad, outside every hull)
  mn05 mn20 dymn04   --s3c/scores_cnn/<task>/...
  mn10unpr (urban)   --unp/scores_unpr/urban/mn10unp_urban_s<seed>/<kind>/mn10_<bb>__<cond>.npz
AST tiers: Stage 3c's (1024-frame from idea 5 Stage 2, 512/400 from Stage 2, 256/384 and the two-window mean from
Stage 3b) plus, from --r: scores_tr/<task>/astshort_<task>_s<seed>/<h>/<n3g64|w3g32>__<cond>.npz (transferred heads,
names n3g64@L, w3g32@L) and scores_own/<task>/own_<cfg>_<task>_s<seed>/<h>/<cfg>__<cond>.npz (own heads, names
<cfg>o@L, cfg in w3g64, n3g64, w3g32).

Hull sets: "stage3c" (every static of Stage 3c's "new" set, mn10unpr for mn10 on urban; E3), "new" (plus the five
new AST systems; H1, H2, S1, S3, E1, E2).

    python scripts/redesign/idea6/optionC/analyze_c.py --task esc50 --r R --s3c S3C --unp N --s3b S3B --s2 S2 --i5 I5 --i7 I7 [--price 0 --out DIR]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
S3BDIR = HERE.parent / "stage3b"
S2DIR = HERE.parent / "stage2"
ROOT = HERE.parents[3]
for p in (ROOT, S2DIR, S3BDIR, HERE.parent):
    sys.path.insert(0, str(p))

import analyze as A  # noqa: E402  Stage 2's helpers
from analyze_3b import AST_HEADS, BUDGETS, CASC_CONDS, CHEAP_CONDS, FULL_TIERS, KINDS, NEW_CFGS, RULES  # noqa: E402
from analyze_3b import S2_SHORT, TASKS  # noqa: E402
from idea5b_common import mix_expected_f1  # noqa: E402

CNN_S3B = {"mn04": 384, "mn10": 960}
CNN_NEW = {"mn05": 480, "mn20": 1920, "dymn04": 384}
P3_CHEAP = [("eff", "w8a8")] + [(m, "w8a8") for m in [*CNN_S3B, *CNN_NEW]]   # Stage 3c's family (E3)
C_CHEAP = [("mn10", "w8a8"), ("mn20", "w8a8")]                               # this run's family (H1)
NEW_TR = ["n3g64", "w3g32"]
NEW_OWN = ["w3g64", "n3g64", "w3g32"]
B2 = [0.2, 0.3, 0.5, 0.7]
SET_ORDER = {"stage3c": ["stage3c"], "new": ["stage3c", "new"]}


def cheap_name(src, bb, k):
    return f"eff:{bb}:{k}" if src == "eff" else f"cnn:{src}:{bb}:{k}"


def ci95(fm):
    k = len(fm)
    if k < 2 or fm.std(ddof=1) == 0:
        return float(fm.mean()), float(fm.mean())
    h = stats.t.ppf(0.975, k - 1) * fm.std(ddof=1) / np.sqrt(k)
    return float(fm.mean() - h), float(fm.mean() + h)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=["esc50", "esc5", "urban"])
    ap.add_argument("--r", required=True, type=Path)
    ap.add_argument("--s3c", required=True, type=Path)
    ap.add_argument("--unp", required=True, type=Path, help="the exact-length rerun folder (urban)")
    ap.add_argument("--s3b", required=True, type=Path)
    ap.add_argument("--s2", required=True, type=Path)
    ap.add_argument("--i5", required=True, type=Path)
    ap.add_argument("--i7", required=True, type=Path)
    ap.add_argument("--price", type=float, default=1024.0, help="bit-ops per rescale (1024 priced, 0 unpriced)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-mean2", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="code check before the new systems are complete: drop the new AST systems lacking files, "
                         "skip E1 when no own tier is left, write to --out outside the results folder")
    a = ap.parse_args(argv)
    if a.check_only and a.out is None:
        raise SystemExit("--check-only needs --out outside the results folder")
    sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6"))
    t0 = time.time()
    task = a.task
    T = TASKS[task]
    K, seeds, SL = T["K"], T["seeds"], T["short"]
    ds = T["ds"]
    priced = a.price > 0
    out = a.out or a.r / "analysis" / task
    out.mkdir(parents=True, exist_ok=True)
    esc = ds == "esc50"
    urban = task == "urban"

    # ── cheap sources ───────────────────────────────────────────────────────
    sources = {}

    def add(src, root, tag, cfg, dim, side, hull_set="stage3c", kinds=KINDS):
        dirs = {(s, k): A.ScoreDir(root / tag.format(s) / k) for s in seeds for k in kinds}
        have = [k for k in kinds if all(dirs[(s, k)].files for s in seeds)]
        if not have:
            raise SystemExit(f"{task}: source {src} has no scores under {root}")
        sources[src] = {"dirs": dirs, "cfg": cfg, "dim": dim, "side": side, "kinds": have, "set": hull_set}

    add("eff", a.s2 / "scores" / task, f"effp_{task}_s{{}}", {"fp32": "fp32", "w8a8": "w8a8"}, 1536,
        {bb: a.s2 / "emb" / ds / f"{bb}__clean.json" for bb in ("fp32", "w8a8")})
    for m, d in CNN_S3B.items():
        name = "mn10pad" if urban and m == "mn10" else m
        add(name, a.s3b / "scores_cnn" / task, f"{m}_{task}_s{{}}", {bb: f"{m}_{bb}" for bb in ("fp32", "w8a8")}, d,
            {bb: a.s3b / "emb_cnn" / ds / f"{m}_{bb}__clean.json" for bb in ("fp32", "w8a8")},
            hull_set="excluded" if name == "mn10pad" else "stage3c")
    if urban:
        add("mn10unpr", a.unp / "scores_unpr" / "urban", "mn10unp_urban_s{}", {bb: f"mn10_{bb}" for bb in ("fp32", "w8a8")},
            960, {bb: a.s3c / "emb_cnn" / "urban8k_unpadded" / f"mn10_{bb}__clean.json" for bb in ("fp32", "w8a8")})
    for m, d in CNN_NEW.items():
        add(m, a.s3c / "scores_cnn" / task, f"{m}_{task}_s{{}}", {bb: f"{m}_{bb}" for bb in ("fp32", "w8a8")}, d,
            {bb: a.s3c / "emb_cnn" / ds / f"{m}_{bb}__clean.json" for bb in ("fp32", "w8a8")})

    def ren(x):
        return "mn10unpr" if urban and x == "mn10" else x

    def cheap_get(src, bb, s, k, cond):
        return sources[src]["dirs"][(s, k)].get(sources[src]["cfg"][bb], cond)

    def cheap_has(src, bb, cond):
        S_ = sources[src]
        return all(S_["dirs"][(s, k)].has(S_["cfg"][bb], cond) for s in seeds for k in S_["kinds"])

    # ── AST tiers (Stage 3c's, then the new ones) ───────────────────────────
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
    if esc and not a.no_mean2:
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
    new_tiers = [f"{c}@{SL}" for c in NEW_TR] + [f"{c}o@{SL}" for c in NEW_OWN]
    own_tiers = [f"{c}o@{SL}" for c in NEW_OWN]
    tier8 = f"w8g64@{SL}"

    def ast_has(name, cond):
        dirs, base, _ = src_ast[name]
        return all(d.has(base, cond) for d in dirs.values())

    def ast_get(name, s, h, cond):
        dirs, base, _ = src_ast[name]
        return dirs[(s, h)].get(base, cond)

    conds = CHEAP_CONDS + ["field"]
    cheap_all = [(x, bb) for x in sources for bb in sources[x]["cfg"]]
    cheap_conds = {cb: [c for c in conds if cheap_has(*cb, c)] for cb in cheap_all}
    ast_conds = {n: [c for c in conds if ast_has(n, c)] for n in src_ast}
    missing = [f"{x}:{bb}" for x, bb in cheap_all if not all(c in cheap_conds[(x, bb)] for c in CASC_CONDS)]
    if missing:
        raise SystemExit(f"{task}: incomplete systems on clean or field: {missing}")
    old_tiers = [n for n in FULL_TIERS if n in src_ast] + [f"{c}@{SL}" for c in S2_SHORT]
    old_tiers += [n for n, v in src_ast.items() if v[2] in ("a256", "a384", "m2")]
    tiers = old_tiers + new_tiers
    bad = [t for t in tiers if not all(c in ast_conds[t] for c in CASC_CONDS)]
    if bad and a.check_only and all(t in new_tiers for t in bad):
        print(f"{task}: check-only, dropping incomplete new systems {bad}", flush=True)
        for t in bad:
            src_ast.pop(t)
            ast_conds.pop(t)
        new_tiers = [t for t in new_tiers if t not in bad]
        own_tiers = [t for t in own_tiers if t not in bad]
        tiers = old_tiers + new_tiers
        bad = []
    if bad:
        raise SystemExit(f"{task}: tiers lacking clean or field (every new system must be complete): {bad}")
    p3_cheap = [(ren(x), bb) for x, bb in P3_CHEAP]
    c_cheap = [(ren(x), bb) for x, bb in C_CHEAP]
    cheap_casc = p3_cheap
    print(f"{task}: cheap sources {list(sources)}, {len(src_ast)} AST systems, tiers {len(tiers)} "
          f"({len(new_tiers)} new), price {a.price}", flush=True)

    # ── costs ───────────────────────────────────────────────────────────────
    from costs_idea6 import head_costs

    with contextlib.redirect_stdout(io.StringIO()):
        dims = sorted({1536, 768, *(S_["dim"] for S_ in sources.values())})
        hc = {d: head_costs(d, KINDS)[task] for d in dims}
    eff_emb = a.s2 / "emb" / ds
    lengths = np.load(eff_emb / "lengths__clean.npy")
    fbank = float(np.mean([A.ast_fbank_frontend_macs(int(n)) for n in lengths])) * 1024
    cheap_cost, crc_side = {}, {}
    bb_key = "backbone_bitops_rescale1024" if priced else "backbone_bitops"
    for x, S_ in sources.items():
        for bb, sp in S_["side"].items():
            sd = json.loads(sp.read_text())
            cc = sd["cost_per_clip"]
            crc_side[(x, bb)] = sd["names_crc32"]
            for k in S_["kinds"]:
                cheap_cost[(x, bb, k)] = cc["frontend_bitops_mean"] + cc[bb_key] + hc[S_["dim"]][k]["bitops"]
    ast_cost, ast_side = {}, {}
    from idea5b_common import bitops as ast_bitops

    side_dirs = {"s2short": a.s2 / "emb_ast_short" / ds, "a256": a.s3b / "emb_ast" / f"{ds}_256",
                 "a384": a.s3b / "emb_ast" / f"{ds}_384", "m2": a.s3b / "emb_ast" / f"{ds}_256x2",
                 "tr": a.r / "emb_ast_short" / ds}
    for n, (dirs, base, g) in src_ast.items():
        if g == "full":
            bo = ast_bitops(base, a.price)
        else:
            L = {"s2short": SL, "a256": 256, "a384": 384, "m2": 256, "tr": SL, "own": SL}[g]
            P_, T_ = A.short_tokens(L)
            mult = 2 if g == "m2" else 1
            bo = mult * A.ast_bitops_tok(base, a.price, T_, P_)
            sdir = side_dirs["s2short"] if g == "own" and base == "w3g64" else side_dirs["tr" if g == "own" else g]
            s_ = json.loads((sdir / f"{base}__clean.json").read_text())
            un = mult * A.ast_bitops_tok(base, 0.0, T_, P_)
            if abs(s_["backbone_cost_per_clip"]["bitops"] - un) > 1e-6 * un or s_["max_length"] != L:
                raise SystemExit(f"{n}: sidecar bit-ops differ")
            ast_side[n] = s_["names_crc32"]
        for h in AST_HEADS:
            ast_cost[(n, h)] = fbank + bo + hc[768][h]["bitops"]
    crc_ref = json.loads((eff_emb / "fp32__clean.json").read_text())["names_crc32"]
    if any(v != crc_ref for v in crc_side.values()) or any(v != crc_ref for v in ast_side.values()):
        raise SystemExit(f"{task}: a names checksum differs: {crc_side} {ast_side}")
    (out / "costs.json").write_text(json.dumps({
        "price": a.price, "cheap": {cheap_name(*kk): v for kk, v in cheap_cost.items()},
        "ast": {f"{n}+{h}": v for (n, h), v in ast_cost.items()}, "fbank_bitops_mean": fbank,
        "heads": {str(d): v for d, v in hc.items()}}, indent=1))

    def sys_cost(name):
        if name.startswith("ast:"):
            n, h = name[4:].split("+")
            return ast_cost[(n, h)]
        parts = name.split(":")
        return cheap_cost[("eff", parts[1], parts[2]) if parts[0] == "eff" else (parts[1], parts[2], parts[3])]

    def sys_set(name):
        if name.startswith("ast:"):
            return ast_set[name[4:].split("+")[0]]
        if name.startswith("eff:"):
            return "stage3c"
        return sources[name.split(":")[1]]["set"]

    # ── one pass over seeds and folds ───────────────────────────────────────
    static, sel_rows, casc, esc_rows, sel3_rows, casc3 = [], [], [], [], [], []
    preds, ys, yiv, iv_static, ivx = {}, {}, {}, {}, {}
    acs, escs, decs, au = {}, {}, {}, {}
    thr_all, thr2_all = {}, {}
    ref0 = cheap_get("eff", "fp32", seeds[0], "esc", "clean")
    folds = sorted(ref0["iv"])
    n_te = np.array([np.sum(ref0["fold"] == f) for f in folds])
    A.NB_RATIO["task"] = float(np.mean(n_te / (len(ref0["y"]) - n_te)))

    def pred3(p, p3, p8, e1, e2, r):
        if r == "ast":
            return np.where(e2, p8.argmax(1), np.where(e1, p3.argmax(1), p.argmax(1)))
        return np.where(e2, ((p + p3 + p8) / 3).argmax(1), np.where(e1, ((p + p3) / 2).argmax(1), p.argmax(1)))

    for s in seeds:
        ref = cheap_get("eff", "fp32", s, "esc", "clean")
        for x, bb in cheap_all:
            for k in sources[x]["kinds"]:
                for c in CASC_CONDS:
                    r_ = cheap_get(x, bb, s, k, c)
                    if not ((r_["y"] == ref["y"]).all() and (r_["fold"] == ref["fold"]).all() and sorted(r_["iv"]) == folds
                            and all(np.array_equal(r_["iv"][f][0], ref["iv"][f][0]) for f in folds)):
                        raise SystemExit(f"{x}:{bb}:{k} {c}: rows differ")
                    if r_["crc"] is not None and r_["crc"] != crc_ref:
                        raise SystemExit(f"{x}:{bb}:{k} {c}: names checksum differs")
        for n in src_ast:
            for h in AST_HEADS:
                for c in [c for c in CASC_CONDS if c in ast_conds[n]]:
                    r_ = ast_get(n, s, h, c)
                    if not ((r_["y"] == ref["y"]).all() and (r_["fold"] == ref["fold"]).all()
                            and all(np.array_equal(r_["iv"][f][0], ref["iv"][f][0]) for f in folds)):
                        raise SystemExit(f"{n}+{h} {c}: rows differ (inner split or folds)")
        y_all = ref["y"]
        for fold in folds:
            te = np.flatnonzero(ref["fold"] == fold)
            iv = ref["iv"][fold][0]
            y, yv = y_all[te], y_all[iv]
            key = (s, fold)
            ys[key], yiv[key], ivx[key] = y, yv, iv
            civ = {}
            for x, bb in cheap_all:
                for k in sources[x]["kinds"]:
                    p = A.softmax(cheap_get(x, bb, s, k, "clean")["iv"][fold][1])
                    civ[(x, bb, k)] = p
                    pr = p.argmax(1)
                    iv_static[(cheap_name(x, bb, k), s, fold)] = (A.f1_fast(yv, pr, K), pr)
            aiv = {}
            for n in src_ast:
                for h in AST_HEADS:
                    p = A.softmax(ast_get(n, s, h, "clean")["iv"][fold][1])
                    aiv[(n, h)] = p
                    pr = p.argmax(1)
                    iv_static[(f"ast:{n}+{h}", s, fold)] = (A.f1_fast(yv, pr, K), pr)
            for e3 in own_tiers:
                m3 = A.margin(aiv[(e3, "esc")])
                for b2 in B2:
                    thr2_all[(e3, b2, s, fold)] = A.thr_rate(m3, b2)
            for x, bb in cheap_casc:
                for k in sources[x]["kinds"]:
                    p = civ[(x, bb, k)]
                    m, ac = A.margin(p), p.argmax(1)
                    for b, rate in BUDGETS.items():
                        t_ = A.thr_rate(m, rate)
                        thr_all[(x, bb, k, b, s, fold)] = t_
                        escm = m < t_
                        for e in tiers:
                            pe = aiv[(e, "esc")]
                            for r in RULES:
                                de = pe.argmax(1) if r == "ast" else ((p + pe) / 2).argmax(1)
                                sel_rows.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r,
                                                 "budget": b, "seed": s, "fold": fold, "iv_rate": float(escm.mean()),
                                                 "f1_iv": A.f1_fast(yv, np.where(escm, de, ac), K)})
                        if (x, bb) not in c_cheap:
                            continue
                        p8 = aiv[(tier8, "esc")]
                        for e3 in own_tiers:
                            p3 = aiv[(e3, "esc")]
                            m3 = A.margin(p3)
                            for b2 in B2:
                                e2 = escm & (m3 < thr2_all[(e3, b2, s, fold)])
                                for r in RULES:
                                    sel3_rows.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "mid": e3, "top": tier8,
                                                      "rule": r, "budget": b, "b2": b2, "seed": s, "fold": fold,
                                                      "iv_rate1": float(escm.mean()), "iv_rate2": float(e2.mean()),
                                                      "f1_iv": A.f1_fast(yv, pred3(p, p3, p8, escm, e2, r), K)})
            for c in conds:
                cp = {}
                for x, bb in cheap_all:
                    if c not in cheap_conds[(x, bb)]:
                        continue
                    for k in sources[x]["kinds"]:
                        z = cheap_get(x, bb, s, k, c)["z"][te]
                        p = A.softmax(z)
                        pr = p.argmax(1)
                        name = cheap_name(x, bb, k)
                        if c in CASC_CONDS:
                            preds[(name, c, s, fold)] = pr.astype(np.int16)
                        static.append({"system": name, "condition": c, "seed": s, "fold": fold,
                                       "f1": A.f1_fast(y, pr, K), "acc": float((pr == y).mean()),
                                       "cost": cheap_cost[(x, bb, k)]})
                        if (x, bb) in cheap_casc:
                            cp[(x, bb, k)] = p
                            m = A.margin(p)
                            for b in BUDGETS:
                                esc_rows.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "budget": b, "condition": c,
                                                 "seed": s, "fold": fold,
                                                 "rate": float((m < thr_all[(x, bb, k, b, s, fold)]).mean()),
                                                 "error": float((pr != y).mean())})
                            a_ = au.setdefault((x, bb, k, c, s), {"score": [], "wrong": []})
                            a_["score"].append(1 - m)
                            a_["wrong"].append(pr != y)
                ap_ = {}
                for n in src_ast:
                    if c not in ast_conds[n]:
                        continue
                    for h in AST_HEADS:
                        p = A.softmax(ast_get(n, s, h, c)["z"][te])
                        pr = p.argmax(1)
                        name = f"ast:{n}+{h}"
                        if c in CASC_CONDS:
                            preds[(name, c, s, fold)] = pr.astype(np.int16)
                        static.append({"system": name, "condition": c, "seed": s, "fold": fold,
                                       "f1": A.f1_fast(y, pr, K), "acc": float((pr == y).mean()),
                                       "cost": ast_cost[(n, h)]})
                        if h == "esc":
                            ap_[n] = p
                if c not in CASC_CONDS:
                    continue
                for (x, bb, k), p in cp.items():
                    m, ac = A.margin(p), p.argmax(1)
                    acs[(x, bb, k, c, s, fold)] = ac.astype(np.int16)
                    qs = {}
                    for e in tiers:
                        pe = ap_[e]
                        for r in RULES:
                            q = pe if r == "ast" else (p + pe) / 2
                            qs[(e, r)] = (q.argmax(1), q.max(1))
                            decs[(x, bb, k, e, r, c, s, fold)] = qs[(e, r)][0].astype(np.int16)
                    for b in BUDGETS:
                        escm = m < thr_all[(x, bb, k, b, s, fold)]
                        escs[(x, bb, k, b, c, s, fold)] = escm
                        rate = float(escm.mean())
                        for (e, r), (de, qmax) in qs.items():
                            pred = np.where(escm, de, ac)
                            casc.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r,
                                         "budget": b, "condition": c, "seed": s, "fold": fold,
                                         "f1": A.f1_fast(y, pred, K), "acc": float((pred == y).mean()), "rate": rate,
                                         "ece": A.ece_conf(np.where(escm, qmax, p.max(1)), pred == y),
                                         "cost": cheap_cost[(x, bb, k)] + rate * ast_cost[(e, "esc")]})
                        if (x, bb) not in c_cheap:
                            continue
                        p8 = ap_[tier8]
                        for e3 in own_tiers:
                            p3 = ap_[e3]
                            m3 = A.margin(p3)
                            for b2 in B2:
                                e2 = escm & (m3 < thr2_all[(e3, b2, s, fold)])
                                r2 = float(e2.mean())
                                for r in RULES:
                                    pred = pred3(p, p3, p8, escm, e2, r)
                                    casc3.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "mid": e3, "top": tier8,
                                                  "rule": r, "budget": b, "b2": b2, "condition": c, "seed": s, "fold": fold,
                                                  "f1": A.f1_fast(y, pred, K), "acc": float((pred == y).mean()),
                                                  "rate1": rate, "rate2": r2,
                                                  "cost": cheap_cost[(x, bb, k)] + rate * ast_cost[(e3, "esc")]
                                                  + r2 * ast_cost[(tier8, "esc")]})
            print(f"{task} seed {s} fold {fold}: {time.time() - t0:.0f}s", flush=True)
        for S_ in sources.values():
            for d in S_["dirs"].values():
                d.cache.clear()
        for n, (dirs, _, _) in src_ast.items():
            for d in dirs.values():
                d.cache.clear()

    S = pd.DataFrame(static)
    S["set"] = S.system.map(sys_set)
    S.to_csv(out / "static.csv", index=False)
    CA = pd.DataFrame(casc)
    CA.to_csv(out / "cascades.csv.gz", index=False)
    CA3 = pd.DataFrame(casc3)
    CA3.to_csv(out / "cascades3.csv.gz", index=False)
    pd.DataFrame(esc_rows).to_csv(out / "escalation.csv", index=False)
    rows = []
    for (x, bb, k, c, s), a_ in au.items():
        sc, w = np.concatenate(a_["score"]), np.concatenate(a_["wrong"])
        rows.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "condition": c, "seed": s,
                     "auroc_wrong": roc_auc_score(w, sc) if 0 < w.sum() < len(w) else np.nan,
                     "error": float(w.mean())})
    pd.DataFrame(rows).to_csv(out / "auroc_wrong.csv", index=False)

    # ── selection on clean inner validation ─────────────────────────────────
    SR = pd.DataFrame(sel_rows)
    SR3 = pd.DataFrame(sel3_rows)
    st = pd.DataFrame([{"system": nm, "seed": s, "fold": f, "f1": v[0], "cost": sys_cost(nm), "set": sys_set(nm)}
                       for (nm, s, f), v in iv_static.items()])
    sp_iv_all = st.groupby(["system", "set"]).agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
    sp_iv_all.to_csv(out / "static_iv.csv", index=False)

    def iv_hull(sp_iv, rng, g, c_iv):
        v_interp, lo, hi, lam = A.hull_pair(sp_iv, c_iv)
        hull = []
        for s, f in zip(g.seed, g.fold):
            plo = iv_static[(lo, s, f)][1]
            if lam == 0.0:
                hull.append(iv_static[(lo, s, f)][0])
                continue
            phi = iv_static[(hi, s, f)][1]
            draws = rng.random((A.DRAWS, len(plo))) < lam
            hull.append(float(A.f1_batch(yiv[(s, f)], np.where(draws, phi[None, :], plo[None, :]), K).mean()))
        return lo, hi, lam, np.array(hull)

    DCOLS = ["cheap_src", "cheap_bb", "cheap", "expensive", "rule", "budget"]

    def select(cheap_fam, tier_set, which, fname):
        """Two-tier selection; groups outside the family consume no random draws (E3 needs Stage 3c's sequence)."""
        sp_iv = sp_iv_all[sp_iv_all["set"].isin(SET_ORDER[which])]
        rng = np.random.default_rng(A.RNG_IV)
        sel_out = []
        fam = set(cheap_fam)
        for (x, bb, k, e, r, b), g in SR.groupby(DCOLS, sort=False):
            if (x, bb) not in fam or e not in tier_set:
                continue
            g = g.sort_values(["seed", "fold"])
            c_iv = float(np.mean(cheap_cost[(x, bb, k)] + g.iv_rate.to_numpy() * ast_cost[(e, "esc")]))
            lo, hi, lam, hull = iv_hull(sp_iv, rng, g, c_iv)
            d = g.f1_iv.to_numpy() - hull
            sel_out.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r, "budget": b,
                            "f1_iv": g.f1_iv.mean(), "iv_rate": g.iv_rate.mean(), "cost_iv": c_iv, "hull_lo": lo,
                            "hull_hi": hi, "lam": lam, "hull_f1_iv": float(np.mean(hull)), "gain_iv": float(d.mean())})
        SO = pd.DataFrame(sel_out).sort_values(["gain_iv", "cost_iv"], ascending=[False, True])
        SO.to_csv(out / fname, index=False)
        top = SO.iloc[0]
        return SO, tuple(str(top[c]) for c in DCOLS)

    def select3(fname):
        sp_iv = sp_iv_all[sp_iv_all["set"].isin(SET_ORDER["new"])]
        rng = np.random.default_rng(A.RNG_IV)
        sel_out = []
        for (x, bb, k, e3, r, b, b2), g in SR3.groupby(["cheap_src", "cheap_bb", "cheap", "mid", "rule", "budget", "b2"], sort=False):
            g = g.sort_values(["seed", "fold"])
            c_iv = float(np.mean(cheap_cost[(x, bb, k)] + g.iv_rate1.to_numpy() * ast_cost[(e3, "esc")]
                                 + g.iv_rate2.to_numpy() * ast_cost[(tier8, "esc")]))
            lo, hi, lam, hull = iv_hull(sp_iv, rng, g, c_iv)
            d = g.f1_iv.to_numpy() - hull
            sel_out.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "mid": e3, "top": tier8, "rule": r, "budget": b,
                            "b2": b2, "f1_iv": g.f1_iv.mean(), "iv_rate1": g.iv_rate1.mean(), "iv_rate2": g.iv_rate2.mean(),
                            "cost_iv": c_iv, "hull_lo": lo, "hull_hi": hi, "lam": lam, "hull_f1_iv": float(np.mean(hull)),
                            "gain_iv": float(d.mean())})
        SO = pd.DataFrame(sel_out).sort_values(["gain_iv", "cost_iv"], ascending=[False, True])
        SO.to_csv(out / fname, index=False)
        top = SO.iloc[0]
        return SO, (top.cheap_src, top.cheap_bb, str(top.cheap), top.mid, top.rule, top.budget, float(top.b2))

    SO_e3, p_e3 = select(p3_cheap, set(old_tiers), "stage3c", "selection_stage3c.csv")
    SO_h, pick = select(c_cheap, set(tiers), "new", "selection.csv")
    if own_tiers:
        SO3, pick3 = select3("selection3.csv")
    else:
        SO3, pick3 = None, None
    # the best two-tier design with the E1 pick's cheap tier, by the same selection
    if pick3 is not None:
        same = SO_h[(SO_h.cheap_src == pick3[0]) & (SO_h.cheap_bb == pick3[1]) & (SO_h.cheap.astype(str) == pick3[2])]
        pick_same = tuple(str(same.iloc[0][c]) for c in DCOLS)
    ref_pick_src = a.unp if urban else a.s3c
    ref_picks = json.loads((ref_pick_src / "analysis" / task / "picks.json").read_text())["P3"]
    ref_design = tuple(str(v) for v in ref_picks["design"])
    e3_ok = p_e3 == ref_design and abs(float(SO_e3.iloc[0].gain_iv) - ref_picks["gain_iv"]) < 1e-9
    print(f"{task}: E3 reselection {p_e3} gain {SO_e3.iloc[0].gain_iv:.6f}; reference {ref_design} gain "
          f"{ref_picks['gain_iv']:.6f}: {'equal' if e3_ok else 'DIFFERENT'}", flush=True)
    if not e3_ok and priced:
        raise SystemExit(f"{task}: E3 does not reproduce the reference pick")
    top = SO_h.iloc[0]
    SO_n = SO_h[SO_h.expensive.isin(new_tiers)]
    picks = {"H1": {"design": list(pick), "gain_iv": float(top.gain_iv), "runner_up_gain_iv": float(SO_h.iloc[1].gain_iv),
                    "runner_up": [str(v) for v in SO_h.iloc[1][DCOLS]], "cost_iv": float(top.cost_iv),
                    "f1_iv": float(top.f1_iv), "iv_rate": float(top.iv_rate), "hull_iv": [top.hull_lo, top.hull_hi, float(top.lam)],
                    "n_designs": int(len(SO_h)), "tier_is_new": pick[3] in new_tiers,
                    "best_new_tier_design": [str(v) for v in SO_n.iloc[0][DCOLS]] if len(SO_n) else None,
                    "best_new_tier_gain_iv": float(SO_n.iloc[0].gain_iv) if len(SO_n) else None},
             "E1": None if pick3 is None else
                   {"design": [str(v) for v in pick3], "gain_iv": float(SO3.iloc[0].gain_iv), "n_designs": int(len(SO3)),
                    "cost_iv": float(SO3.iloc[0].cost_iv), "iv_rate1": float(SO3.iloc[0].iv_rate1),
                    "iv_rate2": float(SO3.iloc[0].iv_rate2), "two_tier_same_cheap": list(pick_same),
                    "two_tier_same_cheap_gain_iv": float(same.iloc[0].gain_iv)},
             "E3": {"design": list(p_e3), "gain_iv": float(SO_e3.iloc[0].gain_iv), "reference": list(ref_design),
                    "reference_gain_iv": ref_picks["gain_iv"], "reference_folder": str(ref_pick_src), "equal": bool(e3_ok)}}
    (out / "picks.json").write_text(json.dumps(picks, indent=1))
    print(f"{task}: H1 pick {pick} (gain_iv {top.gain_iv:.4f}); E1 pick {pick3}", flush=True)

    # ── test comparisons ────────────────────────────────────────────────────
    from analyze import AudioStats, field_arrays, fit_lr, top_k, oracle_pred  # noqa: E402

    stems = [str(x) for x in np.load(a.i7 / "audio_stats" / T["stats"] / "clean.npz", allow_pickle=True)["names"]]
    audio = AudioStats(a.i7 / "audio_stats" / T["stats"], stems)
    test_idx = {f: np.flatnonzero(ref0["fold"] == f) for f in folds}
    noise = {}
    av = field_arrays(len(ref0["y"]))[1]
    for (s, f), iv in ivx.items():
        noise[(s, f)] = fit_lr(audio.rows("field", iv, "val"), (av[iv] != "clean").astype(int))
    s_rec = {(n, c): dict(zip(zip(g.seed, g.fold), g.f1)) for (n, c), g in S.groupby(["system", "condition"])}
    CAg = {kk: g.sort_values(["seed", "fold"]) for kk, g in CA.groupby(DCOLS + ["condition"])}
    CA3g = {kk: g.sort_values(["seed", "fold"]) for kk, g in
            CA3.groupby(["cheap_src", "cheap_bb", "cheap", "mid", "rule", "budget", "b2", "condition"])} if len(CA3) else {}
    sp_cache = {}

    def hull_for(c, cost, which):
        if (c, which) not in sp_cache:
            g = S[(S.condition == c) & S["set"].isin(SET_ORDER[which])]
            sp_cache[(c, which)] = g.groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
        sp = sp_cache[(c, which)]
        v_interp, lo, hi, lam = A.hull_pair(sp, cost)
        ok = sp[sp.cost <= cost * (1 + 1e-12)].sort_values(["f1", "cost"], ascending=[False, True])
        best_any = sp.sort_values(["f1", "cost"], ascending=[False, True]).iloc[0]
        return v_interp, lo, hi, lam, (ok.iloc[0].system if len(ok) else None), (best_any.system, float(best_any.f1))

    def hull_f1(c, lo, hi, lam, kk):
        if lam == 0.0:
            return s_rec[(lo, c)][kk]
        return float(mix_expected_f1(ys[kk], preds[(lo, c, *kk)].astype(np.int64),
                                     preds[(hi, c, *kk)].astype(np.int64), lam, K)[0])

    def compare(design, c, which, baselines=False):
        x, bb, k, e, r, b = design
        g = CAg[(x, bb, k, e, r, b, c)]
        keys = list(zip(g.seed, g.fold))
        xv = g.f1.to_numpy()
        cost = float(g.cost.mean())
        v_interp, lo, hi, lam, best, best_any = hull_for(c, cost, which)
        hull = np.array([hull_f1(c, lo, hi, lam, kk) for kk in keys])
        row = {"hull_set": which, "cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r, "budget": b,
               "condition": c, "f1": float(xv.mean()), "cost": cost, "rate": float(g.rate.mean()),
               "rate_min": float(g.rate.min()), "rate_max": float(g.rate.max()), "ece": float(g.ece.mean()),
               "hull_lo": lo, "hull_hi": hi, "lam": lam, "hull_f1_interp": v_interp, "hull_f1": float(hull.mean())}
        row.update({f"hull_{kk}": vv for kk, vv in A.tests(xv, hull, keys).items()})
        if not baselines:
            return row
        rnd, orc, orc_full, aud = [], [], [], []
        for kk in keys:
            pc = acs[(x, bb, k, c, *kk)].astype(np.int64)
            de = decs[(x, bb, k, e, r, c, *kk)].astype(np.int64)
            escm = escs[(x, bb, k, b, c, *kk)]
            yk = ys[kk]
            rnd.append(float(mix_expected_f1(yk, pc, de, escm.mean(), K)[0]))
            orc.append(A.f1_fast(yk, oracle_pred(yk, pc, de, int(escm.sum())), K))
            fix = (pc != yk) & (de == yk)
            orc_full.append(A.f1_fast(yk, np.where(fix, de, pc), K))
            if audio.available(c):
                sc = noise[kk](audio.rows(c, test_idx[kk[1]], "test"))
                aud.append(A.f1_fast(yk, np.where(top_k(sc, int(escm.sum())), de, pc), K))
        row["random_f1"] = float(np.mean(rnd))
        row.update({f"random_{kk}": vv for kk, vv in A.tests(xv, rnd, keys).items()})
        row["oracle_k_f1"], row["oracle_full_f1"] = float(np.mean(orc)), float(np.mean(orc_full))
        row["best_static"] = best
        zb = np.array([s_rec[(best, c)][kk] for kk in keys])
        row["best_static_f1"], row["best_static_cost"] = float(zb.mean()), sys_cost(best)
        row.update({f"static_{kk}": vv for kk, vv in A.tests(xv, zb, keys).items()})
        row["best_static_any"], row["best_static_any_f1"] = best_any
        row["best_static_any_cost"] = sys_cost(best_any[0])
        if aud:
            row["audio_f1"] = float(np.mean(aud))
            row.update({f"audio_{kk}": vv for kk, vv in A.tests(xv, aud, keys).items()})
        return row

    res = []
    ref_sel = pd.read_csv(ref_pick_src / "analysis" / task / "selected.csv")
    ref_sel = ref_sel[(ref_sel.family == "P3") & (ref_sel.hull_set == "new")]
    repro = {}
    for c in CASC_CONDS:
        row = compare(p_e3, c, "stage3c", baselines=True)
        row["family"], row["base_design"] = "E3", "stage3c_P3"
        res.append(row)
        ref = ref_sel[ref_sel.condition == c].iloc[0]
        repro[c] = {k: float(abs(row[k] - ref[k]) / (abs(ref[k]) if k == "cost" else 1.0))
                    for k in ("f1", "cost", "hull_f1", "hull_diff", "hull_p_t_fold")}
    worst = max(v for d in repro.values() for v in d.values())
    print(f"{task}: E3 reproduction of the reference rows, largest difference {worst:.2e}", flush=True)
    if worst > 1e-9 and priced:
        raise SystemExit(f"{task}: E3 rows do not reproduce the reference: {repro}")
    for fam, c in (("H1", "field"), ("H2", "clean")):
        row = compare(pick, c, "new", baselines=True)
        row["family"], row["base_design"] = fam, "H1"
        res.append(row)
    for c in CASC_CONDS if pick3 is not None else ():
        row = compare(pick_same, c, "new", baselines=True)
        row["family"], row["base_design"] = "E1_two_tier_same_cheap", "E1"
        res.append(row)
    R_ = pd.DataFrame(res)

    # three tiers (E1)
    def compare3(design, c, which):
        x, bb, k, e3, r, b, b2 = design
        g = CA3g[(x, bb, k, e3, r, b, b2, c)]
        keys = list(zip(g.seed, g.fold))
        xv = g.f1.to_numpy()
        cost = float(g.cost.mean())
        v_interp, lo, hi, lam, best, best_any = hull_for(c, cost, which)
        hull = np.array([hull_f1(c, lo, hi, lam, kk) for kk in keys])
        row = {"family": "E1", "hull_set": which, "cheap_src": x, "cheap_bb": bb, "cheap": k, "mid": e3, "top": tier8,
               "rule": r, "budget": b, "b2": b2, "condition": c, "f1": float(xv.mean()), "cost": cost,
               "rate1": float(g.rate1.mean()), "rate2": float(g.rate2.mean()), "hull_lo": lo, "hull_hi": hi, "lam": lam,
               "hull_f1_interp": v_interp, "hull_f1": float(hull.mean()), "best_static": best}
        row.update({f"hull_{kk}": vv for kk, vv in A.tests(xv, hull, keys).items()})
        for nm, d2 in (("h1pick", pick), ("same_cheap", pick_same)):
            g2 = CAg[(*d2, c)]
            zv = g2.f1.to_numpy()
            row[f"{nm}_f1"], row[f"{nm}_cost"] = float(zv.mean()), float(g2.cost.mean())
            row.update({f"{nm}_{kk}": vv for kk, vv in A.tests(xv, zv, list(zip(g2.seed, g2.fold))).items()})
        return row

    if pick3 is not None:
        R3 = pd.DataFrame([compare3(pick3, c, "new") for c in CASC_CONDS])
        R3.to_csv(out / "selected3.csv", index=False)
    R_.to_csv(out / "selected.csv", index=False)

    # E2: every design of this run's family on the field mixture
    e2 = [compare(d, "field", "new") for d in
          [(x, bb, k, e, r, b) for x, bb in c_cheap for k in sources[x]["kinds"] for e in tiers for r in RULES for b in BUDGETS]]
    pd.DataFrame(e2).to_csv(out / "e2.csv", index=False)
    print(f"{task}: E2 done, {time.time() - t0:.0f}s", flush=True)

    # S5: the new tiers as statics, own against transferred, and against w4g128 at the same length
    s5 = []
    for cfg in NEW_OWN:
        for h in AST_HEADS:
            for c in CASC_CONDS:
                own = f"ast:{cfg}o@{SL}+{h}"
                for ref_name, label in ((f"ast:{cfg}@{SL}+{h}", "transferred"), (f"ast:w4g128@{SL}+{h}", "w4g128")):
                    if (own, c) not in s_rec or (ref_name, c) not in s_rec:
                        continue
                    keys = sorted(s_rec[(own, c)])
                    xv = np.array([s_rec[(own, c)][kk] for kk in keys])
                    zv = np.array([s_rec[(ref_name, c)][kk] for kk in keys])
                    row = {"tier": cfg, "head": h, "condition": c, "against": label, "f1_own": float(xv.mean()),
                           "f1_ref": float(zv.mean()), "cost_own": sys_cost(own), "cost_ref": sys_cost(ref_name)}
                    tt = A.tests(xv, zv, keys)
                    row.update(tt)
                    fm = np.array(json.loads(tt["fold_diffs"]))
                    row["ci95_lo"], row["ci95_hi"] = ci95(fm)
                    s5.append(row)
    pd.DataFrame(s5).to_csv(out / "s5_new_tiers.csv", index=False)

    # S2: "cheap wrong, tier right" AUROC for the H1 pick's cheap tier
    rows = []
    x, bb, k, e, r, b = pick
    for c in conds:
        if c not in ast_conds[e] or c not in cheap_conds[(x, bb)]:
            continue
        for s in seeds:
            sc_, fx_ = [], []
            for f in folds:
                te = test_idx[f]
                p = A.softmax(cheap_get(x, bb, s, k, c)["z"][te])
                pe = A.softmax(ast_get(e, s, "esc", c)["z"][te])
                yk = ref0["y"][te]
                sc_.append(1 - A.margin(p))
                fx_.append((p.argmax(1) != yk) & (pe.argmax(1) == yk))
            sc_, fx_ = np.concatenate(sc_), np.concatenate(fx_)
            rows.append({"family": "H1", "cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r,
                         "budget": b, "condition": c, "seed": s, "n_fixable": int(fx_.sum()),
                         "auroc_fixable": roc_auc_score(fx_, sc_) if 0 < fx_.sum() < len(fx_) else np.nan})
    pd.DataFrame(rows).to_csv(out / "auroc_fixable.csv", index=False)
    meta = {"task": task, "K": K, "seeds": seeds, "folds": folds, "nb_ratio": A.NB_RATIO["task"], "price": a.price,
            "tiers": tiers, "new_tiers": new_tiers, "own_tiers": own_tiers, "tier8": tier8,
            "sources": {x: {"kinds": v["kinds"], "set": v["set"], "configs": v["cfg"]} for x, v in sources.items()},
            "ast_conditions": ast_conds, "cheap_conditions": {f"{x}:{bb}": v for (x, bb), v in cheap_conds.items()},
            "picks": picks, "e3_reproduction": repro, "n_designs_h1": int(len(SO_h)),
            "n_designs_e1": int(len(SO3)) if SO3 is not None else 0, "check_only": a.check_only,
            "mean2": "m2" in groups, "seconds": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{task}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
