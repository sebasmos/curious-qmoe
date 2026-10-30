"""Idea 6 Stage 3b analysis of one task, as fixed in ANALYSIS_PLAN.md (commit
5280b3b): the hull of every static system (Stage 2's and the new ones: AST n4
and w4g128 at 256 and 384 frames, the two-window mean on ESC-50, EfficientAT
mn04 and mn10 at FP32 and W8A8), P1 (Stage 2's H1 design against it), P2 (one
design selected on clean inner validation from a family whose cheap tier is
the W8A8 EfficientNet or a W8A8 EfficientAT), and the secondary outputs.

Stage 2's helpers are imported unchanged from scripts/redesign/idea6/stage2
(analyze.py: metrics, thresholds, hull, tests, score loading; idea5b_common.py:
the AST bit-op count and the exact expected F1 of a random mix).

Inputs (per-clip logits of test folds and inner validation, same rows):
  Stage 2       --s2   eff heads scores/<task>, short AST scores_ast_short/<task>,
                       (esc5off) scores_ast1024/esc5off, embedding sidecars
  idea 5 St. 2  --i5   1024-frame AST scores/<task> and sidecars
  this stage    --r    scores_cnn/<task>, scores_ast256/<task>, scores_ast384/<task>,
                       scores_astm2/<task>, emb_cnn/<ds>, emb_ast/<ds>_<len> sidecars
  idea 7        --i7   audio statistics (the noise_lr gate)

    python scripts/redesign/idea6/stage3b/analyze_3b.py --task esc50 --r R --s2 S2 --i5 I5 --i7 I7
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
S2DIR = HERE.parent / "stage2"
ROOT = HERE.parents[3]
for p in (ROOT, S2DIR, HERE.parent):
    sys.path.insert(0, str(p))

import analyze as A  # noqa: E402  Stage 2's analysis module (helpers only; its main is not run)
from audio_shift import FIELD_CELLS, cond_name  # noqa: E402
from idea5b_common import mix_expected_f1  # noqa: E402

TASKS = {
    "esc50": dict(ds="esc50", stats="esc50", seeds=[0, 1, 2, 3, 4], K=50, short=512),
    "esc5": dict(ds="esc50", stats="esc50", seeds=[0, 1, 2, 3, 4], K=5, short=512),
    "urban": dict(ds="urban8k", stats="urban", seeds=[0, 1, 2], K=10, short=400),
    "esc5off": dict(ds="esc50", stats="esc50", seeds=[0, 1, 2, 3, 4], K=5, short=512),
}
KINDS = ["esc", "8", "4", "bitnet"]
AST_HEADS = ["esc", "4"]
CNNS = {"mn04": 384, "mn10": 960}
CHEAP_P2 = [("eff", "w8a8"), ("mn04", "w8a8"), ("mn10", "w8a8")]
FULL_TIERS = ["w4g128", "n4", "w4", "w5", "w6", "w8g64", "fp32"]
S2_SHORT = ["fp32", "w8g64", "w4g128", "n4", "w3g64", "w4"]
NEW_CFGS = ["n4", "w4g128"]
RULES = ["ast", "avg"]
BUDGETS = A.BUDGETS
FIELD = [cond_name(k, s) for k, s in FIELD_CELLS]
CHEAP_CONDS = A.CHEAP_CONDS
CASC_CONDS = ["clean", "field"]
PRICE = 1024.0


def ni_test(x, z, keys, ratio, margin=0.01):
    """One-sided non-inferiority of x against z within `margin` (F1 units), fold level: H0 mean d <= -margin."""
    d = np.asarray(x, float) - np.asarray(z, float)
    folds = np.array([k[1] for k in keys])
    fm = np.array([d[folds == f].mean() for f in np.unique(folds)])
    k = len(fm)
    sd = fm.std(ddof=1)
    if sd == 0:
        p = 0.0 if fm.mean() > -margin else 1.0
        return {"ni_diff": float(d.mean()), "ni_p": p, "ni_p_nb": p}
    t = (fm.mean() + margin) / (sd / np.sqrt(k))
    t_nb = (fm.mean() + margin) / np.sqrt((1.0 / k + ratio) * sd ** 2)
    return {"ni_diff": float(d.mean()), "ni_p": float(stats.t.sf(t, k - 1)), "ni_p_nb": float(stats.t.sf(t_nb, k - 1))}


def cheap_name(src, bb, k):
    return f"eff:{bb}:{k}" if src == "eff" else f"cnn:{src}:{bb}:{k}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--r", required=True, type=Path)
    ap.add_argument("--s2", required=True, type=Path)
    ap.add_argument("--i5", required=True, type=Path)
    ap.add_argument("--i7", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-mean2", action="store_true", help="leave the two-window mean out (plan's exception)")
    ap.add_argument("--check-only", action="store_true",
                    help="load, check rows and costs, run the first (seed, fold), write nothing and stop before any "
                         "comparison (code check while systems are still incomplete)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6"))
    t0 = time.time()
    task = a.task
    T = TASKS[task]
    K, seeds, SL = T["K"], T["seeds"], T["short"]
    out = a.out or a.r / "analysis" / task
    out.mkdir(parents=True, exist_ok=True)
    esc = T["ds"] == "esc50"

    # ── score sources ───────────────────────────────────────────────────────
    cheap_dirs = {}  # (src, seed, k) -> ScoreDir
    for s in seeds:
        for k in KINDS:
            cheap_dirs[("eff", s, k)] = A.ScoreDir(a.s2 / "scores" / task / f"effp_{task}_s{s}" / k)
            for m in CNNS:
                cheap_dirs[(m, s, k)] = A.ScoreDir(a.r / "scores_cnn" / task / f"{m}_{task}_s{s}" / k)

    def cheap_get(src, bb, s, k, cond):
        cfg = bb if src == "eff" else f"{src}_{bb}"
        return cheap_dirs[(src, s, k)].get(cfg, cond)

    def cheap_has(src, bb, cond):
        cfg = bb if src == "eff" else f"{src}_{bb}"
        return all(cheap_dirs[(src, s, k)].has(cfg, cond) for s in seeds for k in KINDS)

    full_root = a.s2 / "scores_ast1024" / task if task == "esc5off" else a.i5 / "scores" / task
    full_tag = "astoff1024_esc5off_s{}" if task == "esc5off" else f"ast{task}_s{{}}"
    groups = {  # group -> (dirs by (seed, head), configs, name suffix)
        "full": ({(s, h): A.ScoreDir(full_root / full_tag.format(s) / h) for s in seeds for h in AST_HEADS}, None, ""),
        "s2short": ({(s, h): A.ScoreDir(a.s2 / "scores_ast_short" / task / f"astshort_{task}_s{s}" / h)
                     for s in seeds for h in AST_HEADS}, S2_SHORT, f"@{SL}"),
        "a256": ({(s, h): A.ScoreDir(a.r / "scores_ast256" / task / f"ast256_{task}_s{s}" / h)
                  for s in seeds for h in AST_HEADS}, NEW_CFGS, "@256"),
        "a384": ({(s, h): A.ScoreDir(a.r / "scores_ast384" / task / f"ast384_{task}_s{s}" / h)
                  for s in seeds for h in AST_HEADS}, NEW_CFGS, "@384"),
    }
    if esc and not a.no_mean2:
        groups["m2"] = ({(s, h): A.ScoreDir(a.r / "scores_astm2" / task / f"astm2_{task}_s{s}" / h)
                         for s in seeds for h in AST_HEADS}, NEW_CFGS, "@256x2")
    src = {}  # AST system name -> (dirs, base config, group)
    for g, (dirs, cfgs, sfx) in groups.items():
        if cfgs is None:
            cfgs = sorted({f.split("__")[0] for f in dirs[(seeds[0], "esc")].files})
        for c in cfgs:
            src[c + sfx] = (dirs, c, g)
    new_ast = [n for n, v in src.items() if v[2] in ("a256", "a384", "m2")]

    def ast_has(name, cond):
        dirs, base, _ = src[name]
        return all(d.has(base, cond) for d in dirs.values())

    def ast_get(name, s, h, cond):
        dirs, base, _ = src[name]
        return dirs[(s, h)].get(base, cond)

    conds = CHEAP_CONDS + ["field"]
    cheap_all = [(x, bb) for x in ["eff", *CNNS] for bb in ("fp32", "w8a8")]
    cheap_p2 = list(CHEAP_P2)
    cheap_conds = {cb: [c for c in conds if cheap_has(*cb, c)] for cb in cheap_all}
    ast_conds = {n: [c for c in conds if ast_has(n, c)] for n in src}
    # completeness rules of the plan: every new system on clean and field (and every Stage 2 system used)
    missing = [n for n in new_ast if not all(c in ast_conds[n] for c in CASC_CONDS)]
    missing += [f"{x}:{bb}" for x, bb in cheap_all if not all(c in cheap_conds[(x, bb)] for c in CASC_CONDS)]
    if missing and not a.check_only:
        raise SystemExit(f"{task}: incomplete systems on clean or field: {missing}")
    if a.check_only:
        print(f"{task}: incomplete (allowed in --check-only): {missing}", flush=True)
    tiers = [n for n in FULL_TIERS if n in src] + [f"{c}{groups['s2short'][2]}" for c in S2_SHORT]
    tiers += [n for n in new_ast]
    if a.check_only:  # keep only what is complete so the loop can run
        tiers = [t for t in tiers if all(c in ast_conds[t] for c in CASC_CONDS)]
        cheap_all = [cb for cb in cheap_all if all(c in cheap_conds[cb] for c in CASC_CONDS)]
        cheap_p2 = [cb for cb in cheap_p2 if cb in cheap_all]
        src = {n: v for n, v in src.items() if "clean" in ast_conds[n]}
    if not all(all(c in ast_conds[t] for c in CASC_CONDS) for t in tiers):
        raise SystemExit(f"{task}: a tier lacks clean or field")
    print(f"{task}: {len(src)} AST systems ({len(new_ast)} new), tiers {tiers}", flush=True)

    # ── costs (priced) ──────────────────────────────────────────────────────
    from costs_idea6 import head_costs

    with contextlib.redirect_stdout(io.StringIO()):
        hk = "esc5" if task == "esc5off" else task
        hc = {d: head_costs(d, KINDS)[hk] for d in (1536, 768, *CNNS.values())}
    eff_emb = a.s2 / "emb" / T["ds"]
    lengths = np.load(eff_emb / "lengths__clean.npy")
    fbank = float(np.mean([A.ast_fbank_frontend_macs(int(n)) for n in lengths])) * 1024
    cheap_cost = {}
    for bb in ("fp32", "w8a8"):
        sd = json.loads((eff_emb / f"{bb}__clean.json").read_text())["cost_per_clip"]
        for k in KINDS:
            cheap_cost[("eff", bb, k)] = sd["frontend_bitops_mean"] + sd["backbone_bitops_rescale1024"] + hc[1536][k]["bitops"]
        for m, dim in CNNS.items():
            sc = json.loads((a.r / "emb_cnn" / T["ds"] / f"{m}_{bb}__clean.json").read_text())["cost_per_clip"]
            for k in KINDS:
                cheap_cost[(m, bb, k)] = sc["frontend_bitops_mean"] + sc["backbone_bitops_rescale1024"] + hc[dim][k]["bitops"]
    ast_cost, ast_side = {}, {}
    from idea5b_common import bitops as ast_bitops

    for n, (dirs, base, g) in src.items():
        if g == "full":
            bo, side_dir, mult, L = ast_bitops(base, PRICE), a.i5 / "emb" / T["ds"], 1, 1024
        else:
            L = {"s2short": SL, "a256": 256, "a384": 384, "m2": 256}[g]
            P_, T_ = A.short_tokens(L)
            mult = 2 if g == "m2" else 1
            bo = mult * A.ast_bitops_tok(base, PRICE, T_, P_)
            side_dir = {"s2short": a.s2 / "emb_ast_short" / T["ds"], "a256": a.r / "emb_ast" / f"{T['ds']}_256",
                        "a384": a.r / "emb_ast" / f"{T['ds']}_384", "m2": a.r / "emb_ast" / f"{T['ds']}_256x2"}[g]
        if g != "full":  # the unpriced count must equal the extraction sidecar
            s_ = json.loads((side_dir / f"{base}__clean.json").read_text())
            un = mult * A.ast_bitops_tok(base, 0.0, *A.short_tokens(L)[::-1])
            if abs(s_["backbone_cost_per_clip"]["bitops"] - un) > 1e-6 * un or s_["max_length"] != L:
                raise SystemExit(f"{n}: sidecar bit-ops {s_['backbone_cost_per_clip']['bitops']} != {un}")
            ast_side[n] = s_["names_crc32"]
        for h in AST_HEADS:
            ast_cost[(n, h)] = fbank + bo + hc[768][h]["bitops"]
    crc_ref = json.loads((eff_emb / "fp32__clean.json").read_text())["names_crc32"]
    for m in CNNS:
        for bb in ("fp32", "w8a8"):
            if json.loads((a.r / "emb_cnn" / T["ds"] / f"{m}_{bb}__clean.json").read_text())["names_crc32"] != crc_ref:
                raise SystemExit(f"{m}_{bb}: names checksum differs")
    if any(v != crc_ref for v in ast_side.values()):
        raise SystemExit(f"AST sidecar names checksum differs: {ast_side}")
    (out / "costs.json").write_text(json.dumps({
        "cheap": {cheap_name(*kk): v for kk, v in cheap_cost.items()},
        "ast": {f"{n}+{h}": v for (n, h), v in ast_cost.items()}, "fbank_bitops_mean": fbank,
        "heads": {str(d): v for d, v in hc.items()}}, indent=1))

    def sys_cost(name):
        if name.startswith("ast:"):
            n, h = name[4:].split("+")
            return ast_cost[(n, h)]
        parts = name.split(":")
        return cheap_cost[("eff", parts[1], parts[2]) if parts[0] == "eff" else (parts[1], parts[2], parts[3])]

    # ── one pass over seeds and folds ───────────────────────────────────────
    static, sel_rows, casc, esc_rows, au_rows = [], [], [], [], []
    preds, ys, yiv, iv_static, ivx = {}, {}, {}, {}, {}
    acs, escs, decs = {}, {}, {}
    au = {}
    designs = [(x, bb, k, e, r, b) for x, bb in cheap_p2 for k in KINDS for e in tiers for r in RULES
               for b in BUDGETS]
    thr_all = {}
    ref0 = cheap_get("eff", "fp32", seeds[0], "esc", "clean")
    folds = sorted(ref0["iv"])
    n_te = np.array([np.sum(ref0["fold"] == f) for f in folds])
    A.NB_RATIO["task"] = float(np.mean(n_te / (len(ref0["y"]) - n_te)))
    for s in seeds:
        ref = cheap_get("eff", "fp32", s, "esc", "clean")
        # rows, folds and inner splits equal in every file used
        chk = [(x, bb, k, c) for x, bb in cheap_all for k in KINDS for c in CASC_CONDS]
        for x, bb, k, c in chk:
            r_ = cheap_get(x, bb, s, k, c)
            if not ((r_["y"] == ref["y"]).all() and (r_["fold"] == ref["fold"]).all() and sorted(r_["iv"]) == folds
                    and all(np.array_equal(r_["iv"][f][0], ref["iv"][f][0]) for f in folds)):
                raise SystemExit(f"{x}:{bb}:{k} {c}: rows differ")
            if r_["crc"] is not None and r_["crc"] != crc_ref:
                raise SystemExit(f"{x}:{bb}:{k} {c}: names checksum differs")
        for n in src:
            for h in AST_HEADS:
                for c in [c for c in CASC_CONDS if c in ast_conds[n]]:
                    r_ = ast_get(n, s, h, c)
                    if not ((r_["y"] == ref["y"]).all() and (r_["fold"] == ref["fold"]).all()
                            and all(np.array_equal(r_["iv"][f][0], ref["iv"][f][0]) for f in folds)):
                        raise SystemExit(f"{n}+{h} {c}: rows differ")
        y_all = ref["y"]
        for fold in folds:
            te = np.flatnonzero(ref["fold"] == fold)
            iv = ref["iv"][fold][0]
            y, yv = y_all[te], y_all[iv]
            key = (s, fold)
            ys[key], yiv[key], ivx[key] = y, yv, iv
            # clean inner validation: every static, thresholds, selection rows
            civ = {}
            for x, bb in cheap_all:
                for k in KINDS:
                    p = A.softmax(cheap_get(x, bb, s, k, "clean")["iv"][fold][1])
                    civ[(x, bb, k)] = p
                    pr = p.argmax(1)
                    iv_static[(cheap_name(x, bb, k), s, fold)] = (A.f1_fast(yv, pr, K), pr)
            aiv = {}
            for n in [n for n in src if "clean" in ast_conds[n]]:
                for h in AST_HEADS:
                    p = A.softmax(ast_get(n, s, h, "clean")["iv"][fold][1])
                    aiv[(n, h)] = p
                    pr = p.argmax(1)
                    iv_static[(f"ast:{n}+{h}", s, fold)] = (A.f1_fast(yv, pr, K), pr)
            for x, bb in cheap_p2:
                for k in KINDS:
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
            # test rows
            for c in conds:
                cp = {}
                for x, bb in cheap_all:
                    if c not in cheap_conds[(x, bb)]:
                        continue
                    for k in KINDS:
                        z = cheap_get(x, bb, s, k, c)["z"][te]
                        p = A.softmax(z)
                        pr = p.argmax(1)
                        name = cheap_name(x, bb, k)
                        if c in CASC_CONDS:
                            preds[(name, c, s, fold)] = pr.astype(np.int16)
                        static.append({"system": name, "condition": c, "seed": s, "fold": fold,
                                       "f1": A.f1_fast(y, pr, K), "acc": float((pr == y).mean()),
                                       "cost": cheap_cost[(x, bb, k)]})
                        if (x, bb) in cheap_p2:
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
                for n in src:
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
            print(f"{task} seed {s} fold {fold}: {time.time() - t0:.0f}s", flush=True)
            if a.check_only:
                print(f"{task}: check-only pass ran one (seed, fold): {len(static)} static rows, {len(sel_rows)} "
                      f"selection rows, {len(casc)} cascade rows; nothing written", flush=True)
                return
        for d in list(cheap_dirs.values()) + [d for g in groups.values() for d in g[0].values()]:
            d.cache.clear()

    S = pd.DataFrame(static)
    S.to_csv(out / "static.csv", index=False)
    CA = pd.DataFrame(casc)
    CA.to_csv(out / "cascades.csv.gz", index=False)
    pd.DataFrame(esc_rows).to_csv(out / "escalation.csv", index=False)
    rows = []
    for (x, bb, k, c, s), a_ in au.items():
        sc, w = np.concatenate(a_["score"]), np.concatenate(a_["wrong"])
        rows.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "condition": c, "seed": s,
                     "auroc_wrong": roc_auc_score(w, sc) if 0 < w.sum() < len(w) else np.nan,
                     "error": float(w.mean())})
    pd.DataFrame(rows).to_csv(out / "auroc_wrong.csv", index=False)

    # ── P2 selection on clean inner validation ──────────────────────────────
    SR = pd.DataFrame(sel_rows)
    st = pd.DataFrame([{"system": nm, "seed": s, "fold": f, "f1": v[0], "cost": sys_cost(nm)}
                       for (nm, s, f), v in iv_static.items()])
    sp_iv = st.groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
    sp_iv.to_csv(out / "static_iv.csv", index=False)
    rng = np.random.default_rng(A.RNG_IV)
    sel_out = []
    for (x, bb, k, e, r, b), g in SR.groupby(["cheap_src", "cheap_bb", "cheap", "expensive", "rule", "budget"],
                                              sort=False):
        g = g.sort_values(["seed", "fold"])
        c_iv = float(np.mean(cheap_cost[(x, bb, k)] + g.iv_rate.to_numpy() * ast_cost[(e, "esc")]))
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
        d = g.f1_iv.to_numpy() - np.array(hull)
        sel_out.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r, "budget": b,
                        "f1_iv": g.f1_iv.mean(), "iv_rate": g.iv_rate.mean(), "cost_iv": c_iv, "hull_lo": lo,
                        "hull_hi": hi, "lam": lam, "hull_f1_iv": float(np.mean(hull)), "gain_iv": float(d.mean())})
    SO = pd.DataFrame(sel_out).sort_values(["gain_iv", "cost_iv"], ascending=[False, True])
    SO.to_csv(out / "selection.csv", index=False)
    top = SO.iloc[0]
    p2 = (top.cheap_src, top.cheap_bb, top.cheap, top.expensive, top.rule, top.budget)
    s2pick = json.loads((a.s2 / "analysis" / task / "picks.json").read_text())["H1"]["design"]
    p1 = ("eff", s2pick[0], s2pick[1], s2pick[2], s2pick[3], s2pick[4])
    picks = {"P1": {"design": list(p1), "source": "Stage 2 analysis/<task>/picks.json H1, unchanged"},
             "P2": {"design": list(p2), "gain_iv": float(top.gain_iv), "runner_up_gain_iv": float(SO.iloc[1].gain_iv),
                    "runner_up": [str(v) for v in SO.iloc[1][["cheap_src", "cheap_bb", "cheap", "expensive", "rule",
                                                              "budget"]]],
                    "cost_iv": float(top.cost_iv), "f1_iv": float(top.f1_iv), "iv_rate": float(top.iv_rate),
                    "hull_iv": [top.hull_lo, top.hull_hi, float(top.lam)]}}
    (out / "picks.json").write_text(json.dumps(picks, indent=1))
    print(f"{task}: P1 {p1}  P2 {p2} (gain_iv {top.gain_iv:.4f})", flush=True)

    # ── test comparisons ────────────────────────────────────────────────────
    from analyze import AudioStats, field_arrays, fit_lr, top_k, oracle_pred  # noqa: E402

    stems = [str(x) for x in np.load(a.i7 / "audio_stats" / T["stats"] / "clean.npz", allow_pickle=True)["names"]]
    audio = AudioStats(a.i7 / "audio_stats" / T["stats"], stems)
    test_idx = {f: np.flatnonzero(ref0["fold"] == f) for f in folds}
    noise = {}
    av = field_arrays(len(ref0["y"]))[1]
    for (s, f), iv in ivx.items():  # each seed's own inner split, as Stage 2
        noise[(s, f)] = fit_lr(audio.rows("field", iv, "val"), (av[iv] != "clean").astype(int))
    s_rec = {(n, c): dict(zip(zip(g.seed, g.fold), g.f1)) for (n, c), g in S.groupby(["system", "condition"])}
    CAg = {kk: g.sort_values(["seed", "fold"]) for kk, g in
           CA.groupby(["cheap_src", "cheap_bb", "cheap", "expensive", "rule", "budget", "condition"])}
    s2_static = {n for n in S.system.unique() if n.startswith("eff:") or
                 (n.startswith("ast:") and src[n[4:].split("+")[0]][2] in ("full", "s2short"))}
    sp_cache = {}

    def hull_for(c, cost, which):
        if (c, which) not in sp_cache:
            g = S[S.condition == c]
            if which == "stage2":
                g = g[g.system.isin(s2_static)]
            sp_cache[(c, which)] = g.groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
        sp = sp_cache[(c, which)]
        v_interp, lo, hi, lam = A.hull_pair(sp, cost)
        ok = sp[sp.cost <= cost * (1 + 1e-12)].sort_values(["f1", "cost"], ascending=[False, True])
        return v_interp, lo, hi, lam, (ok.iloc[0].system if len(ok) else None)

    def hull_f1(c, lo, hi, lam, kk):
        if lam == 0.0:
            return s_rec[(lo, c)][kk]
        return float(mix_expected_f1(ys[kk], preds[(lo, c, *kk)].astype(np.int64),
                                     preds[(hi, c, *kk)].astype(np.int64), lam, K)[0])

    ni_ref = f"ast:n4@{SL}+esc"

    def compare(design, c, which="new", baselines=False):
        x, bb, k, e, r, b = design
        g = CAg[(x, bb, k, e, r, b, c)]
        keys = list(zip(g.seed, g.fold))
        xv = g.f1.to_numpy()
        cost = float(g.cost.mean())
        v_interp, lo, hi, lam, best = hull_for(c, cost, which)
        hull = np.array([hull_f1(c, lo, hi, lam, kk) for kk in keys])
        row = {"hull_set": which, "cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r, "budget": b,
               "condition": c, "f1": float(xv.mean()), "cost": cost, "rate": float(g.rate.mean()),
               "rate_min": float(g.rate.min()), "rate_max": float(g.rate.max()), "ece": float(g.ece.mean()),
               "hull_lo": lo, "hull_hi": hi, "lam": lam, "hull_f1_interp": v_interp, "hull_f1": float(hull.mean())}
        row.update({f"hull_{kk}": vv for kk, vv in A.tests(xv, hull, keys).items()})
        if not baselines:
            return row
        z = np.array([s_rec[(ni_ref, c)][kk] for kk in keys])
        row.update({"ni_ref": ni_ref, "ni_ref_f1": float(z.mean()), "ni_ref_cost": sys_cost(ni_ref)})
        row.update(ni_test(xv, z, keys, A.NB_RATIO["task"]))
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
        if aud:
            row["audio_f1"] = float(np.mean(aud))
            row.update({f"audio_{kk}": vv for kk, vv in A.tests(xv, aud, keys).items()})
        return row

    res = []
    for fam, d in (("P1", p1), ("P2", p2)):
        for c in CASC_CONDS:
            for which in (("new", "stage2") if fam == "P1" else ("new",)):  # E4 (Stage 2's hull) is for P1 only
                row = compare(d, c, which, baselines=(which == "new"))
                row["family"] = fam
                res.append(row)
    R_ = pd.DataFrame(res)
    R_.to_csv(out / "selected.csv", index=False)
    # Stage 2 reproduction: P1's cascade rows equal Stage 2's stored ones
    s2c = pd.read_csv(a.s2 / "analysis" / task / "cascades.csv.gz")
    s2c = s2c[(s2c.cheap_bb == p1[1]) & (s2c.cheap == p1[2]) & (s2c.expensive == p1[3]) & (s2c.rule == p1[4])
              & (s2c.budget == p1[5]) & (s2c.condition.isin(CASC_CONDS))].sort_values(["condition", "seed", "fold"])
    mine = CA[(CA.cheap_src == "eff") & (CA.cheap_bb == p1[1]) & (CA.cheap == p1[2]) & (CA.expensive == p1[3])
              & (CA.rule == p1[4]) & (CA.budget == p1[5])].sort_values(["condition", "seed", "fold"])
    repro = {"rows": len(mine), "f1_maxabs": float(np.abs(mine.f1.to_numpy() - s2c.f1.to_numpy()).max()),
             "rate_maxabs": float(np.abs(mine.rate.to_numpy() - s2c.rate.to_numpy()).max()),
             "cost_maxrel": float(np.abs(mine.cost.to_numpy() / s2c.cost_priced.to_numpy() - 1).max())}
    print(f"{task}: P1 reproduction against Stage 2 {repro}", flush=True)
    if repro["f1_maxabs"] > 1e-9 or repro["rate_maxabs"] > 1e-12 or repro["cost_maxrel"] > 1e-9:
        raise SystemExit(f"{task}: P1 cascade does not reproduce Stage 2: {repro}")

    e2 = [compare(d, "field") for d in designs]
    pd.DataFrame(e2).to_csv(out / "e2.csv", index=False)
    print(f"{task}: E2 done, {time.time() - t0:.0f}s", flush=True)

    # ── S2: "cheap wrong, AST tier right" AUROC where the tier exists ───────
    rows = []
    for fam, d in (("P1", p1), ("P2", p2)):
        x, bb, k, e, r, b = d
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
                rows.append({"family": fam, "condition": c, "seed": s, "n_fixable": int(fx_.sum()),
                             "auroc_fixable": roc_auc_score(fx_, sc_) if 0 < fx_.sum() < len(fx_) else np.nan})
    pd.DataFrame(rows).to_csv(out / "auroc_fixable.csv", index=False)
    meta = {"task": task, "K": K, "seeds": seeds, "folds": folds, "nb_ratio": A.NB_RATIO["task"],
            "tiers": tiers, "n_designs": len(designs), "new_ast": new_ast, "ast_conditions": ast_conds,
            "cheap_conditions": {f"{x}:{bb}": v for (x, bb), v in cheap_conds.items()}, "picks": picks,
            "p1_reproduction": repro, "mean2": "m2" in groups, "seconds": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{task}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
