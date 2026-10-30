"""Idea 6 option B analysis on TAU 2020 Mobile, as fixed in scripts/redesign/idea6/optionB/ANALYSIS_PLAN.md
(commit 4468dad): Stage 2's helpers (scripts/redesign/idea6/stage2/analyze.py) on the universe score files of
score_optb.py. Conditions per fold: clean (device-A rows), shift (every other device's rows of the fold's
locations, pooled) and dev:<d> (one device). Thresholds, selection and the inner-validation hull use the clean
device-A inner-validation rows only.

Inputs: --r/scores/<tag>/<kind>/<config>__all.npz (tags mn10_tau_s<seed>, mn20_tau_s<seed>, ast_tau_s<seed>),
--r/emb_cnn/tau and --r/emb_ast/tau sidecars (costs), --r/audio_stats/tau/clean.npz (idea 7's statistics, the
shift-detector gate), --meta (the universe CSV: folds, devices, locations).

Outputs in --out: costs.json, static.csv (every static on every condition), static_iv.csv, selection.csv,
picks.json, cascades.csv.gz (every design on every condition), selected.csv (the pick on every condition against
the hull; baselines on clean and shift), hypotheses.csv (H1, H2 with Holm), e2.csv (every design on clean and
shift), escalation.csv, auroc.csv, headroom.csv (S1), meta.json.

    python scripts/redesign/idea6/optionB/analyze_optb.py --r R --meta ~/Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv
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
ROOT = HERE.parents[3]
for p in (ROOT, HERE.parent / "stage2", HERE.parent / "stage3b", HERE.parent):
    sys.path.insert(0, str(p))

import analyze as A  # noqa: E402  Stage 2's helpers
from analyze_3b import ni_test  # noqa: E402
from idea5b_common import bitops as ast_bitops, holm, mix_expected_f1  # noqa: E402

KINDS = ["esc", "8", "4", "bitnet"]
AST_HEADS = ["esc", "4"]
CNNS = {"mn10": 960, "mn20": 1920}
BACKBONES = ["fp32", "w8a8"]
AST_CFGS = ["fp32", "w8g64", "w4g128", "n4"]
RULES = ["ast", "avg"]
BUDGETS = A.BUDGETS
SEEDS = [0, 1, 2]
K = 10
PRICE = 1024.0
DEVICES = ["b", "c", "s1", "s2", "s3", "s4", "s5", "s6"]
CONDS = ["clean", "shift"] + [f"dev:{d}" for d in DEVICES]
CASC_CONDS = ["clean", "shift"]
CHEAP_CASC = [(m, "w8a8") for m in CNNS]
N_SAMPLES_16K = 160000


class Src:
    """Universe score files of one run (seed, kind)."""

    def __init__(self, d: Path):
        self.d, self.cache = d, {}

    def get(self, cfg):
        if cfg not in self.cache:
            z = np.load(self.d / f"{cfg}__all.npz")
            iv = {int(k[7:]): z[k] for k in z.files if k.startswith("iv_idx_")}
            self.cache[cfg] = {"y": z["y"].astype(np.int64), "fold": z["fold"].astype(int), "device": z["device"].astype(str),
                               "z": z["test_logits"].astype(np.float64), "n_a": int(z["n_a"]), "crc": int(z["names_crc32"]),
                               "iv": {f: (iv[f], z[f"iv_logits_{f}"].astype(np.float64)) for f in sorted(iv)}}
        return self.cache[cfg]


def ci95(fm):
    k = len(fm)
    if k < 2 or fm.std(ddof=1) == 0:
        return float(fm.mean()), float(fm.mean())
    h = stats.t.ppf(0.975, k - 1) * fm.std(ddof=1) / np.sqrt(k)
    return float(fm.mean() - h), float(fm.mean() + h)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--r", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--partial", action="store_true", help="code check: drop AST configurations without scores")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = a.out or a.r / "analysis" / "tau"
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(a.meta)
    loc_u = meta.location.to_numpy()

    cnn = {(m, s, k): Src(a.r / "scores" / f"{m}_tau_s{s}" / k) for m in CNNS for s in SEEDS for k in KINDS}
    ast = {(s, h): Src(a.r / "scores" / f"ast_tau_s{s}" / h) for s in SEEDS for h in AST_HEADS}
    ast_cfgs = list(AST_CFGS)
    if a.partial:
        ast_cfgs = [c for c in AST_CFGS if all((d.d / f"{c}__all.npz").exists() for d in ast.values())]
        print(f"partial: AST configurations {ast_cfgs}", flush=True)
    for d in cnn.values():
        for bb in BACKBONES:
            if not (d.d / f"{d.d.parent.name.split('_')[0]}_{bb}__all.npz").exists():
                raise SystemExit(f"{d.d}: missing {bb}")
    for d in ast.values():
        for c in ast_cfgs:
            if not (d.d / f"{c}__all.npz").exists():
                raise SystemExit(f"{d.d}: missing {c}")

    # ── costs (priced bit-ops) ─────────────────────────────────────────────
    from costs_idea6 import head_costs

    with contextlib.redirect_stdout(io.StringIO()):
        hc = {d: head_costs(d, KINDS)["urban"] for d in sorted({768, *CNNS.values()})}
    cheap_cost, side_crc = {}, {}
    for m, dim in CNNS.items():
        for bb in BACKBONES:
            sd = json.loads((a.r / "emb_cnn" / "tau" / f"{m}_{bb}__clean.json").read_text())
            side_crc[f"{m}_{bb}"] = sd["names_crc32"]
            cc = sd["cost_per_clip"]
            for k in KINDS:
                cheap_cost[(m, bb, k)] = cc["frontend_bitops_mean"] + cc["backbone_bitops_rescale1024"] + hc[dim][k]["bitops"]
    lengths = np.load(a.r / "emb_cnn" / "tau" / "lengths__clean.npy")
    if lengths.min() != N_SAMPLES_16K or lengths.max() != N_SAMPLES_16K:
        raise SystemExit(f"clip lengths {lengths.min()}..{lengths.max()} are not all {N_SAMPLES_16K}")
    fbank = A.ast_fbank_frontend_macs(N_SAMPLES_16K) * 1024
    ast_cost = {}
    for c in ast_cfgs:
        sd = json.loads((a.r / "emb_ast" / "tau" / f"{c}__clean.json").read_text())
        side_crc[f"ast_{c}"] = sd["names_crc32"]
        sc = sd["backbone_cost_per_clip"]["bitops"]
        if sd["max_length"] != 1024 or abs(sc - ast_bitops(c, 0.0)) > 1e-6 * sc:
            raise SystemExit(f"{c}: sidecar bit-ops {sc} != {ast_bitops(c, 0.0)} or max_length {sd['max_length']}")
        for h in AST_HEADS:
            ast_cost[(c, h)] = fbank + ast_bitops(c, PRICE) + hc[768][h]["bitops"]
    if len(set(side_crc.values())) != 1:
        raise SystemExit(f"names checksums differ across sidecars: {side_crc}")
    crc_ref = next(iter(side_crc.values()))
    (out / "costs.json").write_text(json.dumps({
        "cheap": {f"cnn:{m}:{bb}:{k}": v for (m, bb, k), v in cheap_cost.items()},
        "ast": {f"ast:{c}+{h}": v for (c, h), v in ast_cost.items()}, "fbank_bitops": fbank,
        "heads": {str(d): v for d, v in hc.items()}, "price": PRICE}, indent=1))

    def sys_cost(name):
        if name.startswith("ast:"):
            c, h = name[4:].split("+")
            return ast_cost[(c, h)]
        _, m, bb, k = name.split(":")
        return cheap_cost[(m, bb, k)]

    # ── audio statistics (idea 7) for the shift-detector gate ──────────────
    st = np.load(a.r / "audio_stats" / "tau" / "clean.npz", allow_pickle=True)
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    if [str(s) for s in st["names"]] != stems:
        raise SystemExit("audio statistics rows differ from the universe CSV")
    STATS = st["stats"].astype(np.float64)
    if not np.isfinite(STATS).all():
        raise SystemExit("audio statistics have NaN rows")

    # ── one pass over seeds and folds ──────────────────────────────────────
    static, sel_rows, casc, esc_rows = [], [], [], []
    preds, ys, yiv, iv_static, rows_of = {}, {}, {}, {}, {}
    acs, escs, decs, au, thr_all, det = {}, {}, {}, {}, {}, {}
    ref0 = cnn[("mn10", SEEDS[0], "esc")].get("mn10_fp32")
    if ref0["crc"] != crc_ref:
        raise SystemExit("score names checksum differs from the sidecars")
    folds = sorted(ref0["iv"])
    y_u, fold_u, dev_u, n_a = ref0["y"], ref0["fold"], ref0["device"], ref0["n_a"]
    if not np.array_equal(y_u, meta.class_id.to_numpy()) or not np.array_equal(fold_u, meta.folder.to_numpy()):
        raise SystemExit("score rows differ from the universe CSV")
    n_train = {f: int(np.sum((fold_u[:n_a] != f))) for f in folds}
    masks = {}
    for f in folds:
        rf = np.flatnonzero(fold_u == f)
        d = dev_u[rf]
        masks[f] = {"clean": rf[d == "a"], "shift": rf[d != "a"], **{f"dev:{x}": rf[d == x] for x in DEVICES}}
    nb_ratio = {c: float(np.mean([len(masks[f][c]) / n_train[f] for f in folds])) for c in CONDS}
    designs = [(m, k, e, r, b) for m in CNNS for k in KINDS for e in ast_cfgs for r in RULES for b in BUDGETS]

    for s in SEEDS:
        ref = cnn[("mn10", s, "esc")].get("mn10_fp32")
        srcs = [(cnn[(m, s, k)], f"{m}_{bb}") for m in CNNS for k in KINDS for bb in BACKBONES]
        srcs += [(ast[(s, h)], c) for h in AST_HEADS for c in ast_cfgs]
        for d, cfg in srcs:
            r_ = d.get(cfg)
            if not ((r_["y"] == ref["y"]).all() and (r_["fold"] == ref["fold"]).all() and (r_["device"] == ref["device"]).all()
                    and sorted(r_["iv"]) == folds and all(np.array_equal(r_["iv"][f][0], ref["iv"][f][0]) for f in folds)
                    and r_["crc"] == crc_ref):
                raise SystemExit(f"{d.d}/{cfg}: rows, folds, devices or inner splits differ")
        for f in folds:
            iv = ref["iv"][f][0]
            yv = y_u[iv]
            key = (s, f)
            yiv[key] = yv
            # the shift detector: inner-validation device-A rows and the other devices' rows of those locations
            iv_locs = set(loc_u[iv])
            dev_rows = np.flatnonzero((fold_u != f) & (dev_u != "a") & np.isin(loc_u, list(iv_locs)))
            tr = np.concatenate([iv, dev_rows])
            det[key] = A.fit_lr(STATS[tr], (dev_u[tr] != "a").astype(int))
            # clean inner validation: statics, thresholds, selection rows
            civ = {(m, bb, k): A.softmax(cnn[(m, s, k)].get(f"{m}_{bb}")["iv"][f][1]) for m in CNNS for bb in BACKBONES for k in KINDS}
            aiv = {(c, h): A.softmax(ast[(s, h)].get(c)["iv"][f][1]) for c in ast_cfgs for h in AST_HEADS}
            for (m, bb, k), p in civ.items():
                pr = p.argmax(1)
                iv_static[(f"cnn:{m}:{bb}:{k}", s, f)] = (A.f1_fast(yv, pr, K), pr)
            for (c, h), p in aiv.items():
                pr = p.argmax(1)
                iv_static[(f"ast:{c}+{h}", s, f)] = (A.f1_fast(yv, pr, K), pr)
            for m, bb in CHEAP_CASC:
                for k in KINDS:
                    p = civ[(m, bb, k)]
                    mg, ac = A.margin(p), p.argmax(1)
                    for b, rate in BUDGETS.items():
                        t_ = A.thr_rate(mg, rate)
                        thr_all[(m, k, b, s, f)] = t_
                        escm = mg < t_
                        for e in ast_cfgs:
                            pe = aiv[(e, "esc")]
                            for r in RULES:
                                de = pe.argmax(1) if r == "ast" else ((p + pe) / 2).argmax(1)
                                sel_rows.append({"cheap_bb": m, "cheap": k, "expensive": e, "rule": r, "budget": b,
                                                 "seed": s, "fold": f, "threshold": float(t_), "iv_rate": float(escm.mean()),
                                                 "f1_iv": A.f1_fast(yv, np.where(escm, de, ac), K)})
            for c in CONDS:
                rows = masks[f][c]
                y = y_u[rows]
                ys[(s, f, c)], rows_of[(f, c)] = y, rows
                cp, ap_ = {}, {}
                for m in CNNS:
                    for bb in BACKBONES:
                        for k in KINDS:
                            p = A.softmax(cnn[(m, s, k)].get(f"{m}_{bb}")["z"][rows])
                            pr = p.argmax(1)
                            name = f"cnn:{m}:{bb}:{k}"
                            preds[(name, c, s, f)] = pr.astype(np.int16)
                            static.append({"system": name, "condition": c, "seed": s, "fold": f, "n": len(rows),
                                           "f1": A.f1_fast(y, pr, K), "acc": float((pr == y).mean()), "cost": cheap_cost[(m, bb, k)]})
                            if (m, bb) in CHEAP_CASC:
                                cp[(m, k)] = p
                                mg = A.margin(p)
                                for b in BUDGETS:
                                    esc_rows.append({"cheap_bb": m, "cheap": k, "budget": b, "condition": c, "seed": s, "fold": f,
                                                     "rate": float((mg < thr_all[(m, k, b, s, f)]).mean()), "error": float((pr != y).mean())})
                for cfg in ast_cfgs:
                    for h in AST_HEADS:
                        p = A.softmax(ast[(s, h)].get(cfg)["z"][rows])
                        pr = p.argmax(1)
                        name = f"ast:{cfg}+{h}"
                        preds[(name, c, s, f)] = pr.astype(np.int16)
                        static.append({"system": name, "condition": c, "seed": s, "fold": f, "n": len(rows),
                                       "f1": A.f1_fast(y, pr, K), "acc": float((pr == y).mean()), "cost": ast_cost[(cfg, h)]})
                        if h == "esc":
                            ap_[cfg] = p
                for (m, k), p in cp.items():
                    mg, ac = A.margin(p), p.argmax(1)
                    acs[(m, k, c, s, f)] = ac.astype(np.int16)
                    qs = {}
                    for e in ast_cfgs:
                        pe = ap_[e]
                        for r in RULES:
                            q = pe if r == "ast" else (p + pe) / 2
                            qs[(e, r)] = (q.argmax(1), q.max(1))
                            decs[(m, k, e, r, c, s, f)] = qs[(e, r)][0].astype(np.int16)
                        a_ = au.setdefault((m, k, e, c, s), {"score": [], "wrong": [], "fixable": []})
                        a_["score"].append(1 - mg)
                        a_["wrong"].append(ac != y)
                        a_["fixable"].append((ac != y) & (pe.argmax(1) == y))
                    for b in BUDGETS:
                        escm = mg < thr_all[(m, k, b, s, f)]
                        escs[(m, k, b, c, s, f)] = escm
                        rate = float(escm.mean())
                        for (e, r), (de, qmax) in qs.items():
                            pred = np.where(escm, de, ac)
                            casc.append({"cheap_bb": m, "cheap": k, "expensive": e, "rule": r, "budget": b, "condition": c,
                                         "seed": s, "fold": f, "f1": A.f1_fast(y, pred, K), "acc": float((pred == y).mean()),
                                         "rate": rate, "ece": A.ece_conf(np.where(escm, qmax, p.max(1)), pred == y),
                                         "cost": cheap_cost[(m, "w8a8", k)] + rate * ast_cost[(e, "esc")]})
            print(f"seed {s} fold {f}: {time.time() - t0:.0f}s", flush=True)
        for d in list(cnn.values()) + list(ast.values()):
            d.cache.clear()

    S = pd.DataFrame(static)
    S.to_csv(out / "static.csv", index=False)
    CA = pd.DataFrame(casc)
    CA.to_csv(out / "cascades.csv.gz", index=False)
    pd.DataFrame(esc_rows).to_csv(out / "escalation.csv", index=False)
    rows = []
    for (m, k, e, c, s), a_ in au.items():
        sc, w, fx = (np.concatenate(a_[x]) for x in ("score", "wrong", "fixable"))
        rows.append({"cheap_bb": m, "cheap": k, "expensive": e, "condition": c, "seed": s, "error": float(w.mean()),
                     "auroc_wrong": roc_auc_score(w, sc) if 0 < w.sum() < len(w) else np.nan,
                     "auroc_fixable": roc_auc_score(fx, sc) if 0 < fx.sum() < len(fx) else np.nan, "n_fixable": int(fx.sum())})
    pd.DataFrame(rows).to_csv(out / "auroc.csv", index=False)

    # ── S1 headroom: best cheap W8A8 static against the best AST static, clean and shift ──
    smean = S.groupby(["system", "condition"]).f1.mean()
    srec = {(n, c): dict(zip(zip(g.seed, g.fold), g.f1)) for (n, c), g in S.groupby(["system", "condition"])}
    keys = [(s, f) for s in SEEDS for f in folds]
    hr = []
    for c in CONDS:
        sm = smean.xs(c, level="condition")
        best_w8 = sm[[n for n in sm.index if ":w8a8:" in n]].idxmax()
        best_cnn = sm[[n for n in sm.index if n.startswith("cnn:")]].idxmax()
        best_ast = sm[[n for n in sm.index if n.startswith("ast:")]].idxmax()
        for lab, lo in (("w8a8", best_w8), ("any_cnn", best_cnn)):
            x = np.array([srec[(best_ast, c)][kk] for kk in keys])
            z = np.array([srec[(lo, c)][kk] for kk in keys])
            t_ = A.tests(x, z, keys, nb_ratio[c])
            fm = np.array(json.loads(t_["fold_diffs"]))
            lo_ci, hi_ci = ci95(fm)
            hr.append({"condition": c, "cheap_kind": lab, "cheap": lo, "cheap_f1": float(z.mean()), "cheap_cost": sys_cost(lo),
                       "ast": best_ast, "ast_f1": float(x.mean()), "ast_cost": sys_cost(best_ast), "headroom": t_["diff"],
                       "ci_lo": lo_ci, "ci_hi": hi_ci, "p_t_fold": t_["p_t_fold"], "fold_diffs": t_["fold_diffs"]})
    pd.DataFrame(hr).to_csv(out / "headroom.csv", index=False)

    # ── selection on clean inner validation ───────────────────────────────
    SR = pd.DataFrame(sel_rows)
    st_iv = pd.DataFrame([{"system": nm, "seed": s, "fold": f, "f1": v[0], "cost": sys_cost(nm)} for (nm, s, f), v in iv_static.items()])
    sp_iv = st_iv.groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
    sp_iv.to_csv(out / "static_iv.csv", index=False)
    rng = np.random.default_rng(A.RNG_IV)
    sel_out = []
    for (m, k, e, r, b), g in SR.groupby(["cheap_bb", "cheap", "expensive", "rule", "budget"], sort=False):
        g = g.sort_values(["seed", "fold"])
        c_iv = float(np.mean(cheap_cost[(m, "w8a8", k)] + g.iv_rate.to_numpy() * ast_cost[(e, "esc")]))
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
        sel_out.append({"cheap_bb": m, "cheap": k, "expensive": e, "rule": r, "budget": b, "f1_iv": g.f1_iv.mean(),
                        "iv_rate": g.iv_rate.mean(), "cost_iv": c_iv, "hull_lo": lo, "hull_hi": hi, "lam": lam,
                        "hull_f1_iv": float(np.mean(hull)), "gain_iv": float(d.mean()), "gain_iv_sd_pairs": float(d.std(ddof=1))})
    SO = pd.DataFrame(sel_out).sort_values(["gain_iv", "cost_iv"], ascending=[False, True]).reset_index(drop=True)
    SO.to_csv(out / "selection.csv", index=False)
    top, second = SO.iloc[0], SO.iloc[1]
    pick = (top.cheap_bb, top.cheap, top.expensive, top.rule, top.budget)
    picks = {"pick": {"design": list(pick), "gain_iv": float(top.gain_iv), "cost_iv": float(top.cost_iv), "f1_iv": float(top.f1_iv),
                      "iv_rate": float(top.iv_rate), "hull_iv": [top.hull_lo, top.hull_hi, float(top.lam)]},
             "runner_up": {"design": [second.cheap_bb, second.cheap, second.expensive, second.rule, second.budget],
                           "gain_iv": float(second.gain_iv), "cost_iv": float(second.cost_iv)},
             "n_designs": int(len(SO)), "n_positive_gain": int((SO.gain_iv > 0).sum())}
    (out / "picks.json").write_text(json.dumps(picks, indent=1))
    print(f"pick {pick} gain_iv {top.gain_iv:.4f} (runner-up {second.gain_iv:.4f}), {time.time() - t0:.0f}s", flush=True)

    # ── test comparisons ──────────────────────────────────────────────────
    CAg = {key: g.sort_values(["seed", "fold"]) for key, g in CA.groupby(["cheap_bb", "cheap", "expensive", "rule", "budget", "condition"])}
    sp_cache = {}

    def hull_for(c, cost):
        if c not in sp_cache:
            sp_cache[c] = S[S.condition == c].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
        sp = sp_cache[c]
        v_interp, lo, hi, lam = A.hull_pair(sp, cost)
        ok = sp[sp.cost <= cost * (1 + 1e-12)].sort_values(["f1", "cost"], ascending=[False, True]).iloc[0]
        return v_interp, lo, hi, lam, ok.system

    def hull_f1(c, lo, hi, lam, kk):
        if lam == 0.0:
            return srec[(lo, c)][kk]
        return float(mix_expected_f1(ys[(*kk, c)], preds[(lo, c, *kk)].astype(np.int64), preds[(hi, c, *kk)].astype(np.int64), lam, K)[0])

    def compare(design, c, with_baselines):
        m, k, e, r, b = design
        g = CAg[(m, k, e, r, b, c)]
        kk_ = list(zip(g.seed, g.fold))
        x = g.f1.to_numpy()
        cost = float(g.cost.mean())
        v_interp, lo, hi, lam, best_static = hull_for(c, cost)
        hull = np.array([hull_f1(c, lo, hi, lam, kk) for kk in kk_])
        row = {"cheap_bb": m, "cheap": k, "expensive": e, "rule": r, "budget": b, "condition": c, "f1": float(x.mean()),
               "cost": cost, "rate": float(g.rate.mean()), "rate_min": float(g.rate.min()), "rate_max": float(g.rate.max()),
               "ece": float(g.ece.mean()), "hull_lo": lo, "hull_hi": hi, "lam": lam, "hull_f1_interp": v_interp,
               "hull_f1": float(hull.mean())}
        row.update({f"hull_{kk}": vv for kk, vv in A.tests(x, hull, kk_, nb_ratio[c]).items()})
        if not with_baselines:
            return row
        rnd, orc, orc_full, aud = [], [], [], []
        for kk in kk_:
            pc = acs[(m, k, c, *kk)].astype(np.int64)
            de = decs[(m, k, e, r, c, *kk)].astype(np.int64)
            escm = escs[(m, k, b, c, *kk)]
            yk = ys[(*kk, c)]
            rnd.append(float(mix_expected_f1(yk, pc, de, escm.mean(), K)[0]))
            orc.append(A.f1_fast(yk, A.oracle_pred(yk, pc, de, int(escm.sum())), K))
            fix = (pc != yk) & (de == yk)
            orc_full.append(A.f1_fast(yk, np.where(fix, de, pc), K))
            sc = det[kk](STATS[rows_of[(kk[1], c)]])
            aud.append(A.f1_fast(yk, np.where(A.top_k(sc, int(escm.sum())), de, pc), K))
        row["random_f1"] = float(np.mean(rnd))
        row.update({f"random_{kk}": vv for kk, vv in A.tests(x, rnd, kk_, nb_ratio[c]).items()})
        row["oracle_k_f1"], row["oracle_full_f1"] = float(np.mean(orc)), float(np.mean(orc_full))
        row["best_static"] = best_static
        z = np.array([srec[(best_static, c)][kk] for kk in kk_])
        row["best_static_f1"], row["best_static_cost"] = float(z.mean()), sys_cost(best_static)
        row.update({f"static_{kk}": vv for kk, vv in A.tests(x, z, kk_, nb_ratio[c]).items()})
        row["detector_f1"] = float(np.mean(aud))
        row.update({f"detector_{kk}": vv for kk, vv in A.tests(x, aud, kk_, nb_ratio[c]).items()})
        # non-inferiority within 1 point against the best AST static (descriptive)
        best_ast = smean.xs(c, level="condition")
        best_ast = best_ast[[n for n in best_ast.index if n.startswith("ast:")]].idxmax()
        za = np.array([srec[(best_ast, c)][kk] for kk in kk_])
        row["ni_ast"], row["ni_ast_f1"], row["ni_ast_cost"] = best_ast, float(za.mean()), sys_cost(best_ast)
        row.update(ni_test(x, za, kk_, nb_ratio[c]))
        return row

    sel = [compare(pick, c, c in CASC_CONDS) for c in CONDS]
    SEL = pd.DataFrame(sel)
    SEL.to_csv(out / "selected.csv", index=False)
    hyp = []
    for h_, c in (("H1", "clean"), ("H2", "shift")):
        r_ = SEL[SEL.condition == c].iloc[0]
        hyp.append({"hypothesis": h_, "condition": c, "design": " ".join(pick), "f1": r_.f1, "cost": r_.cost, "rate": r_.rate,
                    "hull_pair": f"{r_.hull_lo} / {r_.hull_hi} ({r_.lam:.2f})", "hull_f1": r_.hull_f1, "diff": r_.hull_diff,
                    "fold_diffs": r_.hull_fold_diffs, "p_t_fold": r_.hull_p_t_fold, "p_nb": r_.hull_p_nb, "p_signflip": r_.hull_p_signflip})
    H = pd.DataFrame(hyp)
    H["p_t_fold_holm"] = holm(H.p_t_fold.to_numpy())
    H["p_nb_holm"] = holm(H.p_nb.to_numpy())
    H["verdict"] = np.where((H["diff"] > 0) & (H.p_t_fold_holm < 0.05), "PASS", "FAIL")
    H.to_csv(out / "hypotheses.csv", index=False)
    print(H[["hypothesis", "f1", "hull_f1", "diff", "p_t_fold", "p_t_fold_holm", "verdict"]].to_string(), flush=True)

    e2 = [compare(d, c, False) for c in CASC_CONDS for d in designs if d + (c,) in CAg]
    E2 = pd.DataFrame(e2)
    E2["p_holm_all"] = holm(E2.hull_p_t_fold.to_numpy())
    E2.to_csv(out / "e2.csv", index=False)
    # the shift detector's own AUROC for "not device A" on the shift plus clean rows, per (seed, fold)
    drows = []
    for (s, f), fn in det.items():
        rows = np.concatenate([masks[f]["clean"], masks[f]["shift"]])
        drows.append({"seed": s, "fold": f, "auroc_not_a": roc_auc_score(dev_u[rows] != "a", fn(STATS[rows]))})
    pd.DataFrame(drows).to_csv(out / "detector_auroc.csv", index=False)
    meta_out = {"K": K, "seeds": SEEDS, "folds": folds, "conditions": CONDS, "ast_configs": ast_cfgs, "n_designs": len(designs),
                "nb_ratio": nb_ratio, "n_rows": {c: {int(f): int(len(masks[f][c])) for f in folds} for c in CONDS},
                "n_train": n_train, "draws_iv": A.DRAWS, "rng_iv": A.RNG_IV, "picks": picks, "partial": a.partial,
                "seconds": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta_out, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
