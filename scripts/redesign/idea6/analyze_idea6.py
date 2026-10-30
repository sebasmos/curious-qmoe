"""Idea 6 stage 1, one task: a cheap EfficientNet-B3 head whose softmax margin
decides which clips also run a quantized AST, against every static system and
the time-sharing hull of the static front. Clean audio, protocol 3 folds.

Inputs (all per-clip logits, test fold and inner validation, same rows):
  cheap tier   --eff-scores/<tag>/<kind>.npz (score_eff.py), kinds esc (FP32),
               8, 4, bitnet; tags effesc50_s<seed>, esc_singles_s<seed>,
               urban_singles_s<seed>
  AST tier     --ast-scores/<tag>/<head>/<config>__clean.npz (idea 5 Stage 2,
               score_stage2.py), heads esc (FP32) and 4; the expensive tiers
               use the FP32 head on w8g64, w4g128 and fp32
  costs        --costs (costs_idea6.py)
Rows are checked equal before anything is computed: labels, test folds and
every fold's inner validation indices across all files, and the cheap CSV's
file-name checksum against the AST sidecars'.

Static systems: every cheap head (EfficientNet backbone + head) and every
clean AST configuration with either head. Cascades: cheap kind x expensive
config x decision rule x operating point. A clip escalates when the cheap
head's top-1 minus top-2 softmax probability is below a threshold set on the
fold's inner validation split only:
  b05 .. b50  inner-validation escalation rate closest to 5, 10, 20, 30, 50%
              (escalation counts reachable without splitting ties; ties go to
              fewer escalations)
  match       the smallest inner-validation escalation whose cascade F1
              reaches the expensive tier's inner-validation F1 (the best F1
              point when none does)
Decision on escalated clips: `ast` takes the AST head's prediction, `avg`
the argmax of the mean of both tiers' probabilities.

Cost per clip (bit-ops) of a cascade on one (seed, fold): cheap backbone +
cheap head + test escalation rate x (AST backbone + AST head). Two views of
the cheap backbone: fp32 (32 x 32, what the embeddings are) and int8 (8 x 8,
an assumption, same accuracy).

Hull, per cost view and cascade point: at the cascade's mean cost c (over
its (seed, fold) pairs), the best of every static system costing <= c and
every random time-sharing of a static system costing <= c with one costing
more, mixed so the expected cost is c, chosen on mean F1 and cost. Its F1 on
each (seed, fold) is the expected macro F1 of the random mix over --draws
draws of the per-clip assignment (numpy seed --rng-seed). The best static
system costing <= c (front.py's comparator) is reported too.

Tests per cascade point and baseline: paired over (seed, fold), and at fold
level (seeds averaged within each fold); two-sided paired t-test and
Wilcoxon signed-rank. Raw p values here; Holm is applied by summary_idea6.py.

Inner-validation selection (selection_<view>.csv): the same computation on
each fold's inner validation split only, which every threshold was already
set on. Per design: its inner-validation F1 and escalation rate, its cost at
that rate, and the expected F1 of the static systems' best time-sharing at
that cost (static systems scored on the same inner-validation rows; separate
RNG, seed --rng-seed + 1). gain_iv is the mean over (seed, fold) of the
design's F1 minus that hull; gain_iv_per_fold averages it over seeds within
each fold. summary_idea6.py picks each task's primary design from it.

Also: oracle at the same escalation count (escalate first the clips the
cheap head gets wrong and the escalated decision gets right, then clips it
does not change, last the ones it breaks), the full oracle (exactly the
fixable clips), AUROC of 1 - margin for "cheap wrong, AST right" (pooled over
a seed's folds), and top-label ECE (15 bins) of every head on its test folds.

    python scripts/redesign/idea6/analyze_idea6.py --task esc50 --eff-scores R/scores/esc50 \
        --ast-scores STAGE2/scores/esc50 --costs R/costs.json --out R/analysis/esc50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

TAGS = {"esc50": ("effesc50_s{}", "astesc50_s{}", "esc50"),
        "esc5": ("esc_singles_s{}", "astesc5_s{}", "esc50"),
        "urban": ("urban_singles_s{}", "asturban_s{}", "urban8k")}
SEEDS = {"esc50": [0, 1, 2, 3, 4], "esc5": [0, 1, 2, 3, 4], "urban": [0, 1, 2]}
N_CLS = {"esc50": 50, "esc5": 5, "urban": 10}
EFF_KINDS = ["esc", "8", "4", "bitnet"]
AST_HEADS = ["esc", "4"]
EXPENSIVE = ["w8g64", "w4g128", "fp32"]
RULES = ["ast", "avg"]
BUDGETS = {"b05": 0.05, "b10": 0.10, "b20": 0.20, "b30": 0.30, "b50": 0.50}
POINTS = list(BUDGETS) + ["match"]
VIEWS = ["fp32", "int8"]
ECE_BINS = 15


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Scores:
    def __init__(self, path: Path):
        d = np.load(path)
        self.path = path
        self.y, self.tf = d["y"].astype(np.int64), d["test_fold"].astype(int)
        if not np.isfinite(d["test_logits"]).all():
            raise SystemExit(f"{path}: non-finite test logits")
        self.p = softmax(d["test_logits"].astype(np.float64))
        self.folds = sorted(int(k.rsplit("_", 1)[1]) for k in d.files if k.startswith("iv_idx_"))
        self.iv = {f: (d[f"iv_idx_{f}"], softmax(d[f"iv_logits_{f}"].astype(np.float64))) for f in self.folds}
        self.crc = int(d["names_crc32"]) if "names_crc32" in d.files else None


def f1_fast(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))


def f1_batch(y, P, k):
    """Macro F1 of every row of P [D, n] against y [n] (classes present in y or that row)."""
    D = P.shape[0]
    idx = (np.arange(D)[:, None] * (k * k) + y[None, :] * k + P).ravel()
    cm = np.bincount(idx, minlength=D * k * k).reshape(D, k, k)
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float)
    den = cm.sum(1) + cm.sum(2)
    r = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return r.sum(1) / (den > 0).sum(1)


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def candidates(m):
    u = np.unique(m)
    return np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])


def thr_rate(m_iv, rate):
    """Escalate margin < thr; among the reachable escalation counts, the rate closest to `rate`
    (ties: fewer escalations)."""
    c = candidates(m_iv)
    rates = (m_iv[None, :] < c[:, None]).mean(1)
    return c[np.lexsort((rates, np.abs(rates - rate)))[0]]


def thr_match(m_iv, y, pc, pe, target, k):
    """Smallest escalation whose F1 reaches target; else the best F1 (ties: fewer escalations).
    Returns (threshold, reached)."""
    c = candidates(m_iv)
    E = m_iv[None, :] < c[:, None]
    rates = E.mean(1)
    f = f1_batch(y, np.where(E, pe[None, :], pc[None, :]), k)
    ok = np.flatnonzero(f >= target - 1e-12)
    if ok.size:
        return c[ok[np.argmin(rates[ok])]], True
    return c[np.lexsort((rates, -f))[0]], False


def oracle_pred(y, pc, de, n):
    fix, harm = (pc != y) & (de == y), (pc == y) & (de != y)
    order = np.argsort(np.where(fix, 0, np.where(harm, 2, 1)), kind="stable")
    e = np.zeros(len(y), bool)
    e[order[:n]] = True
    return np.where(e, de, pc)


def ece(prob, y, bins=ECE_BINS):
    conf, pred = prob.max(1), prob.argmax(1)
    b = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((b == i).mean() * abs((pred[b == i] == y[b == i]).mean() - conf[b == i].mean())
                     for i in range(bins) if (b == i).any()))


def _p(fn, *a):
    try:
        v = fn(*a).pvalue
        return float(v) if np.isfinite(v) else 0.0  # constant nonzero difference: t is infinite
    except ValueError:
        return 1.0


def tests(x, z, keys):
    """x - z over (seed, fold) keys: pair level and fold level (seeds averaged per fold)."""
    d = np.asarray(x) - np.asarray(z)
    folds = np.array([k[1] for k in keys])
    fm = np.array([d[folds == f].mean() for f in np.unique(folds)])
    out = {"n_pairs": len(d), "n_folds": len(fm), "diff": float(d.mean()), "diff_fold_min": float(fm.min()),
           "diff_fold_max": float(fm.max()), "fold_diffs": json.dumps(np.round(fm, 6).tolist())}
    if np.allclose(d, 0):
        out.update(p_t_pair=1.0, p_w_pair=1.0, p_t_fold=1.0, p_w_fold=1.0)
        return out
    out.update(p_t_pair=_p(stats.ttest_1samp, d, 0.0), p_w_pair=_p(stats.wilcoxon, d),
               p_t_fold=_p(stats.ttest_1samp, fm, 0.0) if not np.allclose(fm, fm[0]) else (0.0 if fm[0] else 1.0),
               p_w_fold=_p(stats.wilcoxon, fm))
    return out


def hull_pair(sp: pd.DataFrame, c: float):
    """Best static system or two-system time-sharing at expected cost c, on mean F1."""
    lo_set = sp[sp.cost <= c * (1 + 1e-12)]
    hi_set = sp[sp.cost > c * (1 + 1e-12)]
    best = None
    for lo in lo_set.itertuples():
        cands = [(lo.f1, lo.system, lo.system, 0.0)]
        for hi in hi_set.itertuples():
            lam = (c - lo.cost) / (hi.cost - lo.cost)
            cands.append(((1 - lam) * lo.f1 + lam * hi.f1, lo.system, hi.system, lam))
        for v in cands:
            if best is None or v[0] > best[0] + 1e-15:
                best = v
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TAGS))
    ap.add_argument("--eff-scores", required=True, type=Path)
    ap.add_argument("--ast-scores", required=True, type=Path)
    ap.add_argument("--costs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--rng-seed", type=int, default=12345)
    a = ap.parse_args(argv)
    task, K = a.task, N_CLS[a.task]
    etag, atag, emb = TAGS[task]
    C = json.loads(a.costs.read_text())
    ast_cost = C["ast_backbone"][emb]
    configs = sorted(ast_cost)
    eff_bb = C["efficientnet_b3"]["bitops"]
    cheap_cost = {v: {k: eff_bb[v] + C["eff_heads"][task][k]["bitops"] for k in EFF_KINDS} for v in VIEWS}
    ast_sys_cost = {(c, h): ast_cost[c]["bitops"] + C["ast_heads"][task][h]["bitops"] for c in configs
                    for h in AST_HEADS}
    crc = {v["names_crc32"] for v in ast_cost.values()}
    a.out.mkdir(parents=True, exist_ok=True)

    static, static_iv, casc, auroc, calib, oracle_full = [], [], [], [], [], []
    preds = {}  # (system, seed, fold) -> test predictions, for the hull draws
    preds_iv = {}  # (system, seed, fold) -> inner-validation predictions, for the selection hull
    ys, ys_iv = {}, {}
    for seed in SEEDS[task]:
        cheap = {k: Scores(a.eff_scores / etag.format(seed) / f"{k}.npz") for k in EFF_KINDS}
        ast = {}
        for c in configs:
            for h in AST_HEADS:
                p = a.ast_scores / atag.format(seed) / h / f"{c}__clean.npz"
                if p.exists():
                    ast[(c, h)] = Scores(p)
        missing = [(c, h) for c in configs for h in AST_HEADS if (c, h) not in ast]
        if missing:
            raise SystemExit(f"seed {seed}: AST scores missing for {missing}")
        ref = cheap["esc"]
        if {ref.crc} != crc or len(crc) != 1:
            raise SystemExit(f"names checksum {ref.crc} != AST sidecars {crc}")
        for s in list(cheap.values()) + list(ast.values()):
            if not ((s.y == ref.y).all() and (s.tf == ref.tf).all() and s.folds == ref.folds
                    and all(np.array_equal(s.iv[f][0], ref.iv[f][0]) for f in ref.folds)):
                raise SystemExit(f"{s.path}: rows, folds or inner splits differ from {ref.path}")
        systems = {f"eff:{k}": (cheap[k], {v: cheap_cost[v][k] for v in VIEWS}) for k in EFF_KINDS}
        systems.update({f"ast:{c}+{h}": (ast[(c, h)], {v: ast_sys_cost[(c, h)] for v in VIEWS})
                        for c in configs for h in AST_HEADS})

        for fold in ref.folds:
            te = np.flatnonzero(ref.tf == fold)
            y = ref.y[te]
            ys[(seed, fold)] = y
            iv_idx = ref.iv[fold][0]
            yv = ref.y[iv_idx]
            ys_iv[(seed, fold)] = yv
            for name, (sc, cost) in systems.items():
                pr_iv = sc.iv[fold][1].argmax(1)
                preds_iv[(name, seed, fold)] = pr_iv
                static_iv.append({"system": name, "seed": seed, "fold": fold, "f1": f1_fast(yv, pr_iv, K),
                                  "cost_fp32": cost["fp32"], "cost_int8": cost["int8"]})
                pt = sc.p[te]
                pr = pt.argmax(1)
                preds[(name, seed, fold)] = pr
                static.append({"system": name, "seed": seed, "fold": fold, "f1": f1_fast(y, pr, K),
                               "acc": float((pr == y).mean()), "cost_fp32": cost["fp32"], "cost_int8": cost["int8"]})
                if name.startswith("eff:") or name.endswith("+esc"):
                    calib.append({"system": name, "seed": seed, "fold": fold, "ece": ece(pt, y),
                                  "conf_gap": float(pt.max(1).mean() - (pr == y).mean())})
            for k in EFF_KINDS:
                pc_te, pc_iv = cheap[k].p[te], cheap[k].iv[fold][1]
                m_te, m_iv = margin(pc_te), margin(pc_iv)
                ac_te, ac_iv = pc_te.argmax(1), pc_iv.argmax(1)
                for e in EXPENSIVE:
                    pe_te, pe_iv = ast[(e, "esc")].p[te], ast[(e, "esc")].iv[fold][1]
                    target = f1_fast(yv, pe_iv.argmax(1), K)
                    e_cost = ast_sys_cost[(e, "esc")]
                    for rule in RULES:
                        if rule == "ast":
                            de_te, de_iv = pe_te.argmax(1), pe_iv.argmax(1)
                        else:
                            de_te, de_iv = ((pc_te + pe_te) / 2).argmax(1), ((pc_iv + pe_iv) / 2).argmax(1)
                        fix = (ac_te != y) & (de_te == y)
                        oracle_full.append({"cheap": k, "expensive": e, "rule": rule, "seed": seed, "fold": fold,
                                            "f1": f1_fast(y, np.where(fix, de_te, ac_te), K),
                                            "rate": float(fix.mean()),
                                            **{f"cost_{v}": cheap_cost[v][k] + fix.mean() * e_cost for v in VIEWS}})
                        for pt_name in POINTS:
                            if pt_name in BUDGETS:
                                t, reached = thr_rate(m_iv, BUDGETS[pt_name]), None
                            else:
                                t, reached = thr_match(m_iv, yv, ac_iv, de_iv, target, K)
                            esc = m_te < t
                            pred = np.where(esc, de_te, ac_te)
                            rate = float(esc.mean())
                            esc_iv = m_iv < t
                            iv_rate = float(esc_iv.mean())
                            casc.append({"cheap": k, "expensive": e, "rule": rule, "point": pt_name, "seed": seed,
                                         "fold": fold, "f1": f1_fast(y, pred, K), "acc": float((pred == y).mean()),
                                         "rate": rate, "iv_rate": iv_rate, "threshold": float(t),
                                         "iv_target_f1": target, "match_reached": reached,
                                         "f1_iv": f1_fast(yv, np.where(esc_iv, de_iv, ac_iv), K),
                                         "oracle_f1": f1_fast(y, oracle_pred(y, ac_te, de_te, int(esc.sum())), K),
                                         **{f"cost_{v}": cheap_cost[v][k] + rate * e_cost for v in VIEWS},
                                         **{f"cost_iv_{v}": cheap_cost[v][k] + iv_rate * e_cost for v in VIEWS}})
        # gate quality, pooled over the seed's folds
        m = ref.tf >= 0
        y = ref.y[m]
        for k in EFF_KINDS:
            pc = cheap[k].p[m]
            score = 1 - margin(pc)
            for e in EXPENSIVE:
                pe = ast[(e, "esc")].p[m]
                tgt = (pc.argmax(1) != y) & (pe.argmax(1) == y)
                wrong = pc.argmax(1) != y
                auroc.append({"cheap": k, "expensive": e, "seed": seed, "n": int(len(y)),
                              "n_cheap_wrong_ast_right": int(tgt.sum()), "n_cheap_wrong": int(wrong.sum()),
                              "auroc_cheap_wrong_ast_right": float(roc_auc_score(tgt, score)),
                              "auroc_cheap_wrong": float(roc_auc_score(wrong, score))})
        print(f"{task} seed {seed}: scored", flush=True)

    S, CA, SV = pd.DataFrame(static), pd.DataFrame(casc), pd.DataFrame(static_iv)
    S.to_csv(a.out / "static.csv", index=False)
    SV.to_csv(a.out / "static_iv.csv", index=False)
    CA.to_csv(a.out / "cascades.csv", index=False)
    pd.DataFrame(auroc).to_csv(a.out / "auroc.csv", index=False)
    pd.DataFrame(calib).to_csv(a.out / "calibration.csv", index=False)
    pd.DataFrame(oracle_full).to_csv(a.out / "oracle_full.csv", index=False)

    rng = np.random.default_rng(a.rng_seed)
    s_rec = {name: dict(zip(zip(g.seed, g.fold), g.f1)) for name, g in S.groupby("system")}
    for view in VIEWS:
        sp = S.groupby("system").agg(f1=("f1", "mean"), cost=(f"cost_{view}", "mean")).reset_index()
        rows = []
        for (k, e, rule, pt_name), g in CA.groupby(["cheap", "expensive", "rule", "point"], sort=False):
            g = g.sort_values(["seed", "fold"])
            keys = list(zip(g.seed, g.fold))
            c = float(g[f"cost_{view}"].mean())
            v_interp, lo, hi, lam = hull_pair(sp, c)
            hull = []
            for key in keys:
                pl = preds[(lo, *key)]
                if lam == 0.0:
                    hull.append(s_rec[lo][key])
                    continue
                ph = preds[(hi, *key)]
                draws = rng.random((a.draws, len(pl))) < lam
                hull.append(float(f1_batch(ys[key], np.where(draws, ph[None, :], pl[None, :]), K).mean()))
            x = g.f1.to_numpy()
            row = {"view": view, "cheap": k, "expensive": e, "rule": rule, "point": pt_name, "f1": x.mean(),
                   "cost": c, "rate": g.rate.mean(), "oracle_f1": g.oracle_f1.mean(),
                   "hull_lo": lo, "hull_hi": hi, "lam": lam, "hull_f1_interp": v_interp, "hull_f1": float(np.mean(hull))}
            row.update({f"hull_{kk}": vv for kk, vv in tests(x, hull, keys).items()})
            ok = sp[sp.cost <= c * (1 + 1e-12)].sort_values(["f1", "cost"], ascending=[False, True])
            comp = ok.iloc[0]
            z = np.array([s_rec[comp.system][kk] for kk in keys])
            row.update({"static_comp": comp.system, "static_comp_cost": comp.cost, "static_comp_f1": comp.f1})
            row.update({f"static_{kk}": vv for kk, vv in tests(x, z, keys).items()})
            rows.append(row)
        pd.DataFrame(rows).to_csv(a.out / f"comparisons_{view}.csv", index=False)
        sp.sort_values("cost").to_csv(a.out / f"static_points_{view}.csv", index=False)
        print(f"{task} view {view}: {len(rows)} cascade points compared", flush=True)

    # Inner-validation selection (no test fold is read): every design's inner-validation F1 minus the
    # time-sharing hull of the static systems' inner-validation F1 at the design's mean inner-validation
    # cost. Own RNG, so the test-side draws above stay as they were.
    rng_iv = np.random.default_rng(a.rng_seed + 1)
    sv_rec = {name: dict(zip(zip(g.seed, g.fold), g.f1)) for name, g in SV.groupby("system")}
    for view in VIEWS:
        sp = SV.groupby("system").agg(f1=("f1", "mean"), cost=(f"cost_{view}", "mean")).reset_index()
        rows = []
        for (k, e, rule, pt_name), g in CA.groupby(["cheap", "expensive", "rule", "point"], sort=False):
            g = g.sort_values(["seed", "fold"])
            keys = list(zip(g.seed, g.fold))
            c = float(g[f"cost_iv_{view}"].mean())
            v_interp, lo, hi, lam = hull_pair(sp, c)
            hull = []
            for key in keys:
                if lam == 0.0:
                    hull.append(sv_rec[lo][key])
                    continue
                pl, ph = preds_iv[(lo, *key)], preds_iv[(hi, *key)]
                draws = rng_iv.random((a.draws, len(pl))) < lam
                hull.append(float(f1_batch(ys_iv[key], np.where(draws, ph[None, :], pl[None, :]), K).mean()))
            d = g.f1_iv.to_numpy() - np.array(hull)
            per_fold = {int(f): float(d[g.fold.to_numpy() == f].mean()) for f in np.unique(g.fold)}
            rows.append({"view": view, "cheap": k, "expensive": e, "rule": rule, "point": pt_name,
                         "f1_iv": g.f1_iv.mean(), "cost_iv": c, "iv_rate": g.iv_rate.mean(), "hull_lo": lo,
                         "hull_hi": hi, "lam": lam, "hull_f1_iv": float(np.mean(hull)), "gain_iv": float(d.mean()),
                         "gain_iv_per_fold": json.dumps(per_fold)})
        pd.DataFrame(rows).to_csv(a.out / f"selection_{view}.csv", index=False)
        print(f"{task} view {view}: inner-validation selection table written", flush=True)
    meta = {"task": task, "n_classes": K, "seeds": SEEDS[task], "draws": a.draws, "rng_seed": a.rng_seed,
            "eff_scores": str(a.eff_scores.resolve()), "ast_scores": str(a.ast_scores.resolve()),
            "costs": str(a.costs.resolve()), "ast_configs": configs, "names_crc32": sorted(crc)}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
