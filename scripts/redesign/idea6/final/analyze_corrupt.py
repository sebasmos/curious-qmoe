"""Idea 6 final round, part B: the picked cascades and the cheap tiers on nine non-additive corruptions, as fixed in
ANALYSIS_PLAN.md (commit 2a05e77). Helpers are Stage 2's analyze.py,
unchanged (metrics, thresholds, hull, tests, score loading, audio statistics, noise_lr).

Scores: clean from the earlier stages' score folders (unchanged), the nine new conditions from --r/scores/<task>.

    python analyze_corrupt.py --task esc50|urban --r R --s2 S2 --s3b S3B --s3c S3C --unp UNP --i7 I7
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
IDEA6 = HERE.parent
ROOT = HERE.parents[3]
for p in (ROOT, IDEA6, IDEA6 / "stage2", HERE):
    sys.path.insert(0, str(p))

import analyze as A  # noqa: E402  Stage 2's helpers
import corrupt as CR  # noqa: E402
from idea5b_common import holm, mix_expected_f1  # noqa: E402

NEW = [CR.cond_label(k, lv) for k in CR.KINDS for lv in CR.LEVELS[k]]
CONDS = ["clean"] + NEW
KINDS = ["esc", "8", "4", "bitnet"]
AST_HEADS = ["esc", "4"]
BUDGETS = A.BUDGETS
PRICE = 1024.0
# the cascade designs fixed in ANALYSIS_PLAN.md: (cheap tier, head kind, budget, AST head, rule)
DESIGN = {"esc50": ("mn20", "4", "b05", "esc", "avg"), "urban": ("mn10", "esc", "b30", "esc", "avg")}


def specs(task, a):
    r = a.r
    if task == "esc50":
        return dict(
            ds="esc50", stats="esc50", seeds=[0, 1, 2, 3, 4], K=50, L=512, hk="esc50",
            cheap={"mn20": dict(old=a.s3c / "scores_cnn/esc50/mn20_esc50_s{}", new=r / "scores/esc50/mn20_esc50_s{}",
                                cfg="mn20_w8a8", dim=1920, side=a.s3c / "emb_cnn/esc50/mn20_w8a8__clean.json"),
                   "mn10": dict(old=a.s3b / "scores_cnn/esc50/mn10_esc50_s{}", new=r / "scores/esc50/mn10_esc50_s{}",
                                cfg="mn10_w8a8", dim=960, side=a.s3b / "emb_cnn/esc50/mn10_w8a8__clean.json")},
            ast=dict(old=a.s2 / "scores_ast_short/esc50/astshort_esc50_s{}", new=r / "scores/esc50/astshort_esc50_s{}",
                     cfg="w4g128", side=a.s2 / "emb_ast_short/esc50/w4g128__clean.json"))
    return dict(
        ds="urban8k", stats="urban", seeds=[0, 1, 2], K=10, L=400, hk="urban",
        cheap={"mn10": dict(old=a.unp / "scores_unpr/urban/mn10unp_urban_s{}", new=r / "scores/urban/mn10unp_urban_s{}",
                            cfg="mn10_w8a8", dim=960, side=a.s3c / "emb_cnn/urban8k_unpadded/mn10_w8a8__clean.json"),
               "mn20": dict(old=a.s3c / "scores_cnn/urban/mn20_urban_s{}", new=r / "scores/urban/mn20_urban_s{}",
                            cfg="mn20_w8a8", dim=1920, side=a.s3c / "emb_cnn/urban8k/mn20_w8a8__clean.json")},
        ast=dict(old=a.s2 / "scores_ast_short/urban/astshort_urban_s{}", new=r / "scores/urban/astshort_urban_s{}",
                 cfg="w4g128", side=a.s2 / "emb_ast_short/urban8k/w4g128__clean.json"))


class Pair:
    """Clean logits from the earlier stage's folder, the new conditions from this run's."""

    def __init__(self, old: Path, new: Path, cfg: str):
        self.old, self.new, self.cfg = old, new, cfg

    def get(self, cond):
        d = self.old if cond == "clean" else self.new
        return d.get(self.cfg, cond)


def ttest_greater(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan")
    if d.std(ddof=1) == 0:
        return 0.0 if d.mean() > 0 else 1.0
    return float(stats.ttest_1samp(d, 0.0, alternative="greater").pvalue)


def fold_means(vals, keys):
    folds = np.array([k[1] for k in keys])
    return np.array([np.mean(np.asarray(vals)[folds == f]) for f in np.unique(folds)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(DESIGN))
    for k in ("r", "s2", "s3b", "s3c", "unp", "i7"):
        ap.add_argument(f"--{k}", required=True, type=Path)
    a = ap.parse_args(argv)
    t0 = time.time()
    task = a.task
    T = specs(task, a)
    K, seeds, ds = T["K"], T["seeds"], T["ds"]
    out = a.r / "analysis" / f"b_{task}"
    out.mkdir(parents=True, exist_ok=True)
    d_tier, d_kind, d_bud, d_ast, d_rule = DESIGN[task]

    # ── score sources ───────────────────────────────────────────────────────
    cheap = {}
    for tier, s in T["cheap"].items():
        for k in KINDS:
            for sd in seeds:
                cheap[(tier, k, sd)] = Pair(A.ScoreDir(Path(str(s["old"]).format(sd)) / k),
                                            A.ScoreDir(Path(str(s["new"]).format(sd)) / k), s["cfg"])
    astp = {(h, sd): Pair(A.ScoreDir(Path(str(T["ast"]["old"]).format(sd)) / h),
                          A.ScoreDir(Path(str(T["ast"]["new"]).format(sd)) / h), T["ast"]["cfg"])
            for h in AST_HEADS for sd in seeds}
    for key, pr in [*cheap.items(), *astp.items()]:
        miss = ([] if pr.old.has(pr.cfg, "clean") else ["clean"]) + [c for c in NEW if not pr.new.has(pr.cfg, c)]
        if miss:
            raise SystemExit(f"{task} {key}: scores missing for {miss}")

    # ── costs (Stage 3c's formulas) ─────────────────────────────────────────
    from costs_idea6 import head_costs

    with contextlib.redirect_stdout(io.StringIO()):
        dims = sorted({768, *(s["dim"] for s in T["cheap"].values())})
        hc = {d: head_costs(d, KINDS)[T["hk"]] for d in dims}
    lengths = np.load(a.s2 / "emb" / ds / "lengths__clean.npy")
    fbank = float(np.mean([A.ast_fbank_frontend_macs(int(n)) for n in lengths])) * 1024
    cheap_cost, ast_cost = {}, {}
    crc_ref = json.loads((a.s2 / "emb" / ds / "fp32__clean.json").read_text())["names_crc32"]
    for tier, s in T["cheap"].items():
        sd_ = json.loads(Path(s["side"]).read_text())
        if sd_["names_crc32"] != crc_ref:
            raise SystemExit(f"{tier}: names checksum differs")
        cc = sd_["cost_per_clip"]
        for k in KINDS:
            cheap_cost[(tier, k)] = cc["frontend_bitops_mean"] + cc["backbone_bitops_rescale1024"] + hc[s["dim"]][k]["bitops"]
    sa = json.loads(Path(T["ast"]["side"]).read_text())
    P_, T_ = A.short_tokens(T["L"])
    un = A.ast_bitops_tok(T["ast"]["cfg"], 0.0, *A.short_tokens(T["L"])[::-1])
    if abs(sa["backbone_cost_per_clip"]["bitops"] - un) > 1e-6 * un or sa["max_length"] != T["L"]:
        raise SystemExit("AST sidecar bit-ops differ")
    bo = A.ast_bitops_tok(T["ast"]["cfg"], PRICE, T_, P_)
    for h in AST_HEADS:
        ast_cost[h] = fbank + bo + hc[768][h]["bitops"]
    # new sidecars must carry the same costs as the clean ones (same lengths)
    new_side = {"esc50": {"mn20": a.r / "emb_cnn/esc50", "mn10": a.r / "emb_cnn/esc50"},
                "urban": {"mn10": a.r / "emb_cnn/urban8k_unpadded", "mn20": a.r / "emb_cnn/urban8k"}}[task]
    for tier, s in T["cheap"].items():
        c0 = json.loads(Path(s["side"]).read_text())["cost_per_clip"]
        for c in NEW:
            c1 = json.loads((new_side[tier] / f"{s['cfg']}__{c}.json").read_text())["cost_per_clip"]
            for kk in ("frontend_bitops_mean", "backbone_bitops_rescale1024"):
                if abs(c1[kk] - c0[kk]) > 1e-9 * abs(c0[kk]):
                    raise SystemExit(f"{tier} {c}: cost {kk} differs from the clean sidecar")
    (out / "costs.json").write_text(json.dumps({
        "cheap": {f"cnn:{t}:w8a8:{k}": v for (t, k), v in cheap_cost.items()},
        "ast": {f"w4g128@{T['L']}+{h}": v for h, v in ast_cost.items()}, "fbank_bitops_mean": fbank}, indent=1))

    # ── audio statistics and the noise-detector gate ────────────────────────
    stems = [str(x) for x in np.load(a.i7 / "audio_stats" / T["stats"] / "clean.npz", allow_pickle=True)["names"]]
    audio = A.AudioStats(a.i7 / "audio_stats" / T["stats"], stems)
    new_stats = {}
    for c in NEW:
        z = np.load(a.r / "audio_stats" / T["stats"] / f"{c}.npz", allow_pickle=True)
        if [str(x) for x in z["names"]] != stems:
            raise SystemExit(f"{c}: audio statistics rows differ")
        new_stats[c] = z["stats"]
    av = A.field_arrays(len(stems))[1]

    def stats_rows(c, idx):
        return audio.rows("clean", idx, "test") if c == "clean" else new_stats[c][idx]

    # ── one pass over seeds and folds ───────────────────────────────────────
    ref0 = cheap[(d_tier, d_kind, seeds[0])].get("clean")
    folds = sorted(ref0["iv"])
    n_te = np.array([np.sum(ref0["fold"] == f) for f in folds])
    A.NB_RATIO["task"] = float(np.mean(n_te / (len(ref0["y"]) - n_te)))
    static_rows, casc_rows, esc_rows, au_rows, det_rows = [], [], [], [], []
    preds, ys = {}, {}
    au_acc = {}
    for sd in seeds:
        ref = cheap[(d_tier, d_kind, sd)].get("clean")
        for key, pr in cheap.items():
            if key[2] != sd:
                continue
            for c in CONDS:
                r_ = pr.get(c)
                if not ((r_["y"] == ref["y"]).all() and (r_["fold"] == ref["fold"]).all() and sorted(r_["iv"]) == folds
                        and all(np.array_equal(r_["iv"][f][0], ref["iv"][f][0]) for f in folds)):
                    raise SystemExit(f"{key} {c}: rows differ")
                if r_["crc"] is not None and r_["crc"] != crc_ref:
                    raise SystemExit(f"{key} {c}: names checksum differs")
        for key, pr in astp.items():
            if key[1] != sd:
                continue
            for c in CONDS:
                r_ = pr.get(c)
                if not ((r_["y"] == ref["y"]).all() and (r_["fold"] == ref["fold"]).all()
                        and all(np.array_equal(r_["iv"][f][0], ref["iv"][f][0]) for f in folds)):
                    raise SystemExit(f"ast {key} {c}: rows differ")
        y_all = ref["y"]
        for fold in folds:
            te = np.flatnonzero(ref["fold"] == fold)
            iv = ref["iv"][fold][0]
            y = y_all[te]
            key = (sd, fold)
            ys[key] = y
            noise = A.fit_lr(audio.rows("field", iv, "val"), (av[iv] != "clean").astype(int))
            nz = {c: noise(stats_rows(c, te)) for c in CONDS}
            # thresholds from clean inner validation, per tier and kind
            thr = {}
            for tier in T["cheap"]:
                for k in KINDS:
                    m_iv = A.margin(A.softmax(cheap[(tier, k, sd)].get("clean")["iv"][fold][1]))
                    for b, rate in BUDGETS.items():
                        thr[(tier, k, b)] = A.thr_rate(m_iv, rate)
            pe = {h: {c: A.softmax(astp[(h, sd)].get(c)["z"][te]) for c in CONDS} for h in AST_HEADS}
            pc = {(t, k): {c: A.softmax(cheap[(t, k, sd)].get(c)["z"][te]) for c in CONDS}
                  for t in T["cheap"] for k in KINDS}
            for c in CONDS:
                for (t, k), pp in pc.items():
                    pr_ = pp[c].argmax(1)
                    preds[(f"{t}:{k}", c, sd, fold)] = pr_.astype(np.int16)
                    static_rows.append({"system": f"{t}:{k}", "condition": c, "seed": sd, "fold": fold,
                                        "f1": A.f1_fast(y, pr_, K), "acc": float((pr_ == y).mean()),
                                        "cost": cheap_cost[(t, k)]})
                    m = A.margin(pp[c])
                    for b in BUDGETS:
                        esc_rows.append({"tier": t, "kind": k, "budget": b, "condition": c, "seed": sd, "fold": fold,
                                         "rate": float((m < thr[(t, k, b)]).mean()), "error": float((pr_ != y).mean())})
                    a_ = au_acc.setdefault((t, k, c, sd), {"m": [], "n": [], "wrong": [], "fix": []})
                    a_["m"].append(1 - m)
                    a_["n"].append(nz[c])
                    a_["wrong"].append(pr_ != y)
                    a_["fix"].append((pr_ != y) & (pe["esc"][c].argmax(1) == y))
                    if c != "clean":  # detection of the corrupted clips against the same fold's clean clips
                        mc = 1 - A.margin(pp["clean"])
                        lab = np.r_[np.zeros(len(te)), np.ones(len(te))]
                        det_rows.append({"tier": t, "kind": k, "condition": c, "seed": sd, "fold": fold,
                                         "auroc_margin": roc_auc_score(lab, np.r_[mc, 1 - m]),
                                         "auroc_noise_lr": roc_auc_score(lab, np.r_[nz["clean"], nz[c]])})
                for h in AST_HEADS:
                    pr_ = pe[h][c].argmax(1)
                    preds[(f"ast:{h}", c, sd, fold)] = pr_.astype(np.int16)
                    static_rows.append({"system": f"ast:{h}", "condition": c, "seed": sd, "fold": fold,
                                        "f1": A.f1_fast(y, pr_, K), "acc": float((pr_ == y).mean()),
                                        "cost": ast_cost[h]})
            # the picked cascade
            p_c = pc[(d_tier, d_kind)]
            for c in CONDS:
                m, ac = A.margin(p_c[c]), p_c[c].argmax(1)
                escm = m < thr[(d_tier, d_kind, d_bud)]
                q = (p_c[c] + pe[d_ast][c]) / 2 if d_rule == "avg" else pe[d_ast][c]
                de = q.argmax(1)
                pred = np.where(escm, de, ac)
                rate = float(escm.mean())
                n_esc = int(escm.sum())
                rnd = float(mix_expected_f1(y, ac.astype(np.int64), de.astype(np.int64), rate, K)[0])
                nsy = A.f1_fast(y, np.where(A.top_k(nz[c], n_esc), de, ac), K)
                orc = A.f1_fast(y, A.oracle_pred(y, ac, de, n_esc), K)
                casc_rows.append({"condition": c, "seed": sd, "fold": fold, "f1": A.f1_fast(y, pred, K),
                                  "acc": float((pred == y).mean()), "rate": rate, "n_esc": n_esc,
                                  "cost": cheap_cost[(d_tier, d_kind)] + rate * ast_cost[d_ast],
                                  "random_f1": rnd, "noise_f1": nsy, "oracle_f1": orc})
        print(f"{task} seed {sd}: {time.time() - t0:.0f}s", flush=True)
        for pr in [*cheap.values(), *astp.values()]:
            for d in (pr.old, pr.new):
                d.cache.clear()

    S = pd.DataFrame(static_rows)
    S.to_csv(out / "static.csv", index=False)
    CA = pd.DataFrame(casc_rows)
    ESC = pd.DataFrame(esc_rows)
    ESC.to_csv(out / "escalation.csv", index=False)
    pd.DataFrame(det_rows).to_csv(out / "detect.csv", index=False)
    rows = []
    for (t, k, c, sd), a_ in au_acc.items():
        w, mm, nn_, fx = (np.concatenate(a_[x]) for x in ("wrong", "m", "n", "fix"))
        ok = 0 < w.sum() < len(w)
        rows.append({"tier": t, "kind": k, "condition": c, "seed": sd, "error": float(w.mean()),
                     "auroc_wrong": roc_auc_score(w, mm) if ok else np.nan,
                     "auroc_wrong_noise_lr": roc_auc_score(w, nn_) if ok else np.nan,
                     "auroc_fixable": roc_auc_score(fx, mm) if 0 < fx.sum() < len(fx) else np.nan,
                     "n_fixable": int(fx.sum())})
    pd.DataFrame(rows).to_csv(out / "gate_auroc.csv", index=False)

    # ── the hull on each condition (restricted to the statics extracted here) ──
    s_rec = {(n, c): dict(zip(zip(g.seed, g.fold), g.f1)) for (n, c), g in S.groupby(["system", "condition"])}
    res = []
    for c in CONDS:
        g = CA[CA.condition == c].sort_values(["seed", "fold"])
        keys = list(zip(g.seed, g.fold))
        sp = S[S.condition == c].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
        cost = float(g.cost.mean())
        v_interp, lo, hi, lam = A.hull_pair(sp, cost)
        hull = []
        for kk in keys:
            if lam == 0.0:
                hull.append(s_rec[(lo, c)][kk])
            else:
                hull.append(float(mix_expected_f1(ys[kk], preds[(lo, c, *kk)].astype(np.int64),
                                                  preds[(hi, c, *kk)].astype(np.int64), lam, K)[0]))
        xv = g.f1.to_numpy()
        row = {"condition": c, "f1": float(xv.mean()), "cost": cost, "rate": float(g.rate.mean()),
               "hull_lo": lo, "hull_hi": hi, "lam": lam, "hull_f1": float(np.mean(hull)),
               "random_f1": float(g.random_f1.mean()), "noise_f1": float(g.noise_f1.mean()),
               "oracle_f1": float(g.oracle_f1.mean()),
               "cheap_f1": float(np.mean([s_rec[(f"{d_tier}:{d_kind}", c)][kk] for kk in keys])),
               "ast_f1": float(np.mean([s_rec[(f"ast:{d_ast}", c)][kk] for kk in keys]))}
        for nm, comp in (("hull", hull), ("random", g.random_f1.to_numpy()), ("noise", g.noise_f1.to_numpy())):
            row.update({f"{nm}_{kk}": vv for kk, vv in A.tests(xv, comp, keys).items()})
        res.append(row)
    B2 = pd.DataFrame(res)
    B2.to_csv(out / "cascade_vs_comparators.csv", index=False)

    # ── B1: the escalation trend of the picked cheap tier at its budget ──
    e = ESC[(ESC.tier == d_tier) & (ESC.kind == d_kind) & (ESC.budget == d_bud)]
    piv = {c: e[e.condition == c].sort_values(["seed", "fold"]) for c in CONDS}
    keys = list(zip(piv["clean"].seed, piv["clean"].fold))
    b1 = []
    for kind in CR.KINDS:
        chain = ["clean"] + [CR.cond_label(kind, lv) for lv in CR.LEVELS[kind]]
        for j in range(3):
            lo_c, hi_c = chain[j], chain[j + 1]
            d = fold_means(piv[hi_c].rate.to_numpy() - piv[lo_c].rate.to_numpy(), keys)
            b1.append({"task": task, "corruption": kind, "step": f"{lo_c} -> {hi_c}", "rate_from": float(piv[lo_c].rate.mean()),
                       "rate_to": float(piv[hi_c].rate.mean()), "diff": float(d.mean()), "n_folds": len(d),
                       "fold_diffs": json.dumps(np.round(d, 5).tolist()), "p_greater": ttest_greater(d)})
    pd.DataFrame(b1).to_csv(out / "b1_steps_raw.csv", index=False)

    meta = {"task": task, "design": {"cheap": f"{d_tier} W8A8 + head {d_kind}", "budget": d_bud, "ast": f"w4g128@{T['L']} head {d_ast}",
                                     "rule": d_rule}, "K": K, "seeds": seeds, "folds": folds, "nb_ratio": A.NB_RATIO["task"],
            "conditions": CONDS, "seconds": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{task}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
