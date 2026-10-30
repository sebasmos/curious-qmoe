"""Precision gaps, error-predicting signals and backbone cascades (idea 5, Stage 1).

Reads the per-condition head scores of score_heads.py and the backbone costs
written next to the embeddings by extract_conditions.py.

System = (backbone precision, head precision). Reference = FP32 backbone and
FP32 head on the same audio (same noise kind and SNR).

  F1 table   macro F1 per system and audio condition, mean over the official
             test folds (each fold's macro F1, then the mean, as benchmark.py
             reports), with the gap to the reference in points.
  AUROC      target = clips the system gets wrong and the reference gets right
             (pooled over test folds). Signals, all computed from the system's
             own pass or a second cheap pass: 1 - softmax margin (top1 - top2),
             softmax entropy, MC-dropout variance of the head, total-variation
             distance to the same head on another backbone precision, and to
             another head precision on the same backbone.
  cascade    low system first, the reference (FP32 backbone re-run plus FP32
             head) on escalation; a clip escalates when its score is above the
             threshold. Cost per clip = backbone + head bit-ops of the low system
             (MC passes counted), plus the escalated fraction times the
             reference's bit-ops.
             honest points: threshold chosen per fold on that fold's inner
               validation rows (same condition, same pipeline), applied once to
               the test fold. max_f1 = best inner-validation macro F1 (ties: fewer
               escalations); budget_<S> = escalation rate whose expected cost
               equals single system S, rate set on inner validation.
             upper bound (tuned on test): per fold, escalate the top q of test
               clips by score, q swept on a grid, the best q picked by test F1.
             oracle: escalate exactly the clips the target set holds.
  front      a point beats the single-precision front when its F1 is above
             every single system on the same audio whose cost is at most its own.

    python scripts/redesign/idea5/analyze.py --scores SCORES --emb-dir EMB \
        --runs-root outputs --task esc5=astesc5_a_s0,astesc5_b_s0 --out OUT
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

REF = ("fp32", "esc")
HEAD_ORDER = ["bitnet", "1", "4", "8", "esc"]
BACKBONE_ORDER = ["w2", "w3", "w4a8", "w4", "w8", "fp32"]
HEAD_PAIR = {"esc": "4", "8": "4", "4": "1", "1": "bitnet", "bitnet": "1"}
BACKBONE_PAIR = {"fp32": "w8", "w8": "w4", "w4": "w3", "w3": "w2", "w2": "w3", "w4a8": "w4"}
CASCADE_LOW = ["w8", "w4", "w4a8", "w3", "w2"]
CASCADE_SIGNALS = ["margin", "entropy", "mcvar"]
Q_GRID = np.round(np.linspace(0, 1, 101), 2)
# the claimed operating points (low, high, signal, point); every other honest point is exploratory
PRIMARY = {("w4a8+esc", "fp32+esc", "margin", "max_f1"), ("w4a8+esc", "fp32+esc", "margin", "budget_w8+esc"),
           ("w4a8+esc", "w8+esc", "margin", "max_f1")}
N_BOOT = 1000


def boot_diff(y, pa, pb, folds, seed=0):
    """95% interval of F1(pa) - F1(pb), test clips resampled within each fold (paired)."""
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(folds == k) for k in np.unique(folds)]
    d = []
    for _ in range(N_BOOT):
        r = [rng.choice(m, len(m)) for m in idx]
        d.append(np.mean([macro_f1(y[m], pa[m]) - macro_f1(y[m], pb[m]) for m in r]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def parse_cond(c):
    m = re.match(r"([a-z0-9]+)__([a-z]+)(?:__(-?[\d.]+)db)?$", c)
    return m.group(1), m.group(2) + ("" if m.group(3) is None else f"@{m.group(3)}")


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macro_f1(y, p):
    """sklearn's macro F1 (zero_division=0) over the labels in y or p, via 2TP / (2TP + FP + FN)."""
    labels = np.union1d(y, p)
    tp = np.array([np.sum((y == c) & (p == c)) for c in labels], float)
    fp = np.array([np.sum((y != c) & (p == c)) for c in labels], float)
    fn = np.array([np.sum((y == c) & (p != c)) for c in labels], float)
    return float(np.mean(2 * tp / (2 * tp + fp + fn)))


def fold_f1(y, pred, folds):
    return float(np.mean([macro_f1(y[folds == k], pred[folds == k]) for k in np.unique(folds)]))


class Sys:
    """One (backbone, head) system on one audio condition."""

    def __init__(self, npz):
        d = np.load(npz)
        self.y, self.fold = d["y"], d["test_fold"].astype(int)
        self.p = softmax(d["test_logits"].astype(np.float64))
        self.mcvar = d["test_mcvar"].astype(np.float64)
        self.pred = self.p.argmax(1)
        self.iv = {int(k.split("_")[-1]): (d[k], softmax(d[f"iv_logits_{k.split('_')[-1]}"].astype(np.float64)),
                                            d[f"iv_mcvar_{k.split('_')[-1]}"].astype(np.float64))
                   for k in d.files if k.startswith("iv_idx_")}
        self.f1 = fold_f1(self.y, self.pred, self.fold)


def own_scores(p, mcvar):
    s = np.sort(p, 1)
    return {"margin": 1 - (s[:, -1] - s[:, -2]), "entropy": -(p * np.log(np.clip(p, 1e-12, 1))).sum(1),
            "mcvar": mcvar}


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(1)


def auroc(target, score):
    if target.all() or not target.any():
        return None
    return float(roc_auc_score(target, score))


def candidate_thresholds(scores):
    u = np.unique(scores)
    return np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])


def threshold_max_f1(s, y, pl, ph):
    best = None
    for t in candidate_thresholds(s):
        esc = s > t
        key = (-macro_f1(y, np.where(esc, ph, pl)), esc.mean())
        if best is None or key < best[0]:
            best = (key, t)
    return best[1]


def threshold_rate(s, rate):
    """Largest escalation (score > t) not above rate on these rows."""
    best = np.inf
    for t in candidate_thresholds(s)[::-1]:
        if (s > t).mean() <= rate + 1e-12:
            best = t
        else:
            break
    return best


def head_costs(runs_root, tags):
    from omegaconf import OmegaConf
    from scripts.analysis.cascade import cost, load_model

    out = {}
    for tag in tags:
        for run in sorted((runs_root / tag).iterdir()):
            if not (run / "summary.json").exists():
                continue
            cfg = OmegaConf.load(run / "config.yaml")
            import pandas as pd
            n_cls = int(pd.read_csv(cfg.experiment.datasets.esc.csv,
                                    usecols=[cfg.experiment.datasets.esc.get("label_col", "class_id")]).nunique().iloc[0])
            with contextlib.redirect_stdout(io.StringIO()):
                model, _ = load_model(run, 1, 768, n_cls)
            out[run.name] = float(cost(model)[1])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--emb-dir", required=True, type=Path)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--task", action="append", required=True, help="name=tag1,tag2")
    ap.add_argument("--mc-passes", type=int, default=10)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    bb_cost = {json.loads(p.read_text())["backbone"]: json.loads(p.read_text())["backbone_cost_per_clip"]["bops"]
               for p in args.emb_dir.glob("*.json") if p.name != "bg_plan.json"}
    report = {"backbone_bops": bb_cost, "tasks": {}}

    for spec in args.task:
        task, tags = spec.split("=")
        tags = tags.split(",")
        hc = head_costs(args.runs_root, tags)
        systems = {}  # (audio, backbone, head) -> Sys
        for tag in tags:
            for d in sorted((args.scores / tag).iterdir()):
                for npz in sorted(d.glob("*.npz")):
                    b, audio = parse_cond(npz.stem)
                    systems[(audio, b, d.name)] = Sys(npz)
        audios = sorted({a for a, _, _ in systems}, key=lambda a: (a != "clean", a))
        heads = [h for h in HEAD_ORDER if any(k[2] == h for k in systems)]

        def sys_cost(b, h, passes=1):
            return bb_cost[b] + passes * hc[h]

        rows, aurocs, cascades, front_checks = [], [], [], []
        for audio in audios:
            if (audio, *REF) not in systems:
                continue
            ref = systems[(audio, *REF)]
            singles = [(k[1], k[2], s.f1, sys_cost(k[1], k[2])) for k, s in systems.items() if k[0] == audio]

            def best_single_at(c):
                ok = [f for _, _, f, cc in singles if cc <= c * (1 + 1e-9)]
                return max(ok) if ok else None

            def best_single_key(c):
                ok = [(f, sb, sh) for sb, sh, f, cc in singles if cc <= c * (1 + 1e-9)]
                return max(ok)[1:] if ok else None

            for (a, b, h), s in sorted(systems.items(), key=lambda kv: (BACKBONE_ORDER.index(kv[0][1]),
                                                                        HEAD_ORDER.index(kv[0][2]))):
                if a != audio:
                    continue
                folds_f1 = [macro_f1(s.y[s.fold == k], s.pred[s.fold == k]) for k in np.unique(s.fold)]
                rows.append({"audio": audio, "backbone": b, "head": h, "f1": s.f1, "f1_sd": float(np.std(folds_f1)),
                             "gap_pts": 100 * (ref.f1 - s.f1), "bops": sys_cost(b, h),
                             "accuracy": float((s.pred == s.y).mean())})
                if (b, h) == REF:
                    continue
                target = (s.pred != s.y) & (ref.pred == s.y)
                sc = own_scores(s.p, s.mcvar)
                rec = {"audio": audio, "backbone": b, "head": h, "gap_pts": 100 * (ref.f1 - s.f1),
                       "n_target": int(target.sum()), "n_errors": int((s.pred != s.y).sum()),
                       "auroc": {k: auroc(target, v) for k, v in sc.items()},
                       "auroc_any_error": {k: auroc(s.pred != s.y, v) for k, v in sc.items()}}
                h2 = HEAD_PAIR.get(h)
                if (audio, b, h2) in systems:
                    o = systems[(audio, b, h2)]
                    rec["auroc"][f"head_tv_vs_{h2}"] = auroc(target, tv(s.p, o.p))
                    rec["auroc"][f"head_disagree_vs_{h2}"] = auroc(target, (s.pred != o.pred).astype(float))
                b2 = BACKBONE_PAIR.get(b)
                if (audio, b2, h) in systems:
                    o = systems[(audio, b2, h)]
                    rec["auroc"][f"backbone_tv_vs_{b2}"] = auroc(target, tv(s.p, o.p))
                    rec["auroc"][f"backbone_disagree_vs_{b2}"] = auroc(target, (s.pred != o.pred).astype(float))
                aurocs.append(rec)

            # backbone cascades: low system -> FP32 backbone + FP32 head (the reference), and,
            # as an extra variant, -> W8 backbone + FP32 head
            for tgt in (REF, ("w8", "esc")):
                if (audio, *tgt) not in systems:
                    continue
                hi = systems[(audio, *tgt)]
                for b in CASCADE_LOW:
                    for h in ("esc", "4"):
                        if (audio, b, h) not in systems or b == tgt[0]:
                            continue
                        low = systems[(audio, b, h)]
                        c_ref = sys_cost(*tgt)
                        target = (low.pred != low.y) & (hi.pred == low.y)
                        oracle_pred = np.where(target, hi.pred, low.pred)
                        base = {"audio": audio, "low": f"{b}+{h}", "high": "+".join(tgt),
                                "low_f1": low.f1, "high_f1": hi.f1,
                                "low_bops": sys_cost(b, h), "high_bops": c_ref}
                        q_or = float(target.mean())
                        c = sys_cost(b, h) + q_or * c_ref
                        cascades.append({**base, "signal": "oracle", "point": "oracle", "escalation": q_or,
                                         "f1": fold_f1(low.y, oracle_pred, low.fold), "bops": c,
                                         "best_single_at_cost": best_single_at(c)})
                        for sig in CASCADE_SIGNALS:
                            passes = 1 + (args.mc_passes if sig == "mcvar" else 0)
                            c_low = sys_cost(b, h, passes)
                            s_te = own_scores(low.p, low.mcvar)[sig]
                            # upper bound (tuned on test): per fold escalate the top q by score
                            curve = []
                            for q in Q_GRID:
                                pred = low.pred.copy()
                                for k in np.unique(low.fold):
                                    m = np.flatnonzero(low.fold == k)
                                    n_esc = int(round(q * len(m)))
                                    if n_esc:
                                        top = m[np.argsort(-s_te[m], kind="stable")[:n_esc]]
                                        pred[top] = hi.pred[top]
                                cc = c_low + q * c_ref
                                curve.append({"q": float(q), "f1": fold_f1(low.y, pred, low.fold), "bops": cc,
                                              "best_single_at_cost": best_single_at(cc)})
                            cascades.append({**base, "signal": sig, "point": "upper_bound_curve (tuned on test)",
                                             "curve": curve})
                            # honest points: thresholds on each fold's inner validation rows
                            budgets = {"max_f1": None}
                            for (sb, sh, _, sc_) in singles:
                                rate = (sc_ - c_low) / c_ref
                                if 0 < rate < 1 and sb != b and sh in ("esc", h):
                                    budgets[f"budget_{sb}+{sh}"] = rate
                            for point, rate in budgets.items():
                                pred, esc_all, iv_esc = low.pred.copy(), np.zeros(len(low.y), bool), []
                                for k in np.unique(low.fold):
                                    iv_idx, p_iv_low, mc_iv_low = low.iv[k]
                                    _, p_iv_ref, _ = hi.iv[k]
                                    s_iv = own_scores(p_iv_low, mc_iv_low)[sig]
                                    y_iv = low.y[iv_idx]
                                    if rate is None:
                                        t = threshold_max_f1(s_iv, y_iv, p_iv_low.argmax(1), p_iv_ref.argmax(1))
                                    else:
                                        t = threshold_rate(s_iv, rate)
                                    iv_esc.append(float((s_iv > t).mean()))
                                    m = low.fold == k
                                    e = m & (s_te > t)
                                    esc_all |= e
                                    pred[e] = hi.pred[e]
                                q = float(esc_all.mean())
                                cc = c_low + q * c_ref
                                f1 = fold_f1(low.y, pred, low.fold)
                                bs = best_single_at(cc)
                                rec = {**base, "signal": sig, "point": point, "target_rate": rate,
                                       "inner_val_escalation": float(np.mean(iv_esc)), "escalation": q,
                                       "f1": f1, "bops": cc, "best_single_at_cost": bs,
                                       "beats_front": bool(bs is not None and f1 > bs),
                                       "primary": (base["low"], base["high"], sig, point) in PRIMARY}
                                if rec["primary"] and bs is not None:
                                    k = best_single_key(cc)
                                    rec["best_single"] = "+".join(k)
                                    rec["diff_vs_front_ci95"] = boot_diff(low.y, pred, systems[(audio, *k)].pred,
                                                                          low.fold)
                                cascades.append(rec)
        report["tasks"][task] = {"head_bops": hc, "f1_table": rows, "auroc": aurocs, "cascades": cascades}
    (args.out / "analysis.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out / 'analysis.json'}")


if __name__ == "__main__":
    main()
