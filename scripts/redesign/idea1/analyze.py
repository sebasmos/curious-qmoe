"""Redesign idea 1, stage 1: uncertainty-escalation cascades against the single-model front.

A cheap head L in {bitnet, 1, 2, 4} answers every clip; clips whose uncertainty
signal exceeds a threshold are re-run on an expensive head H in {8, 16, esc}.
Signals: 1-maxp, negative softmax margin, entropy, temperature-scaled entropy
(T fit by NLL on the fold's inner validation), MC-dropout predictive variance
(10 passes, and the first 4 of them; extra passes are counted in the cost).
Thresholds come from inner validation only:
  match  fewest escalations whose inner-val cascade F1 reaches H's inner-val F1
         (if none does, the threshold with the best inner-val F1, flagged)
  b10/b20/b30  the inner-val quantile that escalates 10/20/30 percent of inner-val clips
Cost per clip = passes x bit-ops(L) + (escalated fraction on test) x bit-ops(H).
Front: the best single model (by mean test F1 over the same seed x fold units)
among those with bit-ops <= the operating point's mean bit-ops. Paired
differences are over seed x fold units.
"Upper bound (tuned on test)": the same signals ranked on test at 10/20/30
percent escalation, reported only as a ceiling.

usage: python analyze.py --name urban --out DIR EXTRACT_DIR [EXTRACT_DIR ...]   (one dir per seed)
"""
import argparse, json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

MODELS = ["bitnet", "1", "2", "4", "8", "16", "esc", "qesc"]
CHEAP = ["bitnet", "1", "2", "4"]
EXP = ["8", "16", "esc"]
SIGS = ["1-maxp", "margin", "entropy", "T-entropy", "mc4_var", "mc10_var"]
RULES = ["match", "b10", "b20", "b30"]
BUDGET = {"b10": 0.10, "b20": 0.20, "b30": 0.30}
EPS = 1e-12


def load_units(dirs):
    units, costs, sizes, nc = [], None, {}, None
    for d in dirs:
        chk = json.loads((d / "checks.json").read_text())
        costs = {m: float(c[1]) for m, c in chk["costs"].items()}
        nc = chk["n_classes"]
        for c in chk["checks"]:
            sizes[c["model"]] = c["size_bytes"]
        for k in chk["folds"]:
            u = dict(np.load(d / f"fold_{k}.npz"))
            u["seed"], u["fold"] = chk["seed"], k
            units.append(u)
    return units, costs, sizes, nc


def mf1(y, p, nc):
    """sklearn macro F1 (labels present in y or pred), fast."""
    cm = np.bincount(y * nc + p, minlength=nc * nc).reshape(nc, nc)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f[denom > 0].mean())


def softmax(l, T=1.0):
    z = l / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def entropy(p):
    return -(p * np.log(np.clip(p, EPS, 1))).sum(1)


def fit_T(l, y):
    def nll(lt):
        p = softmax(l, np.exp(lt))
        return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def signals(u, L):
    """{signal: (score_iv, score_te, passes)}; higher score = more uncertain."""
    out = {}
    T = fit_T(u[f"l_iv_{L}"], u["y_iv"])
    sc = {}
    for part in ("iv", "te"):
        p = softmax(u[f"l_{part}_{L}"])
        s = np.sort(p, 1)
        mc = u[f"mc_{part}_{L}"]
        sc[part] = {"1-maxp": 1 - s[:, -1], "margin": -(s[:, -1] - s[:, -2]), "entropy": entropy(p),
                    "T-entropy": entropy(softmax(u[f"l_{part}_{L}"], T)),
                    "mc4_var": mc[:4].var(0).sum(1), "mc10_var": mc.var(0).sum(1)}
    passes = {"1-maxp": 1, "margin": 1, "entropy": 1, "T-entropy": 1, "mc4_var": 5, "mc10_var": 11}
    for g in SIGS:
        out[g] = (sc["iv"][g], sc["te"][g], passes[g])
    return out, T


def threshold(rule, s_iv, y_iv, pl, ph, nc):
    """Escalate clips with score > t. Returns (t, reached_target_or_None)."""
    if rule in BUDGET:
        return float(np.quantile(s_iv, 1 - BUDGET[rule])), None
    target = mf1(y_iv, ph, nc)
    u = np.unique(s_iv)
    cands = np.concatenate([[np.inf], ((u[:-1] + u[1:]) / 2)[::-1], [-np.inf]])
    best = None
    for t in cands:  # fewest escalations first
        f = mf1(y_iv, np.where(s_iv > t, ph, pl), nc)
        if f >= target - 1e-12:
            return float(t), True
        if best is None or f > best[1]:
            best = (float(t), f)
    return best[0], False


def paired(d):
    d = np.asarray(d)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("dirs", type=Path, nargs="+")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    units, COST, SIZE, NC = load_units(a.dirs)
    nu = len(units)
    lines = [f"# {a.name}: {nu} seed x fold units "
             f"({sorted(set((u['seed'], u['fold']) for u in units))})"]

    # singles and front
    F1 = {m: np.array([mf1(u["y_te"], u[f"l_te_{m}"].argmax(1), NC) for u in units]) for m in MODELS}
    F1iv = {m: np.array([mf1(u["y_iv"], u[f"l_iv_{m}"].argmax(1), NC) for u in units]) for m in MODELS}
    lines.append("## singles: model | test F1 mean (sd) | inner-val F1 | bit-ops M | size B")
    for m in sorted(MODELS, key=lambda m: COST[m]):
        lines.append(f"{m:6s} | {F1[m].mean():.4f} ({F1[m].std(ddof=1) if nu > 1 else 0:.4f}) | "
                     f"{F1iv[m].mean():.4f} | {COST[m] / 1e6:.1f} | {SIZE[m]}")

    def best_single(c):
        ok = [m for m in MODELS if COST[m] <= c * (1 + 1e-9)]
        return max(ok, key=lambda m: F1[m].mean()) if ok else None

    front = []
    for m in sorted(MODELS, key=lambda m: COST[m]):
        if best_single(COST[m]) == m:
            front.append(m)
    lines.append(f"front (bit-ops ascending, each beats every cheaper model): {front}")

    # AUROC
    auroc = {}
    lines.append("\n## AUROC, pooled over units: (i) L wrong | (ii) L wrong and H right, H = 8 / 16 / esc")
    sig_cache, temps = {}, {}
    for L in CHEAP:
        per = [signals(u, L) for u in units]
        sig_cache[L] = [p[0] for p in per]
        temps[L] = [p[1] for p in per]
        lw = np.concatenate([u[f"l_te_{L}"].argmax(1) != u["y_te"] for u in units])
        row = []
        for g in SIGS:
            s = np.concatenate([sc[g][1] for sc in sig_cache[L]])
            r = {"L_wrong": roc_auc_score(lw, s)}
            for H in EXP:
                fix = lw & np.concatenate([u[f"l_te_{H}"].argmax(1) == u["y_te"] for u in units])
                r[f"fix_{H}"] = roc_auc_score(fix, s)
            auroc[f"{L}|{g}"] = r
            row.append(f"{g} {r['L_wrong']:.3f}|{r['fix_8']:.3f}/{r['fix_16']:.3f}/{r['fix_esc']:.3f}")
        lines.append(f"{L:6s} T={np.mean(temps[L]):.2f}: " + "  ".join(row))

    # cascades
    ops = []
    for L in CHEAP:
        for H in EXP:
            for g in SIGS:
                for rule in RULES:
                    f1s, costs_u, fr_iv, fr_te, hits = [], [], [], [], []
                    for u, sc in zip(units, sig_cache[L]):
                        s_iv, s_te, passes = sc[g]
                        pl_iv, ph_iv = u[f"l_iv_{L}"].argmax(1), u[f"l_iv_{H}"].argmax(1)
                        t, hit = threshold(rule, s_iv, u["y_iv"], pl_iv, ph_iv, NC)
                        e = s_te > t
                        f1s.append(mf1(u["y_te"], np.where(e, u[f"l_te_{H}"].argmax(1), u[f"l_te_{L}"].argmax(1)), NC))
                        costs_u.append(passes * COST[L] + e.mean() * COST[H])
                        fr_iv.append((s_iv > t).mean()); fr_te.append(e.mean()); hits.append(hit)
                    f1s, c = np.array(f1s), float(np.mean(costs_u))
                    b = best_single(c)
                    d = f1s - F1[b] if b else None
                    ops.append({"L": L, "H": H, "signal": g, "rule": rule, "f1": f1s.mean(),
                                "f1_units": f1s.tolist(), "bitops": c, "esc_frac_iv": float(np.mean(fr_iv)),
                                "esc_frac_test": float(np.mean(fr_te)),
                                "match_reached": (int(sum(bool(h) for h in hits)) if rule == "match" else None),
                                "best_single": b, "best_single_f1": float(F1[b].mean()) if b else None,
                                "diff": float(d.mean()) if b else None,
                                "wins": int((d > 0).sum()) if b else None,
                                "p": paired(d) if b else None})
    above = sorted([o for o in ops if o["diff"] is not None and o["diff"] > 0], key=lambda o: -o["diff"])
    lines.append(f"\n## cascades: {len(ops)} operating points (thresholds from inner val), "
                 f"{len(above)} above the single-model front (mean paired diff > 0)")
    hdr = "L>H | signal | rule | test F1 | bit-ops M | esc% iv/test | best single <= cost | diff | wins/n | p"
    fmt = lambda o: (f"{o['L']}>{o['H']} | {o['signal']} | {o['rule']} | {o['f1']:.4f} | {o['bitops'] / 1e6:.1f} | "
                     f"{100 * o['esc_frac_iv']:.0f}/{100 * o['esc_frac_test']:.0f} | {o['best_single']} "
                     f"{o['best_single_f1']:.4f} | {o['diff']:+.4f} | {o['wins']}/{nu} | {o['p']:.3f}"
                     + (f" | match reached on iv {o['match_reached']}/{nu}" if o["rule"] == "match" else ""))
    lines.append("top 12 above front: " + hdr)
    lines += [fmt(o) for o in above[:12]]
    lines.append("best operating point per signal (any L, H, rule): " + hdr)
    for g in SIGS:
        cand = [o for o in ops if o["signal"] == g and o["diff"] is not None]
        lines.append(fmt(max(cand, key=lambda o: o["diff"])))
    lines.append("margin, every L>H at each rule: " + hdr)
    lines += [fmt(o) for o in ops if o["signal"] == "margin"]

    # upper bound (tuned on test): rank test scores, escalate exactly 10/20/30 percent
    ub = []
    for L in CHEAP:
        for H in EXP:
            for g in SIGS:
                for fr in (0.1, 0.2, 0.3):
                    f1s, cs = [], []
                    for u, sc in zip(units, sig_cache[L]):
                        s_te, passes = sc[g][1], sc[g][2]
                        k = int(round(fr * len(s_te)))
                        e = np.zeros(len(s_te), bool); e[np.argsort(-s_te, kind="stable")[:k]] = True
                        f1s.append(mf1(u["y_te"], np.where(e, u[f"l_te_{H}"].argmax(1), u[f"l_te_{L}"].argmax(1)), NC))
                        cs.append(passes * COST[L] + e.mean() * COST[H])
                    b = best_single(np.mean(cs))
                    ub.append({"L": L, "H": H, "signal": g, "frac": fr, "f1": float(np.mean(f1s)),
                               "bitops": float(np.mean(cs)), "best_single": b,
                               "diff": float(np.mean(np.array(f1s) - F1[b]))})
    bu = max(ub, key=lambda o: o["diff"])
    lines.append(f"\nupper bound (tuned on test), best of {len(ub)}: {bu['L']}>{bu['H']} {bu['signal']} "
                 f"{int(100 * bu['frac'])}% F1 {bu['f1']:.4f} bit-ops {bu['bitops'] / 1e6:.1f}M vs {bu['best_single']} "
                 f"diff {bu['diff']:+.4f}; {sum(o['diff'] > 0 for o in ub)} of {len(ub)} above front")

    txt = "\n".join(lines)
    print(txt)
    (a.out / f"{a.name}_idea1.txt").write_text(txt + "\n")
    keys = ["L", "H", "signal", "rule", "f1", "bitops", "esc_frac_iv", "esc_frac_test", "match_reached",
            "best_single", "best_single_f1", "diff", "wins", "p"]
    (a.out / f"{a.name}_idea1_ops.csv").write_text(
        ",".join(keys) + ",n_units\n" + "".join(",".join(str(o[k]) for k in keys) + f",{nu}\n" for o in ops))
    (a.out / f"{a.name}_idea1.json").write_text(json.dumps(
        {"units": [(int(u["seed"]), int(u["fold"])) for u in units], "costs": COST, "sizes": SIZE,
         "singles": {m: {"f1": F1[m].tolist(), "f1_iv": F1iv[m].tolist()} for m in MODELS}, "front": front,
         "temperature": temps, "auroc": auroc, "ops": ops, "upper_bound_test_tuned": ub}, indent=1))


if __name__ == "__main__":
    main()
