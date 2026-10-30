"""Redesign idea 3, stage 1: disagreement between precisions as the epistemic signal.

(a) Precision pairs (bitnet/1, bitnet/2, 1/2, 2/4, bitnet/4): both cheap heads run,
    the answer is their averaged softmax. Signals: hard disagreement (ties broken
    by symmetric KL), symmetric KL, and the averaged answer's own negative margin;
    reference: the single-pass margin of the pair's higher-precision member.
    "probe A/B": the same two passes, but the cheaper head A answers and B is run
    only to detect disagreement (the setting where the headroom study saw
    disagreement beat margin); signals: disagree, symKL, A's own margin.
(b) Stochastic-rounding (SR) ensembles of one cheap head (K = 4 or 8 SR weight
    samples, R seeded replicate sets, results averaged over replicates). The answer
    is the K-sample mean. Signals: vote disagreement (1 - modal vote share, ties
    broken by variance), predictive variance (sum over classes), mutual
    information, and the mean's own negative margin.
For both: AUROC for (i) the answer's errors and (ii) answer wrong and H right
(H in 8, 16, esc); macro F1 and ECE (15 equal-width bins on max prob) of the
answer; cascades driven by each signal with thresholds from inner validation
(match H's inner-val F1, or escalate 10/20/30 percent of inner-val clips).
Cost = every cheap pass + escalated fraction on test x bit-ops(H). Front = best
single model (mean test F1 over the same seed x fold units) with bit-ops <= cost;
storage of an SR ensemble is reported two ways: latent FP32 weights sampled at
run time, or K pre-sampled k-bit codes.

usage: python analyze.py --name urban --out DIR EXTRACT_DIR [EXTRACT_DIR ...]   (one dir per seed)
"""
import argparse, json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

MODELS = ["bitnet", "1", "2", "4", "8", "16", "esc", "qesc"]
CHEAP = ["bitnet", "1", "2", "4"]
EXP = ["8", "16", "esc"]
PAIRS = [("bitnet", "1"), ("bitnet", "2"), ("1", "2"), ("2", "4"), ("bitnet", "4")]
KS = [4, 8]
RULES = ["match", "b10", "b20", "b30"]
BUDGET = {"b10": 0.10, "b20": 0.20, "b30": 0.30}
BITS = {"bitnet": 2, "1": 1, "2": 2, "4": 4}
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
    cm = np.bincount(y * nc + p, minlength=nc * nc).reshape(nc, nc)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f[denom > 0].mean())


def softmax(l):
    z = l - l.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def entropy(p):
    return -(p * np.log(np.clip(p, EPS, 1))).sum(-1)


def neg_margin(p):
    s = np.sort(p, 1)
    return -(s[:, -1] - s[:, -2])


def symkl(p, q):
    p, q = np.clip(p, EPS, 1), np.clip(q, EPS, 1)
    return ((p - q) * (np.log(p) - np.log(q))).sum(1)


def ece(y, p, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def threshold(rule, s_iv, y_iv, pl, ph, nc):
    if rule in BUDGET:
        return float(np.quantile(s_iv, 1 - BUDGET[rule])), None
    target = mf1(y_iv, ph, nc)
    u = np.unique(s_iv)
    cands = np.concatenate([[np.inf], ((u[:-1] + u[1:]) / 2)[::-1], [-np.inf]])
    best = None
    for t in cands:
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


def pair_view(u, A, B, part, answer="avg"):
    pa, pb = softmax(u[f"l_{part}_{A}"]), softmax(u[f"l_{part}_{B}"])
    kl = symkl(pa, pb)
    sig = {"disagree": (pa.argmax(1) != pb.argmax(1)) + 1e-3 * np.tanh(kl), "symKL": kl}
    if answer == "A":
        sig["margin_A"] = neg_margin(pa)
        return pa, sig
    p = (pa + pb) / 2
    sig.update({"margin_avg": neg_margin(p), "margin_B": neg_margin(pb)})
    return p, sig


def sr_view(u, L, K, r, part):
    P = u[f"sr_{part}_{L}"][r, :K].astype(np.float64)
    p = P.mean(0)
    votes = np.stack([(P.argmax(2) == c).sum(0) for c in range(P.shape[2])], 1)
    var = P.var(0).sum(1)
    sig = {"vote": 1 - votes.max(1) / K + 1e-3 * np.tanh(var), "var": var,
           "MI": entropy(p) - entropy(P).mean(0), "margin_ens": neg_margin(p)}
    return p, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("dirs", type=Path, nargs="+")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    units, COST, SIZE, NC = load_units(a.dirs)
    nu = len(units)
    R = units[0]["sr_te_4"].shape[0]
    lines = [f"# {a.name}: {nu} seed x fold units, SR replicates R={R}"]

    F1 = {m: np.array([mf1(u["y_te"], u[f"l_te_{m}"].argmax(1), NC) for u in units]) for m in MODELS}
    ECE = {m: np.array([ece(u["y_te"], softmax(u[f"l_te_{m}"])) for u in units]) for m in MODELS}
    lines.append("## singles: model | test F1 | ECE | bit-ops M | size B")
    for m in sorted(MODELS, key=lambda m: COST[m]):
        lines.append(f"{m:6s} | {F1[m].mean():.4f} | {ECE[m].mean():.4f} | {COST[m] / 1e6:.1f} | {SIZE[m]}")

    def best_single(c, size=None):
        ok = [m for m in MODELS if COST[m] <= c * (1 + 1e-9) and (size is None or SIZE[m] <= size * (1 + 1e-9))]
        return max(ok, key=lambda m: F1[m].mean()) if ok else None

    # bytes of one model = overhead + bits x n_hidden_weights / 8; recovered from the 1- and 2-bit sizes
    nw8 = SIZE["2"] - SIZE["1"]  # n_hidden_weights / 8
    overhead = SIZE["1"] - nw8

    # systems: name -> per unit list over replicates of (p_iv, sig_iv, p_te, sig_te); cost; signals
    systems = []
    for A, B in PAIRS:
        views = [[(*pair_view(u, A, B, "iv"), *pair_view(u, A, B, "te"))] for u in units]
        systems.append({"name": f"pair {A}/{B}", "kind": "pair", "views": views, "cost": COST[A] + COST[B],
                        "storage": SIZE[A] + SIZE[B], "ref": B, "sigs": ["disagree", "symKL", "margin_avg", "margin_B"]})
    for A, B in PAIRS:
        views = [[(*pair_view(u, A, B, "iv", "A"), *pair_view(u, A, B, "te", "A"))] for u in units]
        systems.append({"name": f"probe {A}/{B}", "kind": "pair", "views": views, "cost": COST[A] + COST[B],
                        "storage": SIZE[A] + SIZE[B], "ref": A, "sigs": ["disagree", "symKL", "margin_A"]})
    for L in CHEAP:
        for K in KS:
            views = [[(*sr_view(u, L, K, r, "iv"), *sr_view(u, L, K, r, "te")) for r in range(R)] for u in units]
            systems.append({"name": f"SR {L} x{K}", "kind": "sr", "L": L, "K": K, "views": views,
                            "cost": K * COST[L], "storage_latent": SIZE["esc"],
                            "storage_presampled": overhead + K * BITS[L] * nw8, "ref": L,
                            "sigs": ["vote", "var", "MI", "margin_ens"]})

    res = {"systems": [], "ops": []}
    lines.append("\n## answers: system | F1 | ECE | bit-ops M | storage (latent | pre-sampled) B | ref head F1 ECE | "
                 "best single <= bit-ops: diff wins p | best single <= bit-ops and pre-sampled storage: diff")
    for S in systems:
        f1u = np.array([np.mean([mf1(u["y_te"], v[2].argmax(1), NC) for v in vs]) for u, vs in zip(units, S["views"])])
        eu = np.array([np.mean([ece(u["y_te"], v[2]) for v in vs]) for u, vs in zip(units, S["views"])])
        b = best_single(S["cost"])
        d = f1u - F1[b]
        st = S.get("storage_presampled", S.get("storage"))
        b2 = best_single(S["cost"], st)
        d2 = (f1u - F1[b2]).mean() if b2 else float("nan")
        S["f1u"] = f1u
        res["systems"].append({"name": S["name"], "f1": f1u.tolist(), "ece": eu.tolist(), "bitops": S["cost"],
                               "storage": st, "best_single": b, "diff": float(d.mean()), "wins": int((d > 0).sum()),
                               "p": paired(d), "best_single_storage": b2, "diff_storage": float(d2)})
        lat = f"{S['storage_latent']}" if S["kind"] == "sr" else "-"
        lines.append(f"{S['name']:15s} | {f1u.mean():.4f} | {eu.mean():.4f} | {S['cost'] / 1e6:.1f} | {lat} | {st} | "
                     f"{S['ref']} {F1[S['ref']].mean():.4f} {ECE[S['ref']].mean():.4f} | {b} {d.mean():+.4f} "
                     f"{int((d > 0).sum())}/{nu} {paired(d):.3f} | {b2} {d2:+.4f}")

    lines.append("\n## AUROC pooled over units (mean over SR replicates): (i) answer wrong | (ii) answer wrong and H right,"
                 " H = 8 / 16 / esc")
    for S in systems:
        row, rec = [], {}
        for g in S["sigs"]:
            vals = {"wrong": [], "8": [], "16": [], "esc": []}
            for r in range(len(S["views"][0])):
                s = np.concatenate([vs[r][3][g] for vs in S["views"]])
                wrong = np.concatenate([vs[r][2].argmax(1) != u["y_te"] for u, vs in zip(units, S["views"])])
                vals["wrong"].append(roc_auc_score(wrong, s))
                for H in EXP:
                    hr = np.concatenate([u[f"l_te_{H}"].argmax(1) == u["y_te"] for u in units])
                    vals[H].append(roc_auc_score(wrong & hr, s))
            rec[g] = {k: float(np.mean(v)) for k, v in vals.items()}
            row.append(f"{g} {rec[g]['wrong']:.3f}|{rec[g]['8']:.3f}/{rec[g]['16']:.3f}/{rec[g]['esc']:.3f}")
        # reference: the reference single head's own margin on its own errors
        L = S["ref"]
        s = np.concatenate([neg_margin(softmax(u[f"l_te_{L}"])) for u in units])
        wrong = np.concatenate([u[f"l_te_{L}"].argmax(1) != u["y_te"] for u in units])
        rref = {"wrong": roc_auc_score(wrong, s)}
        for H in EXP:
            rref[H] = roc_auc_score(wrong & np.concatenate([u[f"l_te_{H}"].argmax(1) == u["y_te"] for u in units]), s)
        rec["ref_single_margin_own_errors"] = rref
        row.append(f"[{L} alone, margin {rref['wrong']:.3f}|{rref['8']:.3f}/{rref['16']:.3f}/{rref['esc']:.3f}]")
        res.setdefault("auroc", {})[S["name"]] = rec
        lines.append(f"{S['name']:15s} " + "  ".join(row))

    # cascades
    for S in systems:
        for H in EXP:
            for g in S["sigs"]:
                for rule in RULES:
                    f1u, cu, fiv, fte = [], [], [], []
                    for u, vs in zip(units, S["views"]):
                        fr, cr, a1, a2 = [], [], [], []
                        for (p_iv, s_iv, p_te, s_te) in vs:
                            t, _ = threshold(rule, s_iv[g], u["y_iv"], p_iv.argmax(1), u[f"l_iv_{H}"].argmax(1), NC)
                            e = s_te[g] > t
                            fr.append(mf1(u["y_te"], np.where(e, u[f"l_te_{H}"].argmax(1), p_te.argmax(1)), NC))
                            cr.append(S["cost"] + e.mean() * COST[H]); a1.append((s_iv[g] > t).mean()); a2.append(e.mean())
                        f1u.append(np.mean(fr)); cu.append(np.mean(cr)); fiv.append(np.mean(a1)); fte.append(np.mean(a2))
                    f1u, c = np.array(f1u), float(np.mean(cu))
                    b = best_single(c)
                    d = f1u - F1[b]
                    res["ops"].append({"system": S["name"], "H": H, "signal": g, "rule": rule, "f1": float(f1u.mean()),
                                       "f1_units": f1u.tolist(), "bitops": c, "esc_frac_iv": float(np.mean(fiv)),
                                       "esc_frac_test": float(np.mean(fte)), "best_single": b,
                                       "best_single_f1": float(F1[b].mean()), "diff": float(d.mean()),
                                       "wins": int((d > 0).sum()), "p": paired(d)})
    ops = res["ops"]
    above = sorted([o for o in ops if o["diff"] > 0], key=lambda o: -o["diff"])
    fmt = lambda o: (f"{o['system']} > {o['H']} | {o['signal']} | {o['rule']} | {o['f1']:.4f} | {o['bitops'] / 1e6:.1f} | "
                     f"{100 * o['esc_frac_iv']:.0f}/{100 * o['esc_frac_test']:.0f} | {o['best_single']} "
                     f"{o['best_single_f1']:.4f} | {o['diff']:+.4f} | {o['wins']}/{nu} | {o['p']:.3f}")
    hdr = "system > H | signal | rule | test F1 | bit-ops M | esc% iv/test | best single <= cost | diff | wins/n | p"
    lines.append(f"\n## cascades: {len(ops)} operating points, {len(above)} above the single-model front")
    lines.append("top 12 above front: " + hdr)
    lines += [fmt(o) for o in above[:12]]
    lines.append("best operating point per system: " + hdr)
    for S in systems:
        lines.append(fmt(max([o for o in ops if o["system"] == S["name"]], key=lambda o: o["diff"])))
    lines.append("best operating point per signal: " + hdr)
    for g in ["disagree", "symKL", "margin_avg", "margin_B", "margin_A", "vote", "var", "MI", "margin_ens"]:
        lines.append(fmt(max([o for o in ops if o["signal"] == g], key=lambda o: o["diff"])))

    txt = "\n".join(lines)
    print(txt)
    (a.out / f"{a.name}_idea3.txt").write_text(txt + "\n")
    res.update({"units": [(int(u["seed"]), int(u["fold"])) for u in units], "costs": COST, "sizes": SIZE,
                "singles": {m: {"f1": F1[m].tolist(), "ece": ECE[m].tolist()} for m in MODELS}})
    (a.out / f"{a.name}_idea3.json").write_text(json.dumps(res, indent=1))
    keys = ["system", "H", "signal", "rule", "f1", "bitops", "esc_frac_iv", "esc_frac_test", "best_single",
            "best_single_f1", "diff", "wins", "p"]
    (a.out / f"{a.name}_idea3_ops.csv").write_text(
        ",".join(keys) + ",n_units\n" + "".join(",".join(str(o[k]) for k in keys) + f",{nu}\n" for o in ops))


if __name__ == "__main__":
    main()
