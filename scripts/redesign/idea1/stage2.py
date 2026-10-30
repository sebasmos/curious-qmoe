"""Redesign idea 1, stage 2: two margin cascades at fixed operating points, scored
with scripts/redesign/front.py on protocol 3 single models.

Cascades (fixed in advance):
  c1to8     the 1-bit head answers; clips whose softmax margin (top-1 minus top-2
            probability) is below a threshold are re-run on the 8-bit head
  avg12to8  the mean of the 1-bit and 2-bit heads' softmax answers; escalation
            to the 8-bit head on the margin of that mean
Thresholds per (seed, fold) from the fold's inner validation split (split.json),
never from test:
  b05 b10 b20 b30  the inner-val margin quantile that escalates 5/10/20/30
                   percent of inner-val clips
  match8           fewest inner-val escalations whose cascade macro F1 reaches
                   the 8-bit head's inner-val macro F1 (100 percent always does)
Cost per clip (QWave.costs, from stage2_extract.py): the cheap pass(es) on every
clip plus the 8-bit pass on the escalated share of test clips. Weight bytes: every
head the cascade stores. Escalated clips report the 8-bit head's softmax, the rest
the cheap answer's; ECE uses 15 equal-width bins of the top probability.

Writes per (seed, fold) records for front.py (singles with is_single true, both
cascades at each operating point), runs front.py on them (one Bonferroni family
over every dataset given), and a table per dataset.

usage: python stage2.py --out DIR esc50=EXDIR[,EXDIR...] quinn=... urban=...
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import mf1, softmax

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.redesign.front import main as front_main  # noqa: E402

CASCADES = {"c1to8": ["1"], "avg12to8": ["1", "2"]}
RULES = ["b05", "b10", "b20", "b30", "match8"]
BUDGET = {"b05": 0.05, "b10": 0.10, "b20": 0.20, "b30": 0.30}
ECE_BINS = 15


def load_units(dirs):
    units, costs, wb, models = [], None, None, None
    for d in dirs:
        chk = json.loads((d / "checks.json").read_text())
        c = {m: float(v[1]) for m, v in chk["costs"].items()}
        assert costs is None or (c == costs and chk["weight_bytes"] == wb), d
        costs, wb, nc = c, chk["weight_bytes"], chk["n_classes"]
        models = chk["models"] if models is None else [m for m in models if m in chk["models"]]
        metrics_f1 = {(k["fold"], k["model"]): k["metrics_f1"] for k in chk["checks"]}
        for k in chk["folds"]:
            u = dict(np.load(d / f"fold_{k}.npz"))
            u["seed"], u["fold"], u["metrics_f1"] = int(chk["seed"]), int(k), {
                m: metrics_f1[(k, m)] for m in chk["models"]}
            units.append(u)
    return units, costs, wb, models, nc


def margin_score(p):
    s = np.sort(p, 1)
    return -(s[:, -1] - s[:, -2])  # higher = less sure


def ece(p, y):
    conf, acc = p.max(1), (p.argmax(1) == y).astype(float)
    b = np.clip(np.searchsorted(np.linspace(0, 1, ECE_BINS + 1), conf, side="left") - 1, 0, ECE_BINS - 1)
    return float(sum(abs(acc[b == i].mean() - conf[b == i].mean()) * (b == i).mean()
                     for i in range(ECE_BINS) if (b == i).any()))


def threshold(rule, s_iv, y_iv, pl, ph, nc):
    """Escalate clips with score > t."""
    if rule in BUDGET:
        return float(np.quantile(s_iv, 1 - BUDGET[rule]))
    target = mf1(y_iv, ph, nc)
    u = np.unique(s_iv)
    for t in np.concatenate([[np.inf], ((u[:-1] + u[1:]) / 2)[::-1], [-np.inf]]):  # fewest escalations first
        if mf1(y_iv, np.where(s_iv > t, ph, pl), nc) >= target - 1e-12:
            return float(t)
    raise AssertionError("escalating every clip reproduces the 8-bit head")


def run_dataset(name, units, COST, WB, models, NC):
    recs, ops = [], []
    singles = {}
    for m in models:
        f1 = [mf1(u["y_te"], u[f"l_te_{m}"].argmax(1), NC) for u in units]
        for u, f in zip(units, f1):
            assert abs(f - u["metrics_f1"][m]) < 1e-9, (name, u["seed"], u["fold"], m)
            recs.append({"method": m, "dataset": name, "seed": u["seed"], "fold": u["fold"], "f1_macro": f,
                         "bitops_per_clip": COST[m], "weight_bytes": WB[m], "operating_point": "single",
                         "is_single": True})
        singles[m] = {"f1": f1, "f1_iv": [mf1(u["y_iv"], u[f"l_iv_{m}"].argmax(1), NC) for u in units],
                      "ece": [ece(softmax(u[f"l_te_{m}"]), u["y_te"]) for u in units]}
    avg12 = {"f1": [], "ece": []}
    for cname, cheap in CASCADES.items():
        cheap_cost, wbytes = sum(COST[m] for m in cheap), sum(WB[m] for m in cheap) + WB["8"]
        for rule in RULES:
            o = {k: [] for k in ("f1", "bitops", "esc_iv", "esc_te", "agree8", "ece", "ece_cheap")}
            for u in units:
                p = {part: np.mean([softmax(u[f"l_{part}_{m}"]) for m in cheap], 0) for part in ("iv", "te")}
                p8 = {part: softmax(u[f"l_{part}_8"]) for part in ("iv", "te")}
                s_iv, s_te = margin_score(p["iv"]), margin_score(p["te"])
                t = threshold(rule, s_iv, u["y_iv"], p["iv"].argmax(1), p8["iv"].argmax(1), NC)
                e = s_te > t
                pred = np.where(e, p8["te"].argmax(1), p["te"].argmax(1))
                o["f1"].append(mf1(u["y_te"], pred, NC))
                o["bitops"].append(cheap_cost + e.mean() * COST["8"])
                o["esc_iv"].append(float((s_iv > t).mean()))
                o["esc_te"].append(float(e.mean()))
                o["agree8"].append(float((pred == p8["te"].argmax(1)).mean()))
                o["ece"].append(ece(np.where(e[:, None], p8["te"], p["te"]), u["y_te"]))
                o["ece_cheap"].append(ece(p["te"], u["y_te"]))
                if cname == "avg12to8" and rule == RULES[0]:
                    avg12["f1"].append(mf1(u["y_te"], p["te"].argmax(1), NC))
                    avg12["ece"].append(o["ece_cheap"][-1])
                recs.append({"method": cname, "dataset": name, "seed": u["seed"], "fold": u["fold"],
                             "f1_macro": o["f1"][-1], "bitops_per_clip": o["bitops"][-1], "weight_bytes": wbytes,
                             "operating_point": rule, "is_single": False})
            ops.append({"cascade": cname, "rule": rule, "weight_bytes": wbytes, **o})
    return recs, singles, ops, avg12


def table(name, units, COST, WB, models, singles, ops, avg12, comp):
    n = len(units)
    ece8 = np.mean(singles["8"]["ece"])
    L = [f"## {name}: {n} (seed, fold) pairs, seeds {sorted(set(u['seed'] for u in units))}",
         "singles: model | test F1 mean (sd) | inner-val F1 | bit-ops M | weight KB | ECE"]
    for m in sorted(models, key=lambda m: COST[m]):
        s = singles[m]
        L.append(f"{m:6s} | {np.mean(s['f1']):.4f} ({np.std(s['f1'], ddof=1):.4f}) | {np.mean(s['f1_iv']):.4f} | "
                 f"{COST[m] / 1e6:.2f} | {WB[m] / 1e3:.1f} | {np.mean(s['ece']):.4f}")
    L.append(f"avg(1,2) alone (no escalation): F1 {np.mean(avg12['f1']):.4f}, ECE {np.mean(avg12['ece']):.4f}, "
             f"bit-ops {(COST['1'] + COST['2']) / 1e6:.2f}M")
    L.append("cascade | rule | esc% inner val / test (test min-max) | test F1 | bit-ops M | x 8-bit | weight KB | "
             "8-bit agreement mean (min) | ECE cascade / cheap / 8-bit | comparator F1 | diff | "
             "p_t / p_w Bonferroni | verdict")
    c = comp[comp["dataset"] == name].set_index(["method", "operating_point"])
    for o in ops:
        r = c.loc[(o["cascade"], o["rule"])]
        L.append(f"{o['cascade']} | {o['rule']} | {100 * np.mean(o['esc_iv']):.1f} / {100 * np.mean(o['esc_te']):.1f} "
                 f"({100 * np.min(o['esc_te']):.0f}-{100 * np.max(o['esc_te']):.0f}) | {np.mean(o['f1']):.4f} | "
                 f"{np.mean(o['bitops']) / 1e6:.2f} | {np.mean(o['bitops']) / COST['8']:.2f} | "
                 f"{o['weight_bytes'] / 1e3:.1f} | {100 * np.mean(o['agree8']):.1f} ({100 * np.min(o['agree8']):.1f}) | "
                 f"{np.mean(o['ece']):.4f} / {np.mean(o['ece_cheap']):.4f} / {ece8:.4f} | "
                 f"{r['comparator_method']} {r['comparator_f1_macro']:.4f} | {r['mean_diff']:+.4f} | "
                 f"{r['p_ttest_bonferroni']:.3g} / {r['p_wilcoxon_bonferroni']:.3g} | {r['verdict']}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("datasets", nargs="+", help="name=EXTRACT_DIR[,EXTRACT_DIR...] (one dir per seed)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data, recs = {}, []
    for kv in a.datasets:
        name, dirs = kv.split("=", 1)
        units, COST, WB, models, NC = load_units([Path(d) for d in dirs.split(",")])
        r, singles, ops, avg12 = run_dataset(name, units, COST, WB, models, NC)
        recs += r
        data[name] = (units, COST, WB, models, singles, ops, avg12)
    rec_path = a.out / "records.jsonl"
    rec_path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    verdict = front_main([str(rec_path), "--out", str(a.out / "front")])
    comp = pd.read_csv(a.out / "front" / "comparisons.csv")

    lines = [f"# idea 1 stage 2: front.py family of {verdict['family_size']} comparisons over {list(data)}", ""]
    for name, (units, COST, WB, models, singles, ops, avg12) in data.items():
        lines += table(name, units, COST, WB, models, singles, ops, avg12, comp) + [""]
        (a.out / f"{name}_stage2.json").write_text(json.dumps(
            {"units": [(u["seed"], u["fold"]) for u in units], "costs": COST, "weight_bytes": WB,
             "singles": singles, "avg12": avg12, "ops": ops}, indent=1))
    lines.append("overall (PASS only when PASS on every dataset):")
    for k, v in verdict["points"].items():
        lines.append(f"{k}: {v['verdict']}  {v['per_dataset']}")
    txt = "\n".join(lines)
    print(txt)
    (a.out / "stage2_table.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
