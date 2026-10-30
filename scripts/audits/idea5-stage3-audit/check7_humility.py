"""Check 7: humility. For the 3-bit cheap passes (n3g64, w3g64), FP32 head, every dataset and
condition:
  - test escalation rate under thresholds set once on clean inner validation (5, 10, 20%)
  - AUROC of the margin gate (low margin = escalate) for the clips the cheap pass gets wrong
    and the W8 parent gets right, pooled over the folds of each seed; Mann-Whitney p per seed
  - ECE (15 equal-width bins, top label) and mean confidence minus accuracy of the b05
    cascade (per-condition threshold) and of the static cheap and parent passes
Also writes the per-pass break-even group-scale price (check 4).

    python check7_humility.py --stage2 STAGE2_DIR --out OUT
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute import Run, margin, parent, thr_rate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage2", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return sum((b == k).mean() * abs((pred[b == k] == y[b == k]).mean() - conf[b == k].mean())
               for k in range(bins) if (b == k).any())


rows = []
for ds in ["esc50", "esc5", "urban"]:
    sc = a.stage2 / "scores" / ds
    tags = sorted(p.name for p in sc.iterdir() if p.is_dir() and "headw4" not in p.name)
    runs = {int(t.split("_s")[1]): Run(sc / t / "esc") for t in tags}
    conds = pd.read_csv(a.out / "bitops" / ds / "points.csv").cond.unique()
    for ch in ["n3g64", "w3g64"]:
        par = parent(ch)
        for cond in conds:
            rec = dict(dataset=ds, cheap=ch, cond=cond)
            rates = {5: [], 10: [], 20: []}
            aucs, pmw, e_c, e_s, e_p, g_c = [], [], [], [], [], []
            for seed, run in runs.items():
                rc, rp, rcl = run.get(ch, cond), run.get(par, cond), run.get(ch, "clean")
                y = rc["y"]
                pc, pp = rc["p"], rp["p"]
                tgt = (pc.argmax(1) != y) & (pp.argmax(1) == y)
                s = -margin(pc)
                aucs.append(roc_auc_score(tgt, s))
                pmw.append(stats.mannwhitneyu(s[tgt], s[~tgt], alternative="greater").pvalue)
                for fold in sorted(rc["iv"]):
                    te = rc["tf"] == fold
                    for r in rates:
                        t = thr_rate(margin(rcl["iv"][fold][1]), r / 100)
                        rates[r].append((margin(pc[te]) < t).mean())
                    t5 = thr_rate(margin(rc["iv"][fold][1]), 0.05)
                    e = margin(pc[te]) < t5
                    pr = np.where(e[:, None], pp[te], pc[te])
                    e_c.append(ece(pr, y[te]))
                    g_c.append(pr.max(1).mean() - (pr.argmax(1) == y[te]).mean())
                    e_s.append(ece(pc[te], y[te]))
                    e_p.append(ece(pp[te], y[te]))
            for r in rates:
                rec[f"esc_rate_clean_b{r:02d}"] = float(np.mean(rates[r]))
            rec.update(auroc_mean=float(np.mean(aucs)), auroc_min_seed=float(np.min(aucs)),
                       mw_p_max_seed=float(np.max(pmw)), ece_cascade_b05=float(np.mean(e_c)),
                       conf_minus_acc_cascade_b05=float(np.mean(g_c)), ece_cheap=float(np.mean(e_s)),
                       ece_parent=float(np.mean(e_p)))
            rows.append(rec)
        print(ds, ch, "done", flush=True)
H = pd.DataFrame(rows)
H.to_csv(a.out / "check7_humility.csv", index=False)
pd.set_option("display.width", 250)
print(H.to_string(float_format=lambda v: f"{v:.3g}"))
