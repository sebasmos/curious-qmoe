"""Table 1 recomputed at fold level: average seeds inside each fold first, then paired t over folds (df 4 or 9),
and a two one-sided tests (TOST) equivalence check at +-1.0 F1 point. Source: protocol-v3-singles metrics.json."""
import json, numpy as np
from pathlib import Path
from scipy import stats
B = Path("/Users/cajas.sebastian/Documents/qmoe-results/2026-10-20_2127_protocol-v3-singles_code-bc4ff1c/outputs")
for ds, seeds, nf in (("esc", range(5), 5), ("urban", range(3), 10)):
    M = {}
    for k in ["esc", "16", "8", "4", "2", "1", "qesc", "bitnet"]:
        M[k] = np.array([[json.loads((B / f"{ds}_singles_s{s}/{k}/fold_{f}/metrics.json").read_text())["f1_macro"] for s in seeds] for f in range(1, nf + 1)]).mean(1)
    print(f"== {ds}: {nf} folds, seeds averaged inside folds")
    for k in ["16", "8", "4", "2", "1", "qesc", "bitnet"]:
        d = (M[k] - M["esc"]) * 100; n = len(d); se = d.std(ddof=1) / np.sqrt(n); t = d.mean() / se; p = 2 * stats.t.sf(abs(t), n - 1)
        h = stats.t.ppf(0.975, n - 1) * se
        # TOST at +-1.0 point
        t1 = (d.mean() + 1.0) / se; t2 = (d.mean() - 1.0) / se
        ptost = max(stats.t.sf(t1, n - 1), stats.t.cdf(t2, n - 1))
        print(f"  {k:6s} diff {d.mean():+.2f} pts  95% CI [{d.mean()-h:+.2f}, {d.mean()+h:+.2f}]  t p {p:.3f}  TOST(+-1.0) p {ptost:.3f}")
