import json, sys, numpy as np
from pathlib import Path
from scipy import stats
B = Path(sys.argv[1])
for ds, seeds, nf in (("esc", range(5), 5), ("urban", range(3), 10)):
    M, NT, NTR = {}, {}, {}
    for k in ["esc", "16", "8", "4", "2", "1", "bitnet"]:
        rows = [[json.loads((B / f"{ds}_singles_s{s}/{k}/fold_{f}/metrics.json").read_text()) for s in seeds] for f in range(1, nf + 1)]
        M[k] = np.array([[r["f1_macro"] for r in fr] for fr in rows]).mean(1)
        NT[k] = np.array([[r["n_test"] for r in fr] for fr in rows])
        NTR[k] = np.array([[r["n_fit"] + r["n_inner_val"] for r in fr] for fr in rows])
    nt, ntr = NT["esc"][:, 0], NTR["esc"][:, 0]
    ratio = float(np.mean(nt / ntr))
    for k in ["16", "8", "4", "2", "1", "bitnet"]:
        d = (M[k] - M["esc"]) * 100; K = len(d); m = d.mean(); v = d.var(ddof=1)
        out=[]
        for name, mult in (("plain", 1 / K), ("NB", 1 / K + ratio)):
            se = np.sqrt(mult * v)
            p = max(stats.t.sf((m + 1.0) / se, K - 1), stats.t.cdf((m - 1.0) / se, K - 1))
            out.append(f"{name} {p:.4f}")
        print(ds, k, round(m,3), *out)
