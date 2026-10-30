"""Check 2b: is n4a6 above w8a8 on the field mix a head or scaler artefact? (a) seed and fold spread of the T19 heads' scores;
(b) a head-independent probe: standardised multinomial logistic regression (sklearn, C=1) fitted on each configuration's clean
training embeddings of the outer fold, scored on the field-mix test rows. CPU only."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from audit_core import R, S2, TASK, field_assign, mf1, nb
from reproduce_h1_h2 import static_test, folds, ratio

CF = ["clean"] + [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]
res = {}
for task, emb in (("esc50", "esc50"), ("urban", "urban8k")):
    t = TASK[task]
    # (a) T19 heads: fold-level difference n4a6 - w8a8 and per-seed means
    a6, a8 = static_test(task, "n4a6:own", "field"), static_test(task, "w8a8:own", "field")
    res[task] = dict(t19_heads_field=dict(n4a6=a6.mean(), w8a8=a8.mean(), diff=nb(a6 - a8, ratio(task)), folds_n4a6_higher=int((a6 > a8).sum())))
    # (b) probe
    meta = pd.read_csv(t["csv"], usecols=["folder", t["label"]])
    y, fold = meta[t["label"]].to_numpy(), meta["folder"].to_numpy()
    n = len(y)
    at = field_assign(n, 5001)
    probe = {}
    for cfg, d in (("w8a8", R / "emb" / emb), ("n4a6", R / "emb" / emb), ("n4a4", R / "emb" / emb), ("n4a8", R / "emb" / emb),
                   ("w6a6", R / "emb" / emb), ("L:w8g64", S2 / "emb_ast_short" / emb), ("L:fp32", S2 / "emb_ast_short" / emb)):
        stem = cfg.replace("L:", "")
        Xc = np.load(d / f"{stem}__clean.npy")
        Xf = Xc.copy()
        for c in CF[1:]:
            Xn = np.load(d / f"{stem}__{c}.npy", mmap_mode="r")
            m = at == c
            Xf[m] = Xn[m]
        assert np.isfinite(Xf).all() and np.isfinite(Xc).all(), cfg
        fc, ff = [], []
        for k in folds(task):
            tr, te = fold != k, fold == k
            sc = StandardScaler().fit(Xc[tr])
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xc[tr]), y[tr])
            fc.append(mf1(y[te], lr.predict(sc.transform(Xc[te])), t["K"]))
            ff.append(mf1(y[te], lr.predict(sc.transform(Xf[te])), t["K"]))
        probe[cfg] = dict(clean=float(np.mean(fc)), field=float(np.mean(ff)), field_folds=np.round(ff, 3).tolist())
        print(task, cfg, f"probe clean {np.mean(fc):.2f} field {np.mean(ff):.2f}", flush=True)
    res[task]["probe_logreg"] = probe
print(json.dumps(res, indent=1, default=float))
json.dump(res, open("n4a6_probe.json", "w"), indent=1, default=float)
