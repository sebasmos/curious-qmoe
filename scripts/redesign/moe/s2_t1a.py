"""T1a: do the cheap model's out-of-fold margins and difficulty predict the strong model's errors? ANALYSIS_PLAN.md B8.
Reads <run>/t1/raw/s*_f*.npz. Strong model = E2 with emb-only input (cross-fit heads on training folds, UA head on test folds).
    python s2_t1a.py --run DIR"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); a = ap.parse_args()
d = load_table(); y = d.label.values.astype(np.int64)
def lmargin(z):
    s = np.sort(z, 1); return s[:, -1] - s[:, -2]
rows = []
for f in sorted(glob.glob(str(a.run / "t1/raw/s*_f*.npz"))):
    R = np.load(f); seed, fold = [int(t[1:]) for t in Path(f).stem.split("_")]
    tr, te = R["tr"], R["te"]
    z1, z2 = R["z1_oof"].astype(np.float64), R["z2_oof"].astype(np.float64)
    p1, p2 = softmax(z1), softmax(z2); m1, m2 = pmargin(p1), pmargin(p2); w2 = p2.argmax(1) != y[tr]
    base = dict(seed=seed, fold=fold, split="train", n=len(tr), err_E1=float((p1.argmax(1) != y[tr]).mean()), err_E2=float(w2.mean()))
    rows.append({**base, "spearman": spearmanr(m1, m2)[0], "auroc": roc_auc_score(w2, 1 - m1), "auroc_logit_margin": roc_auc_score(w2, -lmargin(z1)),
                 "auroc_d": roc_auc_score(w2, R["e1_d"]), "auroc_negAUM": roc_auc_score(w2, -R["e1_aum"]), "auroc_el2n": roc_auc_score(w2, R["e1_el2n"]),
                 "auroc_forget": roc_auc_score(w2, R["e1_forget"]), "spearman_d1_d2": spearmanr(R["e1_d"], R["e2_d"])[0]})
    z2t = R["UA_te"].astype(np.float64); m1t, m2t = R["m1_te"], pmargin(softmax(z2t)); w2t = z2t.argmax(1) != y[te]
    z1t = R["z1_te"].astype(np.float64)
    rows.append(dict(seed=seed, fold=fold, split="test", n=len(te), err_E1=float((z1t.argmax(1) != y[te]).mean()), err_E2=float(w2t.mean()),
                     spearman=spearmanr(m1t, m2t)[0], auroc=roc_auc_score(w2t, 1 - m1t), auroc_logit_margin=roc_auc_score(w2t, -lmargin(z1t))))
(a.run / "t1a").mkdir(exist_ok=True)
P = pd.DataFrame(rows).sort_values(["split", "seed", "fold"]); P.to_csv(a.run / "t1a/t1a_per_seed_fold.csv", index=False)
F = P.groupby(["split", "fold"]).mean(numeric_only=True).drop(columns="seed").reset_index(); F.to_csv(a.run / "t1a/t1a_per_fold.csv", index=False)
v = {"n_seeds_folds": {s: int(len(P[P.split == s])) for s in ("train", "test")}}
for s in ("train", "test"):
    g = F[F.split == s]
    ok = (g.spearman >= 0.3) & (g.auroc >= 0.6)
    v[s] = dict(n_folds=int(len(g)), mean_spearman=float(g.spearman.mean()), min_spearman=float(g.spearman.min()), max_spearman=float(g.spearman.max()),
                mean_auroc=float(g.auroc.mean()), min_auroc=float(g.auroc.min()), max_auroc=float(g.auroc.max()),
                folds_spearman_ge_0_3=int((g.spearman >= 0.3).sum()), folds_auroc_ge_0_6=int((g.auroc >= 0.6).sum()), folds_both=int(ok.sum()),
                mean_auroc_logit_margin=float(g.auroc_logit_margin.mean()),
                mean_err_E1=float(g.err_E1.mean()), mean_err_E2=float(g.err_E2.mean()))
    if s == "train":
        v[s].update({k: float(g[k].mean()) for k in ("auroc_d", "auroc_negAUM", "auroc_el2n", "auroc_forget", "spearman_d1_d2")})
n_f = len(F[F.split == "train"])
pass_ = all(v[s]["folds_both"] >= 7 for s in ("train", "test")) and n_f == 10
stop = all(v[s]["mean_spearman"] < 0.15 for s in ("train", "test"))
v["verdict"] = "PASS" if pass_ else ("STOP" if stop else "INCONCLUSIVE")
v["rule"] = "PASS: Spearman >= 0.3 and AUROC >= 0.6 in >= 7 of 10 folds on training folds and on test folds; STOP: mean fold Spearman < 0.15 on both; else INCONCLUSIVE"
write_json(a.run / "t1a/t1a_verdict.json", v)
pd.set_option("display.width", 200); print(F.round(3).to_string()); print(json.dumps(v, indent=1, default=float))
