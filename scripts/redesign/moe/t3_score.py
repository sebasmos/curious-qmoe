"""T3 step 3 (ANALYSIS_PLAN_T3T5.md section 3): per (task, condition, fit, seed, fold) AUROCs of the cheap margin, the learned router (MLP on the cheap embedding), Jensen-Shannon
disagreement and a regression on class-agnostic features, for the targets A (AST wrong), B (cheap wrong and AST right) and C (counterfactual gain). Reads test labels.
Run after the plan, the input check, the pre-selection and this script were committed.   python t3_score.py --out DIR [--tasks esc50 urban tau] [--shuffle-smoke]
--shuffle-smoke permutes the three targets over rows before any fit (code-path and timing check only; the output is thrown away and says nothing about the data)."""
import argparse, time, warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t3t5_common import *
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--tasks", nargs="+", default=T35); ap.add_argument("--shuffle-smoke", action="store_true")
ap.add_argument("--max-folds", type=int, default=0); ap.add_argument("--seeds", nargs="*", type=int); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
EPS = 1e-12
def auroc(pos, score):
    pos = np.asarray(pos).astype(bool); n1 = int(pos.sum()); n0 = len(pos) - n1
    if n1 < 2 or n0 < 2: return np.nan
    r = rankdata(score); return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def ent(p): return -(p * np.log(p + EPS)).sum(1)
FEAT = ["margin", "entropy", "emb_norm", "emb_var", "floor_db", "level_db", "flatness_db"]
def assemble(task, cells, seed):
    """Stacked rows of the cells for one seed: features, targets, fold, embedding block."""
    P = dict(fold=[], A=[], B=[], C=[], F=[], js=[], m=[], E=[], cell=[])
    for ci, cell in enumerate(cells):
        Dc, Ds = cellload(task, CHEAP, cell, seed), cellload(task, PAIR[task], cell, seed)
        rows = np.flatnonzero(Dc["device"] == "a") if (task == "tau" and cell == "clean") else (np.flatnonzero(Dc["device"] != "a") if task == "tau" else np.arange(len(Dc["y"])))
        y = Dc["y"][rows]; pc, ps = softmax(Dc["z"][rows]), softmax(Ds["z"][rows]); pm = 0.5 * (pc + ps); r = np.arange(len(y))
        wa, wc = ps.argmax(1) != y, pc.argmax(1) != y; ce = lambda p: -np.log(p[r, y] + EPS)
        X = emb(task, cell)[rows]; S = stats(task, cell)[rows]; m = pmargin(pc)
        F = np.column_stack([m, ent(pc), np.linalg.norm(X, axis=1), X.var(1), S[:, STAT_NAMES.index("floor_db")], S[:, STAT_NAMES.index("level_db")], S[:, STAT_NAMES.index("flatness_db")]])
        for k, v in dict(fold=Dc["fold"][rows], A=wa.astype(int), B=(wc & ~wa).astype(int), C=ce(pc) - ce(pm), F=F, js=js(pc, ps), m=m, E=X, cell=np.full(len(y), ci)).items(): P[k].append(v)
    P = {k: np.concatenate(v) for k, v in P.items()}
    if a.shuffle_smoke:
        g = np.random.default_rng(0); pi = g.permutation(len(P["A"]))
        for k in ("A", "B", "C"): P[k] = P[k][pi]
    return P
def mlp_c(Xtr, ytr, Xte, seed, hid, alpha):
    if ytr.sum() < 10 or (1 - ytr).sum() < 10: return None
    sc = StandardScaler().fit(Xtr); m = MLPClassifier(hidden_layer_sizes=(hid,), activation="relu", solver="adam", alpha=alpha, batch_size=256, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=100, random_state=1000 + seed)
    m.fit(sc.transform(Xtr), ytr); return m.predict_proba(sc.transform(Xte))[:, 1]
def mlp_r(Xtr, ytr, Xte, seed, hid, alpha):
    sc = StandardScaler().fit(Xtr); m = MLPRegressor(hidden_layer_sizes=(hid,), activation="relu", solver="adam", alpha=alpha, batch_size=256, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=100, random_state=1000 + seed)
    m.fit(sc.transform(Xtr), ytr); return m.predict(sc.transform(Xte))
def logit(Xtr, ytr, Xte):
    if ytr.sum() < 10 or (1 - ytr).sum() < 10: return None
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr); return m.decision_function(sc.transform(Xte))
def ridge(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr); return m.predict(sc.transform(Xte))
def met(tag, tgt, score, out):
    if score is None: out.update({f"{tag}_{k}": np.nan for k in tgt}); return
    for k, v in tgt.items(): out[f"{tag}_{k}"] = v(score)
rows_out = []
with threadpool_limits(2):
    for task in a.tasks:
        T = TASKS[task]; cond_sets = {"pool": POOL[task], "clean": ["clean"]}; t0 = time.time()
        for seed in (a.seeds if a.seeds is not None else T["seeds"]):
            D = {c: assemble(task, cells, seed) for c, cells in cond_sets.items()}
            for cond in ("pool", "clean"):
                P = D[cond]; folds = sorted(set(P["fold"].tolist()))
                for fi, f in enumerate(folds):
                    if a.max_folds and fi >= a.max_folds: break
                    te, tr = P["fold"] == f, P["fold"] != f
                    for fit in (("cf", "transfer") if cond == "pool" else ("cf",)):
                        if fit == "cf": Xtr_E, Xtr_F, ytr = P["E"][tr], P["F"][tr], {k: P[k][tr] for k in "ABC"}
                        else:
                            Q_ = D["clean"]; trc = Q_["fold"] != f; Xtr_E, Xtr_F, ytr = Q_["E"][trc], Q_["F"][trc], {k: Q_[k][trc] for k in "ABC"}
                        Ete, Fte = P["E"][te], P["F"][te]; tg = {k: P[k][te] for k in "ABC"}
                        r = dict(task=task, cond=cond, fit=fit, seed=seed, fold=f, n=int(te.sum()), n_train=int(len(ytr["A"])), rate_A=float(tg["A"].mean()), rate_B=float(tg["B"].mean()), pos_C=float((tg["C"] > 0).mean()))
                        bin_t = lambda y: {"auroc": (lambda s: auroc(y, s))}
                        for T_ in "AB":
                            y = tg[T_]; f_ = lambda s, y=y: auroc(y, s)
                            if fit == "cf":
                                r[f"{T_}_margin"] = f_(-P["m"][te]); r[f"{T_}_js"] = f_(P["js"][te]); r[f"{T_}_entropy"] = f_(Fte[:, 1])
                            else: r[f"{T_}_margin"] = f_(-P["m"][te]); r[f"{T_}_js"] = f_(P["js"][te]); r[f"{T_}_entropy"] = f_(Fte[:, 1])
                            s = mlp_c(Xtr_E, ytr[T_], Ete, seed, 256, 1e-4) if (T_ == "A" and fit == "cf") or T_ == "B" else None
                            if (T_ == "A" and fit == "cf") or T_ == "B": r[f"{T_}_r1"] = f_(s) if s is not None else np.nan
                            if T_ == "B" and fit == "cf":
                                s2 = mlp_c(Xtr_E, ytr["B"], Ete, seed, 64, 1e-1); r["B_r2"] = f_(s2) if s2 is not None else np.nan
                            s3 = logit(Xtr_F, ytr[T_], Fte); r[f"{T_}_reg"] = f_(s3) if s3 is not None else np.nan
                        C, pos = tg["C"], tg["C"] > 0
                        sp = lambda s: float(spearmanr(s, C)[0]); ap_ = lambda s: auroc(pos, s)
                        sc_ = {"margin": -P["m"][te], "js": P["js"][te]}
                        if fit == "cf": sc_["r1"] = mlp_r(Xtr_E, ytr["C"], Ete, seed, 256, 1e-4)
                        sc_["reg"] = ridge(Xtr_F, ytr["C"], Fte)
                        for k, s in sc_.items(): r[f"C_spearman_{k}"] = sp(s); r[f"C_auroc_{k}"] = ap_(s)
                        rows_out.append(r)
                    print(task, cond, "seed", seed, "fold", f, "%.0fs" % (time.time() - t0), flush=True)
        pd.DataFrame(rows_out).to_csv(a.out / "t3_fold.csv", index=False)
pd.DataFrame(rows_out).to_csv(a.out / "t3_fold.csv", index=False); print("rows", len(rows_out))
