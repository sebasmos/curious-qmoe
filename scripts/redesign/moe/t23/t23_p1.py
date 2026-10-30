"""T23 P1 (ANALYSIS_PLAN.md C8): fold-1 clean inner-validation macro F1 of the deployed W8A8 waveform expert against the stored MN10-Q8 on the
same rows, and the error correlation phi of expert vs MN10-Q8 and of the stored AST-Q4 vs MN10-Q8. Reads inner-validation rows only.
    python t23_p1.py --r R [--outs outputs_p1] [--fold 1]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
for p_ in (ROOT, HERE.parent, ROOT / "scripts/redesign/idea6/stage2"): sys.path.insert(0, str(p_))
from omegaconf import OmegaConf
from score_eff import fold_scaler, logits_of, train_frame
from scripts.analysis.cascade import load_model
from t13_common import sdir, tau_file
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--outs", default="outputs_p1"); ap.add_argument("--fold", type=int, default=1)
ap.add_argument("--tag", default="frozen"); ap.add_argument("--ft", type=Path, default=None, help="fallback: per-seed embeddings R/ft/<task>/f<fold>_s<seed>/wg__clean.npy")
ap.add_argument("--seeds", nargs="+", type=int, default=None); a = ap.parse_args(); torch.set_num_threads(2)
SEEDS = {"esc50": [0, 1, 2, 3, 4], "urban": [0, 1, 2], "tau": [0, 1, 2]}; K = {"esc50": 50, "urban": 10, "tau": 10}
def f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k); tp = np.diag(cm).astype(float); den = cm.sum(0) + cm.sum(1); ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))
def phi(e1, e2):
    e1, e2 = e1.astype(float), e2.astype(float)
    return float(np.corrcoef(e1, e2)[0, 1]) if e1.std() > 0 and e2.std() > 0 else np.nan
def stored(ds, expert, seed):
    if ds == "tau": return np.load(tau_file("mn10" if expert == "mn10" else "astw4", seed))
    d, cfg = sdir(ds, expert, seed); return np.load(d / f"{cfg}__clean.npz")
rows = []
for ds in ("esc50", "urban", "tau"):
    for seed in (a.seeds or SEEDS[ds]):
        run = a.r / f"{a.outs}/wg_{ds}_s{seed}/esc"
        if not (run / "summary.json").exists(): print("missing", run); continue
        mn = stored(ds, "mn10", seed); ast = stored(ds, "astshort" if ds != "tau" else "astw4", seed)
        iv = mn[f"iv_idx_{a.fold}"].astype(int); y = mn["y"].astype(int)
        if not np.array_equal(iv, ast[f"iv_idx_{a.fold}"]): raise SystemExit(f"{ds} s{seed}: stored MN10 and AST inner validation differ")
        cfg = OmegaConf.load(run / "config.yaml"); Xc, yc = train_frame(cfg, 2048); n_cls = len(np.unique(yc))
        split = json.loads((run / f"fold_{a.fold}" / "split.json").read_text()); ours = np.asarray(split["inner_val_idx"])
        same_order = bool(np.array_equal(ours, iv)); same_set = bool(np.array_equal(np.sort(ours), np.sort(iv)))
        if not same_set: raise SystemExit(f"{ds} s{seed}: our inner validation differs from the stored one ({len(ours)} vs {len(iv)})")
        if not np.array_equal(yc, y[:len(yc)]): raise SystemExit(f"{ds}: labels differ")
        model, _ = load_model(run, a.fold, Xc.shape[1], n_cls); scaler = fold_scaler(Xc, yc, split, cfg.experiment.datasets.esc.normalization_type)
        Xq = np.load(a.r / f"emb/{ds}/wg__clean.npy") if a.ft is None else np.load(a.ft / f"{ds}/f{a.fold}_s{seed}/wg__clean.npy")
        pw = logits_of(model, scaler, Xq[iv], n_cls).argmax(1); pm = mn[f"iv_logits_{a.fold}"].argmax(1); pa = ast[f"iv_logits_{a.fold}"].argmax(1); yt = y[iv]
        rows.append(dict(task=ds, seed=seed, fold=a.fold, n_iv=len(iv), iv_same_order=same_order, f1_wave=100 * f1(yt, pw, K[ds]), f1_mn10=100 * f1(yt, pm, K[ds]),
                         f1_ast=100 * f1(yt, pa, K[ds]), phi_wave_mn10=phi(pw != yt, pm != yt), phi_ast_mn10=phi(pa != yt, pm != yt), phi_wave_ast=phi(pw != yt, pa != yt)))
D = pd.DataFrame(rows); out = a.r / "tables"; out.mkdir(exist_ok=True); D.to_csv(out / f"p1_by_seed_{a.tag}.csv", index=False)
g = D.groupby("task").mean(numeric_only=True).drop(columns=["seed", "fold"]); g["n_seeds"] = D.groupby("task").size()
g["gap_points"] = g.f1_wave - g.f1_mn10; g["within_10"] = g.gap_points >= -10.0; g["phi_below_ast"] = g.phi_wave_mn10 < g.phi_ast_mn10
g["task_pass"] = g.within_10 & g.phi_below_ast; g.to_csv(out / f"p1_{a.tag}.csv")
v = {"rule": "within 10 points of MN10-Q8 on clean inner validation (fold 1) on two of three tasks, and phi with MN10-Q8 below the AST-Q4's on those tasks",
     "tasks_passing": [t for t in g.index if g.loc[t, "task_pass"]], "n_tasks_scored": int(len(g)), "P1_pass": bool(g.task_pass.sum() >= 2)}
(out / f"p1_verdict_{a.tag}.json").write_text(json.dumps(v, indent=1)); print(g.round(3).to_string()); print(v)
