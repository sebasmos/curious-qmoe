"""T15/T18 precondition (ANALYSIS_PLAN_T15.md section 5): clean inner-validation macro-F1 of the fold-1 heads of each view against the stored MN10-Q8 heads on the same
clips. Reads labels of inner-validation rows only (no test row is scored). Writes precond/precond_views.csv.
   python t15_precond_heads.py --r R --views v1 v2 ... --datasets urban esc50 --seeds 0 1 2"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2"))
from omegaconf import OmegaConf
from score_eff import fold_scaler, logits_of, train_frame
from scripts.analysis.cascade import load_model
from t13_common import sdir
from t15_views import DIM
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--views", nargs="+", required=True)
ap.add_argument("--datasets", nargs="+", default=["urban", "esc50"]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--fold", type=int, default=1)
ap.add_argument("--outs", default="outputs")  # outputs_precond_fold1 for the stored T15 run
a = ap.parse_args(); torch.set_num_threads(2)
def f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k); tp = np.diag(cm).astype(float); den = cm.sum(0) + cm.sum(1); ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))
K = {"urban": 10, "esc50": 50}; rows = []
for ds in a.datasets:
    for seed in a.seeds:
        d, cfgname = sdir(ds, "mn10", seed); ref = np.load(d / f"{cfgname}__clean.npz")
        pad_d = {"urban": "S3B", "esc50": "S3B"}
        from common import S3B
        padz = np.load(S3B / f"scores_cnn/{'urban' if ds == 'urban' else 'esc50'}/mn10_{'urban' if ds == 'urban' else 'esc50'}_s{seed}/esc/mn10_w8a8__clean.npz")
        iv = ref[f"iv_idx_{a.fold}"]; y = ref["y"].astype(int)
        assert np.array_equal(iv, padz[f"iv_idx_{a.fold}"])
        r = dict(dataset=ds, seed=seed, fold=a.fold, n_iv=len(iv), mn10_q8_f1=f1(y[iv], ref[f"iv_logits_{a.fold}"].argmax(1), K[ds]), mn10_q8_padded_f1=f1(y[iv], padz[f"iv_logits_{a.fold}"].argmax(1), K[ds]))
        rows.append(dict(r, view="mn10_q8", f1=r["mn10_q8_f1"]))
        for v in a.views:
            run = a.r / f"{a.outs}/{v}_{ds}_s{seed}/esc"
            if not (run / "summary.json").exists(): continue
            cfg = OmegaConf.load(run / "config.yaml"); Xc, yc = train_frame(cfg, DIM[v]); n_cls = len(np.unique(yc))
            split = json.loads((run / f"fold_{a.fold}" / "split.json").read_text()); assert np.array_equal(np.sort(np.asarray(split["inner_val_idx"])), np.sort(iv)), (ds, v, seed, len(iv))
            model, _ = load_model(run, a.fold, Xc.shape[1], n_cls); scaler = fold_scaler(Xc, yc, split, cfg.experiment.datasets.esc.normalization_type)
            Xq = np.load(a.r / f"emb/{ds}/{v}__clean.npy")  # the deployed W8A8 embedding, as score_eff applies the head
            lg = logits_of(model, scaler, Xq[iv], n_cls)
            rows.append(dict(r, view=v, f1=f1(yc[iv], lg.argmax(1), n_cls), acc=float((lg.argmax(1) == yc[iv]).mean())))
D = pd.DataFrame(rows); (a.r / "precond").mkdir(exist_ok=True); D.to_csv(a.r / "precond/precond_views_by_seed.csv", index=False)
g = D.groupby(["dataset", "view"]).agg(f1=("f1", "mean"), n_seeds=("f1", "size"), mn10_q8=("mn10_q8_f1", "mean"), mn10_q8_padded=("mn10_q8_padded_f1", "mean")).reset_index()
g["gap_to_mn10_q8_points"] = 100 * (g.f1 - g.mn10_q8); g["gap_to_padded_mn10_points"] = 100 * (g.f1 - g.mn10_q8_padded)
g["within_5_points"] = g.gap_to_mn10_q8_points >= -5.0
g.to_csv(a.r / "precond/precond_views.csv", index=False); print(g.to_string())
