"""T23 amendment 1: after the protocol 3 head of one (fold, seed) is trained (scripts/benchmark.py, only_folds=[f]) on the fine-tuned tail's
FP32 clean embeddings, fit the fold's scaler as score_eff.py does (fit rows = CSV minus test and inner validation), re-run the head on the
CSV's fold-f test rows and require its macro F1 to equal summary.json to 1e-6. Then save the scaler (scaler.pkl) and check.npz (test rows,
their FP32 features, the summary F1) in the run directory, so t23_ft_score.py can repeat the check after the CSV is deleted.
    python t23_headcheck.py --run R/outputs_ft/wg_esc50_s0_f2/esc --fold 2"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
for p_ in (ROOT, ROOT / "scripts/redesign/idea6/stage2"): sys.path.insert(0, str(p_))
from omegaconf import OmegaConf  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from score_eff import fold_scaler, logits_of, train_frame  # noqa: E402
from scripts.analysis.cascade import load_model  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); ap.add_argument("--fold", type=int, required=True); a = ap.parse_args()
torch.set_num_threads(2); run = a.run.resolve(); cfg = OmegaConf.load(run / "config.yaml")
Xc, y = train_frame(cfg, 2048); n_cls = len(np.unique(y))
summ = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
if sorted(summ) != [a.fold]: raise SystemExit(f"{run}: folds {sorted(summ)} != [{a.fold}]")
split = json.loads((run / f"fold_{a.fold}" / "split.json").read_text()); te = np.asarray(split["test_idx"])
model, _ = load_model(run, a.fold, 2048, n_cls); scaler = fold_scaler(Xc, y, split, cfg.experiment.datasets.esc.normalization_type)
f1 = f1_score(y[te], logits_of(model, scaler, Xc[te], n_cls).argmax(1), average="macro", zero_division=0)
if abs(f1 - summ[a.fold]["f1_macro"]) > 1e-6: raise SystemExit(f"{run}: rescored F1 {f1:.6f} != summary {summ[a.fold]['f1_macro']:.6f}")
with open(run / "scaler.pkl", "wb") as fh: pickle.dump(scaler, fh)
np.savez(run / "check.npz", te=te, X_te=Xc[te], y_te=y[te], f1_summary=summ[a.fold]["f1_macro"], n_csv=len(Xc))
print(f"{run.parent.name}: F1 {f1:.6f} equal to summary.json", flush=True)
