"""Margin AUROC for 'the model is wrong' of AST-Q4ch on ESC-50 (50 classes), over clean + nine noise cells (10, 5, 0 dB),
paired difference to the FP32 backbone with its FP32 head (as Table 10). Head variants as in q4ch_head_transfer.py."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from omegaconf import OmegaConf
from scipy import stats
W = Path(__file__).resolve().parents[3]  # archive root
sys.path.insert(0, str(W)); sys.path.insert(0, str(W / "scripts/redesign/idea5"))
from scripts.analysis.cascade import load_model
from score_stage2 import train_frame, fold_scaler, logits_of
torch.set_num_threads(2)
R = Path("/Users/cajas.sebastian/Documents/qmoe-results/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38")
HERE = Path(__file__).resolve().parent
CONF = ["clean"] + [f"{n}__{s}db" for n in ("bg", "pink", "white") for s in (10, 5, 0)]
def margin_auroc(logits, y):
    p = torch.softmax(torch.tensor(logits), 1).numpy(); s = np.sort(p, 1); m = 1 - (s[:, -1] - s[:, -2])
    wrong = logits.argmax(1) != y
    return roc_auc_score(wrong, m) if 0 < wrong.sum() < len(y) else np.nan
rows = []
for seed in range(5):
    run = R / f"outputs/astesc50_s{seed}/esc"; cfg = OmegaConf.load(run / "config.yaml")
    Xc, y = train_frame(cfg); n_cls = 50; norm = cfg.experiment.datasets.esc.normalization_type
    orun = R / "outputs/astesc50headw4_s0/esc" if seed == 0 else HERE / f"heads_w4/astesc50headw4_s{seed}/esc"
    ocfg = OmegaConf.load(orun / "config.yaml"); Xo, yo = train_frame(ocfg)
    emb = {(c, k): np.load(R / f"emb/esc50/{c}__{k}.npy") for c in ("fp32", "w4") for k in CONF}
    for fold in range(1, 6):
        split = json.loads((run / f"fold_{fold}" / "split.json").read_text()); te = np.asarray(split["test_idx"])
        model, _ = load_model(run, fold, 768, n_cls); sc_fp = fold_scaler(Xc, y, split, norm)
        sc_w4 = fold_scaler(emb[("w4", "clean")], y, split, norm)
        osplit = json.loads((orun / f"fold_{fold}" / "split.json").read_text())
        omodel, _ = load_model(orun, fold, 768, n_cls); sc_own = fold_scaler(Xo, yo, osplit, norm)
        for k in CONF:
            for name, (m, s, X) in {"FP32": (model, sc_fp, emb[("fp32", k)]), "Q4ch_T": (model, sc_fp, emb[("w4", k)]),
                                    "Q4ch_Rfull": (model, sc_w4, emb[("w4", k)]), "Q4ch_O": (omodel, sc_own, emb[("w4", k)])}.items():
                rows.append(dict(seed=seed, fold=fold, cond=k, variant=name, auroc=margin_auroc(logits_of(m, s, X[te], n_cls), y[te])))
    print("seed", seed, flush=True)
d = pd.DataFrame(rows); d.to_csv(HERE / "q4ch_auroc_raw.csv", index=False)
f = d.groupby(["variant", "fold"]).auroc.mean().unstack("variant")      # mean over conditions and seeds, per fold
print("mean margin AUROC over 10 cells:", f.mean().round(4).to_dict())
for v in ["Q4ch_T", "Q4ch_Rfull", "Q4ch_O"]:
    diff = f[v] - f["FP32"]; h = stats.t.ppf(.975, 4) * diff.std(ddof=1) / np.sqrt(5)
    print(f"{v}: diff to FP32 {diff.mean():+.4f}  95% t-interval [{diff.mean()-h:+.4f}, {diff.mean()+h:+.4f}]")
