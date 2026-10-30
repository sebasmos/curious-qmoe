"""Reviewer extras, task D: AST-Q4 group-128 (w4g128, 1024 frames) on ESC-50 (50 classes) with a head trained on its own clean
embeddings (this folder's outputs/astesc50headw4g128_s<seed>/esc, seeds 0 to 4), against the FP32-trained head transferred
unchanged (the paper's Tables 2 and 3). The per-channel 4-bit rows (w4) come from the external-review check
(checks/q4ch_head_transfer_raw.csv), so the two layouts are side by side. Same fold, scaler and scoring code as
q4ch_head_transfer.py (score_stage2.train_frame / fold_scaler / logits_of).

    python d_own_head.py I5 R_NEW CHECKS_RAW_CSV OUT_DIR
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import f1_score

I5, RN, RAW, OUT = map(Path, sys.argv[1:5])
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/redesign/idea5"))
from scripts.analysis.cascade import load_model  # noqa
from score_stage2 import train_frame, fold_scaler, logits_of  # noqa
torch.set_num_threads(2)
OUT.mkdir(parents=True, exist_ok=True)
CONF = ["clean"] + [f"{n}__{s}db" for n in ("bg", "pink", "white") for s in (20, 10, 5, 0)]
rows = []
for seed in range(5):
    run = I5 / f"outputs/astesc50_s{seed}/esc"
    cfg = OmegaConf.load(run / "config.yaml"); Xc, y = train_frame(cfg); n_cls = len(np.unique(y))
    norm = cfg.experiment.datasets.esc.normalization_type
    orun = RN / f"outputs/astesc50headw4g128_s{seed}/esc"
    ocfg = OmegaConf.load(orun / "config.yaml"); Xo, yo = train_frame(ocfg)
    assert np.array_equal(y, yo)
    emb = {k: np.load(I5 / f"emb/esc50/w4g128__{k}.npy") for k in CONF}
    assert np.allclose(emb["clean"], Xo, atol=1e-5), "own-head CSV is not the clean w4g128 embedding"
    for fold in sorted(f["fold"] for f in json.loads((run / "summary.json").read_text())["folds"]):
        split = json.loads((run / f"fold_{fold}" / "split.json").read_text())
        osplit = json.loads((orun / f"fold_{fold}" / "split.json").read_text())
        assert split["test_idx"] == osplit["test_idx"], "test folds differ"
        te = np.asarray(split["test_idx"])
        model, _ = load_model(run, fold, Xc.shape[1], n_cls); sc = fold_scaler(Xc, y, split, norm)
        omodel, _ = load_model(orun, fold, Xo.shape[1], n_cls); osc = fold_scaler(Xo, yo, osplit, norm)
        for k in CONF:
            X = emb[k]
            f = lambda m, s: f1_score(y[te], logits_of(m, s, X[te], n_cls).argmax(1), average="macro", zero_division=0)
            rows += [dict(seed=seed, fold=fold, cond=k, variant="T", f1=f(model, sc)), dict(seed=seed, fold=fold, cond=k, variant="O", f1=f(omodel, osc))]
    print("seed", seed, flush=True)
D = pd.DataFrame(rows); D.to_csv(OUT / "d_w4g128_raw.csv", index=False)
raw = pd.read_csv(RAW)
# reproduction of the check's transferred w4g128 and FP32 rows
t_old = raw[(raw.cfg == "w4g128") & (raw.variant == "T")].set_index(["seed", "fold", "cond"]).f1
t_new = D[D.variant == "T"].set_index(["seed", "fold", "cond"]).f1
print("max |T - check's T| =", float((t_old - t_new).abs().max()))
def fold_mean(df):
    return df.groupby(["fold", "seed"]).f1.mean().groupby("fold").mean()
out = []
for cond in CONF:
    fp = fold_mean(raw[(raw.cfg == "fp32") & (raw.variant == "T") & (raw.cond == cond)])
    g = {"w4g128 transferred": D[(D.variant == "T") & (D.cond == cond)], "w4g128 own head": D[(D.variant == "O") & (D.cond == cond)],
         "w4 transferred": raw[(raw.cfg == "w4") & (raw.variant == "T") & (raw.cond == cond)],
         "w4 own head": raw[(raw.cfg == "w4") & (raw.variant == "O") & (raw.cond == cond)]}
    for name, df in g.items():
        fm = fold_mean(df)
        if name == "w4 own head":
            fm = fm  # own heads exist for the 5 seeds in the check's csv
        d = (fm - fp) * 100; se = d.std(ddof=1) / np.sqrt(len(d)); h = stats.t.ppf(0.975, len(d) - 1) * se
        out.append(dict(cond=cond, variant=name, f1=df.f1.mean(), n_pairs=len(df), loss_vs_fp32_pts=d.mean(), ci_lo=d.mean() - h, ci_hi=d.mean() + h,
                        fp32_f1=raw[(raw.cfg == "fp32") & (raw.variant == "T") & (raw.cond == cond)].f1.mean()))
    for a_, b_ in (("w4g128 own head", "w4g128 transferred"), ("w4 own head", "w4 transferred")):
        d = (fold_mean(g[a_]) - fold_mean(g[b_])) * 100
        out.append(dict(cond=cond, variant=f"{a_} minus {b_}", f1=np.nan, n_pairs=len(d), loss_vs_fp32_pts=d.mean(),
                        ci_lo=np.nan, ci_hi=np.nan, p_t_fold=float(stats.ttest_1samp(d, 0).pvalue)))
O = pd.DataFrame(out); O.to_csv(OUT / "d_summary.csv", index=False)
pd.set_option("display.width", 220)
print(O[O.cond.isin(["clean", "white__0db", "white__5db", "white__10db", "bg__0db", "pink__0db"])].round(4).to_string())
