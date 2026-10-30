"""Idea 6 Stage 2, exploratory real-shift checks (ANALYSIS_PLAN.md amendment 2),
on stored clean embeddings only.

  prepare  subset training CSVs of the four shared classes (common labels 0 dog,
           1 siren, 2 car_horn, 3 engine), for the EfficientNet (rebuilt FP32)
           and AST (clean FP32, 1024 frames) tiers of both datasets
  e6       cross-dataset: heads trained on one dataset's shared-class clips
           (train_realshift.sh) score their own test fold and every shared-class
           clip of the other dataset; cheap and cascade accuracy, escalation,
           gate AUROC, and the cross minus in-domain escalation tests
  e7       out of distribution: each task's H1 cheap system scores the other
           dataset's clips of classes it was not trained on; escalation with the
           clean thresholds, AUROC of the margin and of idea 7's noise_lr for
           OOD against in-domain test clips, tests

    python scripts/redesign/idea6/stage2/realshift.py prepare --results R
    python scripts/redesign/idea6/stage2/realshift.py e6 --results R --stage2 S2
    python scripts/redesign/idea6/stage2/realshift.py e7 --results R --audio-stats I7/audio_stats
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from analyze import AudioStats, BUDGETS, field_arrays, fit_lr, margin, softmax, thr_rate  # noqa: E402
from idea5b_common import holm  # noqa: E402
from scripts.analysis.cascade import load_model  # noqa: E402
from score_eff import fold_scaler  # noqa: E402

E = Path.home() / "Documents" / "DATASETS"
SHARED = {"esc50": {0: 0, 42: 1, 43: 2, 44: 3}, "urban8k": {3: 0, 8: 1, 1: 2, 5: 3}}
COMMON = ["dog", "siren", "car_horn", "engine"]
AST_CSV = {"esc50": E / "ESC-50/ast_768/esc-50.csv", "urban8k": E / "Urban8k/ast_768/urbansound8k.csv"}
EFF_CSV = {"esc50": "csv/esc-50_effb3pool_fp32.csv", "urban8k": "csv/urbansound8k_effb3pool_fp32.csv"}
OTHER = {"esc50": "urban8k", "urban8k": "esc50"}
SEEDS = [0, 1, 2]
TIERS = ["w4g128", "n4"]
TASK_DS = {"esc50": "esc50", "esc5": "esc50", "urban": "urban8k"}
STATS_DS = {"esc50": "esc50", "urban8k": "urban"}


def ttest_fold(d, folds):
    fm = np.array([d[folds == f].mean() for f in np.unique(folds)])
    p = stats.ttest_1samp(fm, 0.0).pvalue if fm.std() > 0 else (0.0 if fm[0] else 1.0)
    return float(fm.mean()), float(p), len(fm)


def prepare(R: Path):
    out = R / "csv" / "realshift"
    out.mkdir(parents=True, exist_ok=True)
    for ds, mp in SHARED.items():
        for tier, src in (("eff", R / EFF_CSV[ds]), ("ast", AST_CSV[ds])):
            df = pd.read_csv(src)
            df = df[df["label"].isin(mp)].copy()
            df["label"] = df["label"].map(mp)
            df["class_id"] = df["label"]
            df["category"] = df["label"].map(dict(enumerate(COMMON)))
            idx = df.index.to_numpy()
            df.to_csv(out / f"{tier}_{ds}_shared.csv", index=False, float_format="%.8g")
            np.save(out / f"{ds}_shared_rows.npy", idx)
            print(f"{tier} {ds}: {len(df)} clips, per class {df.label.value_counts().sort_index().tolist()}, "
                  f"folds {sorted(df.folder.unique())}")


class Heads:
    """Every fold's head of one benchmark.py run with the scaler of its fit portion."""

    def __init__(self, run: Path):
        cfg = OmegaConf.load(run / "config.yaml")
        esc = cfg.experiment.datasets.esc
        df = pd.read_csv(esc.csv)
        feats = [c for c in df.columns if c.isdigit()]
        X = df[feats].to_numpy(np.float32)
        y = df[esc.get("label_col", "class_id")].to_numpy()
        n_cls = len(np.unique(y))
        self.folds = {}
        for f in sorted(int(p.name.split("_")[1]) for p in run.glob("fold_*")):
            split = json.loads((run / f"fold_{f}" / "split.json").read_text())
            model, _ = load_model(run, f, X.shape[1], n_cls)
            with contextlib.redirect_stdout(io.StringIO()):
                sc = fold_scaler(X, y, split, esc.normalization_type)
            self.folds[f] = (np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"]), sc, model)
        self.y = y

    @torch.no_grad()
    def probs(self, f, X):
        _, _, sc, model = self.folds[f]
        return softmax(model(torch.tensor(sc.transform(X), dtype=torch.float32)).numpy().astype(np.float64))


def e6(R: Path, S2: Path):
    rows, tests = [], []
    for src_ds in ("esc50", "urban8k"):
        tgt = OTHER[src_ds]
        rs, rt = np.load(R / "csv/realshift" / f"{src_ds}_shared_rows.npy"), np.load(R / "csv/realshift" / f"{tgt}_shared_rows.npy")
        emb = {bb: {d: np.load(R / "emb" / d / f"{bb}__clean.npy")[r] for d, r in ((src_ds, rs), (tgt, rt))}
               for bb in ("fp32", "w8a8")}
        ast = {c: {d: np.load(S2 / "emb" / d / f"{c}__clean.npy")[r] for d, r in ((src_ds, rs), (tgt, rt))} for c in TIERS}
        y_t = pd.read_csv(R / "csv/realshift" / f"eff_{tgt}_shared.csv", usecols=["label"])["label"].to_numpy()
        for seed in SEEDS:
            he = Heads(R / "outputs" / f"xds_eff_{src_ds}_s{seed}" / "esc")
            ha = Heads(R / "outputs" / f"xds_ast_{src_ds}_s{seed}" / "esc")
            y_s = he.y
            for f, (te, iv, _, _) in he.folds.items():
                pa = {c: {"in": ha.probs(f, ast[c][src_ds][te]).argmax(1), "x": ha.probs(f, ast[c][tgt]).argmax(1)}
                      for c in TIERS}
                for bb in ("fp32", "w8a8"):
                    p_in, p_x = he.probs(f, emb[bb][src_ds][te]), he.probs(f, emb[bb][tgt])
                    m_iv = margin(he.probs(f, emb[bb][src_ds][iv]))
                    for dom, p, yy in (("in", p_in, y_s[te]), ("cross", p_x, y_t)):
                        m, ac = margin(p), p.argmax(1)
                        for c in TIERS:
                            de = pa[c]["in" if dom == "in" else "x"]
                            fix = (ac != yy) & (de == yy)
                            for b, rate in BUDGETS.items():
                                esc = m < thr_rate(m_iv, rate)
                                pred = np.where(esc, de, ac)
                                rows.append({"source": src_ds, "target": tgt, "domain": dom, "seed": seed, "fold": f,
                                             "cheap_bb": bb, "tier": c, "budget": b, "n": len(yy),
                                             "cheap_acc": float((ac == yy).mean()), "ast_acc": float((de == yy).mean()),
                                             "cascade_acc": float((pred == yy).mean()), "rate": float(esc.mean()),
                                             "auroc_wrong": roc_auc_score(ac != yy, 1 - m) if 0 < (ac != yy).sum() < len(yy) else np.nan,
                                             "auroc_fixable": roc_auc_score(fix, 1 - m) if 0 < fix.sum() < len(yy) else np.nan})
            print(f"e6 {src_ds} seed {seed} done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(R / "analysis/e6_cross_dataset_pairs.csv", index=False)
    summ = D.groupby(["source", "target", "cheap_bb", "tier", "budget", "domain"])[
        ["cheap_acc", "ast_acc", "cascade_acc", "rate", "auroc_wrong", "auroc_fixable"]].mean().unstack("domain")
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(R / "analysis/e6_cross_dataset.csv", index=False)
    for (s, bb, b), g in D[D.tier == TIERS[0]].groupby(["source", "cheap_bb", "budget"]):
        w = g.pivot_table(index=["seed", "fold"], columns="domain", values="rate").reset_index()
        d, p, k = ttest_fold((w["cross"] - w["in"]).to_numpy(), w["fold"].to_numpy())
        tests.append({"source": s, "cheap_bb": bb, "budget": b, "rate_in": w["in"].mean(), "rate_cross": w["cross"].mean(),
                      "diff": d, "p_t_fold": p, "n_folds": k})
    T = pd.DataFrame(tests)
    T["p_holm"] = holm(T.p_t_fold.to_numpy())
    T.to_csv(R / "analysis/e6_tests.csv", index=False)
    print(summ.round(3).to_string())
    print(T.round(4).to_string())


def e7(R: Path, I7: Path):
    rows, tests = [], []
    for task, ds in TASK_DS.items():
        other = OTHER[ds]
        bb, k, e, r, b = json.loads((R / "analysis" / task / "picks.json").read_text())["H1"]["design"]
        meta_o = pd.read_csv(AST_CSV[other], usecols=["label"])
        ood = np.flatnonzero(~meta_o["label"].isin(SHARED[other]).to_numpy())
        X_in = np.load(R / "emb" / ds / f"{bb}__clean.npy")
        X_o = np.load(R / "emb" / other / f"{bb}__clean.npy")[ood]
        names_in = [Path(s).stem for s in pd.read_csv(AST_CSV[ds], usecols=["name"])["name"]]
        names_o = [Path(s).stem for s in pd.read_csv(AST_CSV[other], usecols=["name"])["name"]]
        a_in = AudioStats(I7 / STATS_DS[ds], names_in)
        st_o = np.load(I7 / STATS_DS[other] / "clean.npz", allow_pickle=True)
        if [str(s) for s in st_o["names"]] != names_o:
            raise SystemExit("audio statistics rows differ")
        S_o = st_o["stats"][ood]
        tag = {"esc50": "effp_esc50_s{}", "esc5": "effp_esc5_s{}", "urban": "effp_urban_s{}"}[task]
        seeds = [0, 1, 2] if task == "urban" else [0, 1, 2, 3, 4]
        av = field_arrays(len(names_in))[1]
        for seed in seeds:
            h = Heads(R / "outputs" / tag.format(seed) / k)
            for f, (te, iv, _, _) in h.folds.items():
                m_iv = margin(h.probs(f, X_in[iv]))
                m_te, m_o = margin(h.probs(f, X_in[te])), margin(h.probs(f, X_o))
                noise = fit_lr(a_in.rows("field", iv, "val"), (av[iv] != "clean").astype(int))
                s_te, s_o = noise(a_in.rows("clean", te, "test")), noise(S_o)
                lab = np.r_[np.zeros(len(te)), np.ones(len(ood))]
                row = {"task": task, "cheap": f"{bb}:{k}", "h1_budget": b, "seed": seed, "fold": f, "n_ood": len(ood),
                       "auroc_margin": roc_auc_score(lab, np.r_[1 - m_te, 1 - m_o]),
                       "auroc_noise_lr": roc_auc_score(lab, np.r_[s_te, s_o])}
                for bud in sorted({b, "b10", "b20", "b30"}):
                    t = thr_rate(m_iv, BUDGETS[bud])
                    row[f"rate_in_{bud}"], row[f"rate_ood_{bud}"] = float((m_te < t).mean()), float((m_o < t).mean())
                row["rate_in_h1"], row["rate_ood_h1"] = row[f"rate_in_{b}"], row[f"rate_ood_{b}"]
                rows.append(row)
        print(f"e7 {task} done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(R / "analysis/e7_ood_pairs.csv", index=False)
    for task, g in D.groupby("task", sort=False):
        d1, p1, k1 = ttest_fold((g.rate_ood_h1 - g.rate_in_h1).to_numpy(), g.fold.to_numpy())
        d2, p2, _ = ttest_fold((g.auroc_margin - g.auroc_noise_lr).to_numpy(), g.fold.to_numpy())
        tests.append({"task": task, "cheap": g.cheap.iloc[0], "h1_budget": g.h1_budget.iloc[0], "n_ood": g.n_ood.iloc[0],
                      "rate_in": g.rate_in_h1.mean(), "rate_ood": g.rate_ood_h1.mean(), "rate_diff": d1, "p_rate": p1,
                      "auroc_margin": g.auroc_margin.mean(), "auroc_noise_lr": g.auroc_noise_lr.mean(),
                      "auroc_diff": d2, "p_auroc": p2, "n_folds": k1})
    T = pd.DataFrame(tests)
    T["p_rate_holm"], T["p_auroc_holm"] = holm(T.p_rate.to_numpy()), holm(T.p_auroc.to_numpy())
    T.to_csv(R / "analysis/e7_tests.csv", index=False)
    cols = [c for c in D.columns if c.startswith("rate_")]
    D.groupby("task", sort=False)[cols + ["auroc_margin", "auroc_noise_lr"]].mean().to_csv(R / "analysis/e7_ood.csv")
    print(T.round(4).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["prepare", "e6", "e7"])
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--stage2", type=Path)
    ap.add_argument("--audio-stats", type=Path)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    if a.step == "prepare":
        prepare(a.results)
    elif a.step == "e6":
        e6(a.results, a.stage2)
    else:
        e7(a.results, a.audio_stats)


if __name__ == "__main__":
    main()
