"""Idea 6 final round, part A: real-shift checks (E6 cross-dataset, E7 unseen classes) on the small CNNs.

Stage 2's realshift.py (amendment 2) re-run with the EfficientAT tiers of Stage 3c and the 512-frame Q4 AST, plus the
noise-detector gate and the shifted-against-in-domain detection AUROC. Heads, thresholds and fold-level tests follow
Stage 2 (its Heads class, thr_rate and ttest_fold are imported unchanged). ANALYSIS_PLAN.md commit 2a05e77.

  prepare    verify the shared-class map against the datasets' own metadata, write the shared-class meta CSVs
  build-csv  training CSVs of the E6 heads from the clean FP32 embeddings (needs the extractions)
  e6         cross-dataset
  e7         unseen classes

    python realshift_final.py prepare --r R
    python realshift_final.py build-csv --r R --s2 S2 --s3b S3B --s3c S3C
    python realshift_final.py e6 --r R --s2 S2 --s3b S3B --s3c S3C --i5 I5 --i7 I7
    python realshift_final.py e7 --r R --s2 S2 --s3b S3B --s3c S3C --unp UNP --i7 I7
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
IDEA6 = HERE.parent
ROOT = HERE.parents[3]
for p in (ROOT, IDEA6, IDEA6 / "stage2", HERE):
    sys.path.insert(0, str(p))

import realshift as RS  # noqa: E402  Stage 2's E6/E7 (Heads, ttest_fold, class maps)
from analyze import AudioStats, BUDGETS, field_arrays, fit_lr, margin, thr_rate  # noqa: E402
from idea5b_common import holm  # noqa: E402

E = Path.home() / "Documents" / "DATASETS"
META = {"esc50": E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/meta/esc50.csv",
        "urban8k": E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/metadata/UrbanSound8K.csv"}
SHARED, COMMON, OTHER, AST_CSV, STATS_DS = RS.SHARED, RS.COMMON, RS.OTHER, RS.AST_CSV, RS.STATS_DS
SEEDS = [0, 1, 2]
PICKED = {"esc50": "mn20", "urban8k": "mn10"}  # the cheap tier Stage 3c picked on the source dataset
CNNS = ["mn20", "mn10"]
NAMES = {"esc50": {0: "dog", 42: "siren", 43: "car_horn", 44: "engine"},
         "urban8k": {3: "dog_bark", 8: "siren", 1: "car_horn", 5: "engine_idling"}}


def cnn_emb(a, cnn, cfg, ds):
    """Clean embedding file of a CNN on a dataset, exact lengths."""
    if ds == "esc50":
        return (a.s3c if cnn == "mn20" else a.s3b) / "emb_cnn/esc50" / f"{cnn}_{cfg}__clean.npy"
    if cnn == "mn10":
        return a.s3c / "emb_cnn/urban8k_unpadded" / f"mn10_{cfg}__clean.npy"
    return a.r / "emb_cnn/urban8k_exact_e6" / f"mn20_{cfg}__clean.npy"


def load_emb(path: Path, names):
    X = np.load(path)
    side = json.loads(path.with_suffix(".json").read_text())
    if side["names_crc32"] != zlib.crc32("\n".join(names).encode()) or len(X) != len(names) or not np.isfinite(X).all():
        raise SystemExit(f"{path}: rows do not match the dataset CSV")
    return X


def stems_of(ds):
    return [Path(s).stem for s in pd.read_csv(AST_CSV[ds], usecols=["name"])["name"]]


def prepare(a):
    out = a.r / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (a.r / "csv").mkdir(exist_ok=True)
    chk = {}
    esc = pd.read_csv(META["esc50"])
    urb = pd.read_csv(META["urban8k"])
    for t, name in NAMES["esc50"].items():
        got = set(esc.loc[esc.target == t, "category"])
        if got != {name}:
            raise SystemExit(f"ESC-50 target {t}: metadata says {got}, expected {name}")
    for t, name in NAMES["urban8k"].items():
        got = set(urb.loc[urb.classID == t, "class"])
        if got != {name}:
            raise SystemExit(f"UrbanSound8K classID {t}: metadata says {got}, expected {name}")
    # the dataset CSVs the embeddings follow: label equals the metadata's target / classID row by row
    for ds, meta, key, fn in (("esc50", esc, "target", "filename"), ("urban8k", urb, "classID", "slice_file_name")):
        cs = pd.read_csv(AST_CSV[ds], usecols=["name", "label", "folder"])
        lab = dict(zip(meta[fn].map(lambda s: Path(s).stem), meta[key]))
        fold = dict(zip(meta[fn].map(lambda s: Path(s).stem), meta["fold"]))
        stems = cs["name"].map(lambda s: Path(s).stem)
        if not all(lab[s] == l for s, l in zip(stems, cs["label"])) or not all(fold[s] == f for s, f in zip(stems, cs["folder"])):
            raise SystemExit(f"{ds}: the dataset CSV disagrees with the metadata on label or fold")
        mp = SHARED[ds]
        sub = pd.read_csv(AST_CSV[ds], usecols=["folder", "name", "label", "category", "class_id"])
        rows = np.flatnonzero(sub["label"].isin(mp).to_numpy())
        sub = sub.iloc[rows].copy()
        sub["label"] = sub["label"].map(mp)
        sub["class_id"] = sub["label"]
        sub["category"] = sub["label"].map(dict(enumerate(COMMON)))
        sub.to_csv(a.r / "csv" / f"shared_{ds}_meta.csv", index=False)
        np.save(a.r / "csv" / f"{ds}_shared_rows.npy", rows)
        chk[ds] = {"shared_clips": int(len(rows)), "per_common_label": sub["label"].value_counts().sort_index().tolist(),
                   "folds": sorted(int(f) for f in sub["folder"].unique()), "classes": NAMES[ds],
                   "dataset_csv_matches_metadata": True}
    chk["common_labels"] = COMMON
    (out / "class_map_check.json").write_text(json.dumps(chk, indent=1))
    print(json.dumps(chk, indent=1))


def build_csv(a):
    """Training CSVs of the E6 heads: shared-class rows of the clean FP32 embeddings, tier xds_<cnn|ast512>_<dataset>.csv."""
    for ds in ("esc50", "urban8k"):
        rows = np.load(a.r / "csv" / f"{ds}_shared_rows.npy")
        meta = pd.read_csv(a.r / "csv" / f"shared_{ds}_meta.csv")
        names = stems_of(ds)
        for cnn in CNNS:
            X = load_emb(cnn_emb(a, cnn, "fp32", ds), names)[rows]
            df = pd.concat([meta, pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])], axis=1)
            df.to_csv(a.r / "csv" / f"xds_{cnn}_{ds}.csv", index=False, float_format="%.8g")
        p = a.r / "emb_ast/shared512" / ds / "fp32__clean.npy"
        X = np.load(p)
        side = json.loads(p.with_suffix(".json").read_text())
        if len(X) != len(meta) or side["names_crc32"] != zlib.crc32("\n".join(Path(s).stem for s in meta["name"]).encode()):
            raise SystemExit(f"{p}: rows do not match the shared-class CSV")
        df = pd.concat([meta, pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])], axis=1)
        df.to_csv(a.r / "csv" / f"xds_ast512_{ds}.csv", index=False, float_format="%.8g")
        print(ds, "csv written", flush=True)


def one_sided_two(d):
    return RS.ttest_fold(np.asarray(d), np.arange(len(d)))


def e6(a):
    torch.set_num_threads(2)
    out = a.r / "analysis"
    rate_rows, casc_rows, det_rows = [], [], []
    names_all = {ds: stems_of(ds) for ds in SHARED}
    audio = {ds: AudioStats(a.i7 / "audio_stats" / STATS_DS[ds], names_all[ds]) for ds in SHARED}
    folder_all = {ds: pd.read_csv(AST_CSV[ds], usecols=["folder"])["folder"].to_numpy() for ds in SHARED}
    av = {ds: field_arrays(len(names_all[ds]))[1] for ds in SHARED}
    clean_stats = {ds: audio[ds].cond("clean") for ds in SHARED}
    tiers_ast = {"w4g128@512": ("ast512", None), "w4g128@1024": ("old", "w4g128"), "n4@1024": ("old", "n4")}
    for src in ("esc50", "urban8k"):
        tgt = OTHER[src]
        rs, rt = np.load(a.r / "csv" / f"{src}_shared_rows.npy"), np.load(a.r / "csv" / f"{tgt}_shared_rows.npy")
        y_t = pd.read_csv(a.r / "csv" / f"shared_{tgt}_meta.csv", usecols=["label"])["label"].to_numpy()
        emb = {cnn: {d: load_emb(cnn_emb(a, cnn, "w8a8", d), names_all[d])[r] for d, r in ((src, rs), (tgt, rt))}
               for cnn in CNNS}
        ast = {}
        for t, (kind, cfg) in tiers_ast.items():
            if kind == "ast512":
                ast[t] = {d: np.load(a.r / "emb_ast/shared512" / d / "w4g128__clean.npy") for d in (src, tgt)}
            else:
                ast[t] = {d: np.load(a.i5 / "emb" / d / f"{cfg}__clean.npy")[r] for d, r in ((src, rs), (tgt, rt))}
        noise = {}
        for seed in SEEDS:
            hc = {cnn: RS.Heads(a.r / "outputs" / f"xds_{cnn}_{src}_s{seed}" / "esc") for cnn in CNNS}
            ha = {"w4g128@512": RS.Heads(a.r / "outputs" / f"xds_ast512_{src}_s{seed}" / "esc"),
                  "w4g128@1024": RS.Heads(a.s2 / "outputs" / f"xds_ast_{src}_s{seed}" / "esc")}
            ha["n4@1024"] = ha["w4g128@1024"]
            y_s = hc["mn20"].y
            folds = sorted(hc["mn20"].folds)
            for f in folds:
                te, iv = hc["mn20"].folds[f][0], hc["mn20"].folds[f][1]
                for h in [*hc.values(), *ha.values()]:
                    if not (np.array_equal(h.folds[f][0], te) and np.array_equal(h.folds[f][1], iv)):
                        raise SystemExit(f"{src} seed {seed} fold {f}: heads disagree on the split")
                fid = int(folder_all[src][rs[te]][0])
                if fid not in noise:  # noise_lr: all source clips outside the test fold, validation field assignment
                    pool = np.flatnonzero(folder_all[src] != fid)
                    noise[fid] = fit_lr(audio[src].rows("field", pool, "val"), (av[src][pool] != "clean").astype(int))
                nz_in, nz_x = noise[fid](clean_stats[src][rs[te]]), noise[fid](clean_stats[tgt][rt])
                pa = {t: {"in": ha[t].probs(f, ast[t][src][te]).argmax(1), "x": ha[t].probs(f, ast[t][tgt]).argmax(1)}
                      for t in tiers_ast}
                for cnn in CNNS:
                    h = hc[cnn]
                    p_in, p_x = h.probs(f, emb[cnn][src][te]), h.probs(f, emb[cnn][tgt])
                    m_iv = margin(h.probs(f, emb[cnn][src][iv]))
                    m_in, m_x = margin(p_in), margin(p_x)
                    lab = np.r_[np.zeros(len(te)), np.ones(len(rt))]
                    det_rows.append({"source": src, "target": tgt, "cnn": cnn, "seed": seed, "fold": f, "n_in": len(te),
                                     "n_shift": len(rt),
                                     "auroc_margin": roc_auc_score(lab, np.r_[1 - m_in, 1 - m_x]),
                                     "auroc_noise_lr": roc_auc_score(lab, np.r_[nz_in, nz_x]), "auroc_random": 0.5})
                    for dom, p, yy, m in (("in", p_in, y_s[te], m_in), ("cross", p_x, y_t, m_x)):
                        ac = p.argmax(1)
                        wrong = ac != yy
                        for b, rate in BUDGETS.items():
                            esc = m < thr_rate(m_iv, rate)
                            rate_rows.append({"source": src, "target": tgt, "cnn": cnn, "domain": dom, "seed": seed, "fold": f,
                                              "budget": b, "n": len(yy), "cheap_acc": float((ac == yy).mean()),
                                              "rate": float(esc.mean()),
                                              "auroc_wrong": roc_auc_score(wrong, 1 - m) if 0 < wrong.sum() < len(yy) else np.nan})
                            for t in tiers_ast:
                                de = pa[t]["in" if dom == "in" else "x"]
                                pred = np.where(esc, de, ac)
                                fix = wrong & (de == yy)
                                casc_rows.append({"source": src, "target": tgt, "cnn": cnn, "tier": t, "domain": dom,
                                                  "seed": seed, "fold": f, "budget": b, "ast_acc": float((de == yy).mean()),
                                                  "cascade_acc": float((pred == yy).mean()),
                                                  "auroc_fixable": roc_auc_score(fix, 1 - m) if 0 < fix.sum() < len(yy) else np.nan})
            print(f"e6 {src} seed {seed} done", flush=True)
    R_, C_, D_ = pd.DataFrame(rate_rows), pd.DataFrame(casc_rows), pd.DataFrame(det_rows)
    R_.to_csv(out / "e6_rates_pairs.csv", index=False)
    C_.to_csv(out / "e6_cascade_pairs.csv", index=False)
    D_.to_csv(out / "e6_detect_pairs.csv", index=False)
    g = R_.groupby(["source", "target", "cnn", "budget", "domain"])[["cheap_acc", "rate", "auroc_wrong"]].mean().unstack("domain")
    g.columns = [f"{x}_{y}" for x, y in g.columns]
    g = g.reset_index()
    gc = C_.groupby(["source", "target", "cnn", "tier", "budget", "domain"])[["ast_acc", "cascade_acc", "auroc_fixable"]].mean().unstack("domain")
    gc.columns = [f"{x}_{y}" for x, y in gc.columns]
    g.to_csv(out / "e6_rates.csv", index=False)
    gc.reset_index().to_csv(out / "e6_cascade.csv", index=False)
    D_.groupby(["source", "target", "cnn"])[["auroc_margin", "auroc_noise_lr", "auroc_random"]].mean().reset_index().to_csv(
        out / "e6_detect.csv", index=False)
    # tests
    tests = []
    for (s, cnn, b), gg in R_.groupby(["source", "cnn", "budget"]):
        w = gg.pivot_table(index=["seed", "fold"], columns="domain", values="rate").reset_index()
        d, p, k = RS.ttest_fold((w["cross"] - w["in"]).to_numpy(), w["fold"].to_numpy())
        fam = "A1" if cnn == PICKED[s] else "A2"
        tests.append({"family": fam, "source": s, "cnn": cnn, "budget": b, "rate_in": w["in"].mean(),
                      "rate_shift": w["cross"].mean(), "diff": d, "p_t_fold": p, "n_folds": k})
    T1 = pd.DataFrame(tests)
    T1["p_holm"] = np.nan
    for fam, ix in T1.groupby("family").groups.items():
        T1.loc[ix, "p_holm"] = holm(T1.loc[ix, "p_t_fold"].to_numpy())
        T1.loc[ix, "family_size"] = len(ix)
    T1.to_csv(out / "e6_tests_rate.csv", index=False)
    tests = []
    for (s, cnn), gg in D_.groupby(["source", "cnn"]):
        for what, col in (("margin minus noise_lr", gg.auroc_margin - gg.auroc_noise_lr), ("margin minus 0.5", gg.auroc_margin - 0.5)):
            d, p, k = RS.ttest_fold(col.to_numpy(), gg.fold.to_numpy())
            tests.append({"family": "A3" if cnn == PICKED[s] else "A4", "source": s, "cnn": cnn, "contrast": what,
                          "auroc_margin": gg.auroc_margin.mean(), "auroc_noise_lr": gg.auroc_noise_lr.mean(), "diff": d,
                          "p_t_fold": p, "n_folds": k})
    T2 = pd.DataFrame(tests)
    T2["p_holm"] = np.nan
    for fam, ix in T2.groupby("family").groups.items():
        T2.loc[ix, "p_holm"] = holm(T2.loc[ix, "p_t_fold"].to_numpy())
        T2.loc[ix, "family_size"] = len(ix)
    T2.to_csv(out / "e6_tests_detect.csv", index=False)
    print(T1.round(4).to_string())
    print(T2.round(4).to_string())


def e7(a):
    torch.set_num_threads(2)
    out = a.r / "analysis"
    # (task, dataset, cnn, head kind, budget, head folder, seeds, family)
    plans = [("esc50", "esc50", "mn20", "4", "b05", lambda s: a.s3c / "outputs" / f"mn20_esc50_s{s}", [0, 1, 2, 3, 4], "A5"),
             ("esc50", "esc50", "mn10", "4", "b05", lambda s: a.s3b / "outputs" / f"mn10_esc50_s{s}", [0, 1, 2, 3, 4], "A6"),
             ("urban", "urban8k", "mn10", "esc", "b30", lambda s: a.s3c / "outputs_unp" / f"mn10unp_urban_s{s}", [0, 1, 2], "A5")]
    if a.plans:  # code test while part of the embeddings are still missing; the final run takes all three
        plans = [plans[i] for i in a.plans]
    rows = []
    for task, ds, cnn, kind, bud, runf, seeds, fam in plans:
        other = OTHER[ds]
        names_in, names_o = stems_of(ds), stems_of(other)
        meta_o = pd.read_csv(AST_CSV[other], usecols=["label"])
        ood = np.flatnonzero(~meta_o["label"].isin(SHARED[other]).to_numpy())
        X_in = load_emb(cnn_emb(a, cnn, "w8a8", ds), names_in)
        X_o = load_emb(cnn_emb(a, cnn, "w8a8", other), names_o)[ood]
        a_in = AudioStats(a.i7 / "audio_stats" / STATS_DS[ds], names_in)
        st_o = np.load(a.i7 / "audio_stats" / STATS_DS[other] / "clean.npz", allow_pickle=True)
        if [str(s) for s in st_o["names"]] != names_o:
            raise SystemExit("audio statistics rows differ")
        S_o = st_o["stats"][ood]
        avv = field_arrays(len(names_in))[1]
        for seed in seeds:
            h = RS.Heads(runf(seed) / kind)
            for f, (te, iv, _, _) in h.folds.items():
                m_iv = margin(h.probs(f, X_in[iv]))
                m_te, m_o = margin(h.probs(f, X_in[te])), margin(h.probs(f, X_o))
                noise = fit_lr(a_in.rows("field", iv, "val"), (avv[iv] != "clean").astype(int))
                s_te, s_o = noise(a_in.rows("clean", te, "test")), noise(S_o)
                lab = np.r_[np.zeros(len(te)), np.ones(len(ood))]
                row = {"task": task, "cnn": cnn, "kind": kind, "family": fam, "pick_budget": bud, "seed": seed, "fold": f,
                       "n_ood": len(ood), "n_in": len(te), "auroc_margin": roc_auc_score(lab, np.r_[1 - m_te, 1 - m_o]),
                       "auroc_noise_lr": roc_auc_score(lab, np.r_[s_te, s_o]), "auroc_random": 0.5}
                for b, rate in BUDGETS.items():
                    t = thr_rate(m_iv, rate)
                    row[f"rate_in_{b}"], row[f"rate_ood_{b}"] = float((m_te < t).mean()), float((m_o < t).mean())
                row["rate_in_pick"], row["rate_ood_pick"] = row[f"rate_in_{bud}"], row[f"rate_ood_{bud}"]
                rows.append(row)
        print(f"e7 {task} {cnn} done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(out / "e7_pairs.csv", index=False)
    cols = [c for c in D.columns if c.startswith("rate_") or c.startswith("auroc_")]
    D.groupby(["task", "cnn", "family"], sort=False)[["n_ood", *cols]].mean().reset_index().to_csv(out / "e7_ood.csv", index=False)
    tests = []
    for (task, cnn, fam), g in D.groupby(["task", "cnn", "family"], sort=False):
        d1, p1, k = RS.ttest_fold((g.rate_ood_pick - g.rate_in_pick).to_numpy(), g.fold.to_numpy())
        tests.append({"family": fam, "test": "unseen minus in-domain escalation", "task": task, "cnn": cnn,
                      "budget": g.pick_budget.iloc[0], "value_a": g.rate_in_pick.mean(), "value_b": g.rate_ood_pick.mean(),
                      "diff": d1, "p_t_fold": p1, "n_folds": k, "n_ood": int(g.n_ood.iloc[0])})
        for what, col in (("margin minus noise_lr", g.auroc_margin - g.auroc_noise_lr), ("margin minus 0.5", g.auroc_margin - 0.5)):
            d, p, k = RS.ttest_fold(col.to_numpy(), g.fold.to_numpy())
            tests.append({"family": "A7" if fam == "A5" else "A8", "test": "detection AUROC " + what, "task": task, "cnn": cnn,
                          "budget": g.pick_budget.iloc[0], "value_a": g.auroc_margin.mean(),
                          "value_b": g.auroc_noise_lr.mean() if "noise" in what else 0.5, "diff": d, "p_t_fold": p,
                          "n_folds": k, "n_ood": int(g.n_ood.iloc[0])})
    T = pd.DataFrame(tests)
    T["p_holm"], T["family_size"] = np.nan, np.nan
    for fam, ix in T.groupby("family").groups.items():
        T.loc[ix, "p_holm"] = holm(T.loc[ix, "p_t_fold"].to_numpy())
        T.loc[ix, "family_size"] = len(ix)
    T.to_csv(out / "e7_tests.csv", index=False)
    print(T.round(4).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["prepare", "build-csv", "e6", "e7"])
    for k in ("r", "s2", "s3b", "s3c", "unp", "i5", "i7"):
        ap.add_argument(f"--{k}", type=Path)
    ap.add_argument("--plans", nargs="*", type=int, default=None)
    a = ap.parse_args(argv)
    {"prepare": prepare, "build-csv": build_csv, "e6": e6, "e7": e7}[a.step](a)


if __name__ == "__main__":
    main()
