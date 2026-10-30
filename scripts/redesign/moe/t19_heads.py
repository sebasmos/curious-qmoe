"""T19 heads and scores (ANALYSIS_PLAN_T19.md B5, step 19.8), CPU.

Trains the protocol 3 FP32 `esc` head of scripts/benchmark.py on the clean embeddings of one configuration, one seed, per fold: the same
calls in the same order as benchmark.run_cv for one model kind (set_seed, grouped inner_split, EmbeddingAdaptDataset with the standard
scaler fitted on the fit rows, seeded loader, build_model, the two FLOP traces, class weights, train_pytorch_local, best checkpoint).
Features go through the same text round trip as benchmark.py's CSV input ('%.8g'). Then scores every condition file of the
configuration: per file <config>__<cond>.npz with y, test_fold, test_logits (each clip by the head of the fold that holds it out),
iv_idx_<k>, iv_logits_<k> (fold k's inner-validation rows) and, for TAU, ivs_idx_<k>, ivs_logits_<k> (the other devices' rows of fold
k's inner-validation locations). NaN rows of a file give NaN logits.

    python t19_heads.py --task esc50 --config w8a8 --seed 0 [--folds 1 --no-test]   (P1: fold 1 only, no test rows scored)
    python t19_heads.py --task esc50 --config n4a4 --seed 0 --transfer w8a8            (score n4a4 files with the w8a8 heads)
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t19_common import OPTB, ROOT, S2, TASKS, results_dir  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))

torch.set_num_threads(2)


def emb_dir(task, config):
    t = TASKS[task]
    if config.startswith("L:"):
        return OPTB / "emb_ast/tau" if task == "tau" else S2 / "emb_ast_short" / t["emb"]
    return results_dir() / "emb" / t["emb"]


def stem_of(config):
    return config[2:] if config.startswith("L:") else config


def cfg_of(task, seed):
    name = "urban8k" if task in ("urban", "tau") else "esc50"
    cfg = OmegaConf.load(ROOT / "config" / f"{name}.yaml")
    with open_dict(cfg):
        e = cfg.experiment
        e.datasets.esc.normalization_type = "standard"
        e.device, e.track_emissions, e.num_threads, e.seed = "cpu", False, 2, int(seed)
        e.datasets.esc.label_col = TASKS[task]["label_col"]
        e.cross_validation.inner_split_groups = "recording"
        e.models_to_run = ["esc"]
        if task == "tau":
            e.cross_validation.n_splits = 5
            e.cross_validation.recording_token = 2
    return cfg


def csv_roundtrip(X):
    buf = io.StringIO()
    pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]).to_csv(buf, index=False, float_format="%.8g")
    buf.seek(0)
    return pd.read_csv(buf)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["esc50", "esc5", "urban", "tau"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--no-test", action="store_true", help="P1: score inner-validation rows only")
    ap.add_argument("--transfer", default=None, help="score this config's files with the heads of another config")
    ap.add_argument("--only", nargs="*", default=None, help="condition names to score (default: every file)")
    a = ap.parse_args()
    t0 = time.time()
    from benchmark import build_model  # noqa: E402  (scripts/benchmark.py, imported for the identical model factory)
    from QWave.datasets import EmbeddingAdaptDataset, inner_split, make_outer_folds, recording_groups
    from QWave.train_utils import train_pytorch_local
    from QWave.utils import set_seed
    from fvcore.nn import FlopCountAnalysis
    from torch.utils.data import DataLoader

    T = TASKS[a.task]
    R = results_dir()
    cfg = cfg_of(a.task, a.seed)
    meta = pd.read_csv(T["csv"], usecols=lambda c: c in ("folder", "name", "label", "class_id", "device", "location"))
    n_all = len(meta)
    lab_all = meta[T["label_col"]].to_numpy().astype(np.int64)
    if a.task == "tau":
        isa = (meta.device == "a").to_numpy()
        n_tr = int(isa.sum())
        assert isa[:n_tr].all() and not isa[n_tr:].any()
    else:
        n_tr = n_all
    head_cfg = a.transfer or a.config
    hdir = R / "heads" / a.task / head_cfg.replace(":", "_") / f"s{a.seed}"
    cv = cfg.experiment.cross_validation
    df_meta = meta.iloc[:n_tr].copy()
    labels = lab_all[:n_tr]
    folds = make_outer_folds(df_meta, labels, cv)
    if a.folds:
        folds = [f for f in folds if f[0] in set(a.folds)]
    groups = recording_groups(df_meta["name"].to_numpy(), cv.recording_token)
    ed = emb_dir(a.task, a.config)
    stem = stem_of(a.config)
    heads, summary = {}, {"task": a.task, "config": a.config, "seed": a.seed, "head_of": head_cfg, "folds": []}
    if a.transfer is None:
        X = np.load(ed / f"{stem}__clean.npy")[:n_tr]
        feat = csv_roundtrip(X)
        df_feat = pd.concat([feat, pd.DataFrame({"class_id": labels})], axis=1)
        in_dim, n_cls, norm = 768, int(len(np.unique(labels))), cfg.experiment.datasets.esc.normalization_type
        bs = cfg.experiment.train.batch_size
        for fold, tr_idx, te_idx in folds:
            fd = hdir / f"fold_{fold}"
            fd.mkdir(parents=True, exist_ok=True)
            loader_gen = set_seed(a.seed)
            fit_idx, iv_idx = inner_split(tr_idx, labels, float(cv.validation_split_ratio), a.seed, groups)
            assert np.isfinite(X[fit_idx]).all() and np.isfinite(X[iv_idx]).all(), "training rows not extracted"
            tr_ds = quiet(EmbeddingAdaptDataset, df_feat.iloc[fit_idx].reset_index(drop=True), normalization_type=norm)
            iv_ds = quiet(EmbeddingAdaptDataset, df_feat.iloc[iv_idx].reset_index(drop=True), normalization_type=norm,
                          scaler=tr_ds.get_scaler())
            te_ds = quiet(EmbeddingAdaptDataset, df_feat.iloc[te_idx].reset_index(drop=True).fillna(0.0), normalization_type=norm,
                          scaler=tr_ds.get_scaler())
            tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, generator=loader_gen)
            iv_ld = DataLoader(iv_ds, batch_size=bs, shuffle=False)
            with open_dict(cfg):
                cfg.experiment.current_fold = fold
            model = build_model("esc", in_dim, n_cls, cfg)
            pre = {k: v.clone() for k, v in model.state_dict().items()}
            quiet(lambda: FlopCountAnalysis(model, tr_ds.features[:bs]).total())
            quiet(lambda: FlopCountAnalysis(model, te_ds.features[:bs]).total())
            model.load_state_dict(pre)
            cw = torch.tensor(1 / np.bincount(tr_ds.labels.numpy()), dtype=torch.float32)
            ck = fd / "best_model.pth"
            res = quiet(train_pytorch_local, cfg.experiment, model, tr_ld, iv_ld, cw, in_dim, torch.device("cpu"), str(fd), False, str(ck))
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            model.eval()
            sc = tr_ds.get_scaler()
            np.savez(fd / "scaler.npz", mean=sc.mean_, scale=sc.scale_)
            (fd / "split.json").write_text(json.dumps({"fold": int(fold), "seed": a.seed, "test_idx": te_idx.tolist(),
                                                       "inner_val_idx": iv_idx.tolist()}))
            heads[fold] = (model, sc, iv_idx, te_idx)
            summary["folds"].append({"fold": int(fold), "inner_val_f1_selected": float(res[3]), "n_fit": int(len(fit_idx)),
                                     "n_iv": int(len(iv_idx)), "epochs": len(res[1])})
    else:
        from benchmark import build_model as bm
        for fold, tr_idx, te_idx in folds:
            fd = hdir / f"fold_{fold}"
            model = bm("esc", 768, int(len(np.unique(labels))), cfg)
            model.load_state_dict(torch.load(fd / "best_model.pth", map_location="cpu"))
            model.eval()
            z = np.load(fd / "scaler.npz")
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            sc.mean_, sc.scale_, sc.var_, sc.n_features_in_ = z["mean"], z["scale"], z["scale"] ** 2, 768
            sp = json.loads((fd / "split.json").read_text())
            heads[fold] = (model, sc, np.array(sp["inner_val_idx"]), np.array(sp["test_idx"]))

    @torch.no_grad()
    def logits(model, sc, E):
        out = np.full((len(E), int(T["K"])), np.nan, np.float32)
        ok = np.isfinite(E).all(1)
        if ok.any():
            x = sc.transform(csv_roundtrip(E[ok]).to_numpy(np.float64))
            out[ok] = model(torch.tensor(x, dtype=torch.float32)).numpy()
        return out

    fold_all = meta["folder"].to_numpy()
    loc = meta["location"].to_numpy() if a.task == "tau" else None
    files = sorted(ed.glob(f"{stem}__*.npy"))
    if a.only:
        files = [f for f in files if f.stem.split("__", 1)[1] in set(a.only)]
    suffix = "" if a.transfer in (None, a.config) else f"@{a.transfer}"   # --transfer <own config>: rescore with the own heads
    sdir = R / "scores" / a.task / (a.config.replace(":", "_") + suffix) / f"s{a.seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    crc = json.loads((ed / f"{stem}__clean.json").read_text())["names_crc32"]
    for f in files:
        if f.name.endswith(".tmp.npy"):
            continue
        cond = f.stem.split("__", 1)[1]
        E = np.load(f)
        assert len(E) == n_all
        out = {"y": lab_all, "test_fold": fold_all, "names_crc32": crc}
        test = np.full((n_all, int(T["K"])), np.nan, np.float32)
        for fold, (model, sc, iv_idx, te_idx) in heads.items():
            out[f"iv_idx_{fold}"] = iv_idx
            out[f"iv_logits_{fold}"] = logits(model, sc, E[iv_idx])
            if not a.no_test:
                rows = np.flatnonzero(fold_all == fold)          # TAU: every device's rows of the fold
                test[rows] = logits(model, sc, E[rows])
            if a.task == "tau":
                ivloc = set(loc[iv_idx])
                rows = np.flatnonzero((meta.device.to_numpy() != "a") & np.isin(loc, list(ivloc)) & (fold_all != fold))
                out[f"ivs_idx_{fold}"] = rows
                out[f"ivs_logits_{fold}"] = logits(model, sc, E[rows])
        if not a.no_test:
            out["test_logits"] = test
        np.savez(sdir / f"{stem}__{cond}.npz", **out)
    for fs in summary["folds"]:
        fold = fs["fold"]
        model, sc, iv_idx, te_idx = heads[fold]
        z = np.load(sdir / f"{stem}__clean.npz")
        fs["inner_val_f1_clean_rescored"] = float(f1_score(lab_all[iv_idx], np.argmax(z[f"iv_logits_{fold}"], 1), average="macro"))
        if not a.no_test:
            te = te_idx if a.task != "tau" else te_idx
            fs["test_f1_clean"] = float(f1_score(lab_all[te], np.argmax(z["test_logits"][te], 1), average="macro"))
    summary["files_scored"] = [f.stem for f in files]
    summary["seconds"] = time.time() - t0
    (sdir / ("summary.json" if a.transfer is None else f"summary_rescore_{int(time.time())}.json")).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "files_scored"}, indent=1))


if __name__ == "__main__":
    main()
