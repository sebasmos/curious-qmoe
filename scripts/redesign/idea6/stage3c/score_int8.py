"""Score heads on the real-int8 embeddings (idea 6 Stage 3c, P4), in the ScoreDir format of Stage 2's score_eff.py.

Way a (--mode a): Stage 3b's heads (<s3b>/outputs/<model>_<task>_s<seed>/<kind>) with their scalers fitted on the
clean FP32 fit rows of the training CSV, as score_eff.py; every fold's head is first re-run on the training CSV's
test rows and must reproduce summary.json. Test rows come from <emb>/<model>_int8__<cond>.npy, fold f's
inner-validation rows from <emb>/<model>_int8_f<f>__clean.npy on clean (NaN on noisy conditions, never read).

Way b (--mode b): heads retrained per fold on fold f's int8 clean embeddings
(<runs>/<model>r_<task>_s<seed>_f<f>/<kind>, one fold each, benchmark.py with cross_validation.only_folds), scalers
fitted on that run's CSV's fit rows; split.json must equal the Stage 3b run's for that (task, seed, fold).

Output <out>/<model>[r]_<task>_s<seed>/<kind>/<model>_int8__<cond>.npz: y, test_fold, test_logits, iv_idx_<f>,
iv_logits_<f>, names_crc32.

    python scripts/redesign/idea6/stage3c/score_int8.py --mode a --model mn10 --task esc50 --s3b S3B --r R
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6" / "stage2"))

from score_eff import fold_scaler, logits_of, train_frame  # noqa: E402
from scripts.analysis.cascade import load_model  # noqa: E402

KINDS = ["esc", "8", "4", "bitnet"]
DS = {"esc50": "esc50", "esc5": "esc50", "esc5off": "esc50", "urban": "urban8k"}
SEEDS = {"esc50": [0, 1, 2, 3, 4], "esc5": [0, 1, 2, 3, 4], "esc5off": [0, 1, 2, 3, 4], "urban": [0, 1, 2]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["a", "b"], required=True)
    ap.add_argument("--model", default="mn10")
    ap.add_argument("--task", required=True, choices=list(DS))
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    ap.add_argument("--s3b", type=Path, required=True)
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    ds = DS[a.task]
    emb = a.r / "emb_int8" / ds
    files = sorted(p for p in emb.glob(f"{a.model}_int8__*.npy"))
    if not files:
        raise SystemExit(f"no int8 embeddings in {emb}")
    crc = json.loads(files[0].with_suffix(".json").read_text())["names_crc32"]
    conds = [p.stem.split("__", 1)[1] for p in files]
    X_cond = {c: np.load(p) for c, p in zip(conds, files)}
    folds_avail = sorted(int(p.stem.split("_f")[1].split("__")[0]) for p in emb.glob(f"{a.model}_int8_f*__clean.npy"))
    X_fold = {f: np.load(emb / f"{a.model}_int8_f{f}__clean.npy") for f in folds_avail}
    out_root = a.r / ("scores_int8a" if a.mode == "a" else "scores_int8b") / a.task
    for s in SEEDS[a.task]:
        for k in a.kinds:
            tag = f"{a.model}_{a.task}_s{s}" if a.mode == "a" else f"{a.model}r_{a.task}_s{s}"
            out_dir = out_root / tag / k
            if all((out_dir / f"{a.model}_int8__{c}.npz").exists() for c in conds):
                continue
            heads = {}
            if a.mode == "a":
                run = a.s3b / "outputs" / f"{a.model}_{a.task}_s{s}" / k
                cfg = OmegaConf.load(run / "config.yaml")
                Xc, y = train_frame(cfg, X_cond["clean"].shape[1])
                n_cls = len(np.unique(y))
                summary = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
                for fold in sorted(summary):
                    split = json.loads((run / f"fold_{fold}" / "split.json").read_text())
                    model, _ = load_model(run, fold, Xc.shape[1], n_cls)
                    te, iv = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
                    scaler = fold_scaler(Xc, y, split, cfg.experiment.datasets.esc.normalization_type)
                    f1 = f1_score(y[te], logits_of(model, scaler, Xc[te], n_cls).argmax(1), average="macro", zero_division=0)
                    if abs(f1 - summary[fold]["f1_macro"]) > 1e-6:
                        raise SystemExit(f"{run} fold {fold}: rescored F1 {f1:.6f} != summary {summary[fold]['f1_macro']:.6f}")
                    heads[fold] = (te, iv, scaler, model)
            else:
                y = None
                for fold in folds_avail:
                    run = a.r / "outputs_int8b" / f"{a.model}r_{a.task}_s{s}_f{fold}" / k
                    cfg = OmegaConf.load(run / "config.yaml")
                    Xc, y_ = train_frame(cfg, X_fold[fold].shape[1])
                    y = y_ if y is None else y
                    if not np.array_equal(y, y_):
                        raise SystemExit(f"{run}: labels differ between fold CSVs")
                    n_cls = len(np.unique(y))
                    summary = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
                    if list(summary) != [fold]:
                        raise SystemExit(f"{run}: folds {list(summary)} != [{fold}]")
                    split = json.loads((run / f"fold_{fold}" / "split.json").read_text())
                    ref = json.loads((a.s3b / "outputs" / f"{a.model}_{a.task}_s{s}" / k / f"fold_{fold}" / "split.json").read_text())
                    if split["test_idx"] != ref["test_idx"] or split["inner_val_idx"] != ref["inner_val_idx"]:
                        raise SystemExit(f"{run}: split differs from the Stage 3b run")
                    model, _ = load_model(run, fold, Xc.shape[1], n_cls)
                    te, iv = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
                    scaler = fold_scaler(Xc, y, split, cfg.experiment.datasets.esc.normalization_type)
                    f1 = f1_score(y[te], logits_of(model, scaler, Xc[te], n_cls).argmax(1), average="macro", zero_division=0)
                    if abs(f1 - summary[fold]["f1_macro"]) > 1e-6:
                        raise SystemExit(f"{run} fold {fold}: rescored F1 {f1:.6f} != summary {summary[fold]['f1_macro']:.6f}")
                    heads[fold] = (te, iv, scaler, model)
            out_dir.mkdir(parents=True, exist_ok=True)
            for c in conds:
                X = X_cond[c]
                res = {"y": y, "test_fold": np.full(len(y), -1, np.int16),
                       "test_logits": np.full((len(y), n_cls), np.nan, np.float32), "names_crc32": np.int64(crc)}
                for fold, (te, iv, scaler, model) in heads.items():
                    res["test_logits"][te] = logits_of(model, scaler, X[te], n_cls)
                    res["test_fold"][te] = fold
                    res[f"iv_idx_{fold}"] = iv
                    res[f"iv_logits_{fold}"] = (logits_of(model, scaler, X_fold[fold][iv], n_cls) if c == "clean"
                                                else np.full((len(iv), n_cls), np.nan, np.float32))
                np.savez_compressed(out_dir / f"{a.model}_int8__{c}.npz", **res)
            print(f"{tag}/{k}: scored {len(conds)} conditions", flush=True)


if __name__ == "__main__":
    main()
