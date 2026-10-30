"""Score the pick's heads on the real-int8 embeddings with the scaler refit on int8 features (idea 6 option B, S4,
the Stage 3c audit's way): fold f's head weights unchanged, its StandardScaler refit on fold f's int8 embeddings of
the fold's fit rows (device-A training rows minus inner validation), applied to fold f's rows of every device and
to its inner-validation rows (from <model>_int8_f<f>__clean.npy). Also the Stage 3c "way a" (the FP32-fitted
scaler) for the record.

Output: <r>/scores_int8/<model>_tau_s<seed>/<kind>/<model>_int8__all.npz and <model>_int8a__all.npz in the
score_optb.py layout (y, fold, device, test_logits, iv_idx_<f>, iv_logits_<f>, n_a, names_crc32).

    python scripts/redesign/idea6/optionB/score_int8_optb.py --model mn10 --kinds 4 --r R --meta CSV
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6" / "stage2"))

from score_eff import fold_scaler, logits_of, train_frame  # noqa: E402
from scripts.analysis.cascade import load_model  # noqa: E402

SEEDS = [0, 1, 2]


def scaler_on(X, y, rows, norm):
    from QWave.datasets import EmbeddingAdaptDataset

    df = pd.DataFrame(X[rows], columns=[str(i) for i in range(X.shape[1])])
    df["class_id"] = y[rows]
    with contextlib.redirect_stdout(io.StringIO()):
        return EmbeddingAdaptDataset(df, normalization_type=norm).get_scaler()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--kinds", nargs="+", required=True)
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    meta = pd.read_csv(a.meta)
    crc = zlib.crc32("\n".join(meta["name"].map(lambda s: Path(s).stem)).encode())
    y_u, fold_u, dev_u = meta.class_id.to_numpy(np.int64), meta.folder.to_numpy(np.int16), meta.device.to_numpy().astype(str)
    n_a = int((meta.device == "a").sum())
    ed = a.r / "emb_int8" / "tau"
    folds = sorted(int(p.stem.split("_f")[1].split("__")[0]) for p in ed.glob(f"{a.model}_int8_f*__clean.npy"))
    XF = {f: np.load(ed / f"{a.model}_int8_f{f}__clean.npy") for f in folds}
    if json.loads((ed / f"{a.model}_int8__clean.json").read_text())["names_crc32"] != crc:
        raise SystemExit("int8 sidecar names checksum differs from the universe CSV")
    for s in SEEDS:
        for k in a.kinds:
            run = a.r / "outputs" / f"{a.model}_tau_s{s}" / k
            out_dir = a.r / "scores_int8" / f"{a.model}_tau_s{s}" / k
            if (out_dir / f"{a.model}_int8__all.npz").exists() and (out_dir / f"{a.model}_int8a__all.npz").exists():
                continue
            cfg = OmegaConf.load(run / "config.yaml")
            Xc, y = train_frame(cfg, XF[folds[0]].shape[1])
            if len(Xc) != n_a or not np.array_equal(y, y_u[:n_a]):
                raise SystemExit(f"{run}: training CSV is not the device-A block")
            n_cls = len(np.unique(y))
            norm = cfg.experiment.datasets.esc.normalization_type
            summary = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
            res = {w: {"y": y_u, "fold": fold_u, "device": dev_u, "n_a": np.int64(n_a), "names_crc32": np.int64(crc),
                       "test_logits": np.full((len(meta), n_cls), np.nan, np.float32)} for w in ("int8", "int8a")}
            for f in folds:
                split = json.loads((run / f"fold_{f}" / "split.json").read_text())
                te, iv = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
                model, _ = load_model(run, f, Xc.shape[1], n_cls)
                sc32 = fold_scaler(Xc, y, split, norm)
                f1 = f1_score(y[te], logits_of(model, sc32, Xc[te], n_cls).argmax(1), average="macro", zero_division=0)
                if abs(f1 - summary[f]["f1_macro"]) > 1e-6:
                    raise SystemExit(f"{run} fold {f}: rescored F1 {f1:.6f} != summary {summary[f]['f1_macro']:.6f}")
                fit = np.setdiff1d(np.arange(n_a), np.concatenate([te, iv]))
                sc8 = scaler_on(XF[f], y_u, fit, norm)
                rows = np.flatnonzero(fold_u == f)
                for w, sc in (("int8", sc8), ("int8a", sc32)):
                    res[w]["test_logits"][rows] = logits_of(model, sc, XF[f][rows], n_cls)
                    res[w][f"iv_idx_{f}"], res[w][f"iv_logits_{f}"] = iv, logits_of(model, sc, XF[f][iv], n_cls)
            out_dir.mkdir(parents=True, exist_ok=True)
            for w in res:
                if not np.isfinite(res[w]["test_logits"]).all():
                    raise SystemExit(f"{run} {w}: rows left unscored")
                np.savez_compressed(out_dir / f"{a.model}_{w}__all.npz", **res[w])
            print(f"{run.parent.name}/{k}: int8 (scaler refit) and int8a (FP32 scaler) scored", flush=True)


if __name__ == "__main__":
    main()
