"""Check 3b: the stored n3g64 embeddings (extracted on MPS) differ from a CPU re-run by 0.25
relative while every other configuration differs by about 0.05. Hypothesis: torch.round on
MPS breaks the exact .5 ties of the nested child (c8 / 32 with c8 = 32k + 16, about 3% of
weights) away from zero, while CPU rounds them to even. Re-run n3g64 on CPU with both tie
rules (and n4 / n4g128, whose ties are c8 = 16k + 8) and compare with the stored arrays.

    python check3b_ties.py --stage2 STAGE2_DIR --repo WORKTREE --out OUT [--n-clips 16]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stage2", type=Path, required=True)
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n-clips", type=int, default=16)
args = ap.parse_args()
sys.path.insert(0, str(args.repo))
sys.path.insert(0, str(args.repo / "scripts/redesign/idea5"))
torch.set_num_threads(2)
import QWave.backbone_quant as bq  # noqa: E402
from audio_shift import index_wavs, load_clip  # noqa: E402
from transformers import ASTFeatureExtractor, ASTModel  # noqa: E402

MODEL, REV = "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"
model = ASTModel.from_pretrained(MODEL, revision=REV).eval()
meta = pd.read_csv(Path.home() / "Documents/DATASETS/ESC-50/ast_768/esc-50.csv", usecols=["name"])
stems = meta["name"].map(lambda s: Path(s).stem).tolist()
index = index_wavs(Path.home() / "Documents/DATASETS/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio")
rows = np.sort(np.random.default_rng(0).choice(len(stems), args.n_clips, replace=False))
fe = ASTFeatureExtractor.from_pretrained(MODEL, revision=REV)
feats = fe([load_clip(index[stems[i]]) for i in rows], sampling_rate=16000, return_tensors="pt")["input_values"]
bq.USE_METAL = False
qast = bq.QuantAST(model)
orig = bq.nested_codes


def away(parent, k, parent_bits=8):
    step = 2 ** (parent_bits - k)
    x = parent / step
    return torch.clamp(torch.sign(x) * torch.floor(x.abs() + 0.5), -(2 ** (k - 1)), 2 ** (k - 1) - 1)


def rel(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)))


res = {}
for cfg in ["n3g64", "n4", "n4g128"]:
    stored = np.load(args.stage2 / "emb/esc50" / f"{cfg}__clean.npy")[rows]
    for rule, fn in [("ties_to_even", orig), ("ties_away", away)]:
        bq.nested_codes = fn
        qast.set_config(*bq.parse_config(cfg))
        e = torch.cat([qast(feats[b:b + 4]) for b in range(0, len(feats), 4)]).numpy()
        res[f"{cfg}_{rule}_vs_stored"] = rel(e, stored)
        print(cfg, rule, res[f"{cfg}_{rule}_vs_stored"], flush=True)
bq.nested_codes = orig
(args.out / "check3b_ties.json").write_text(json.dumps(res, indent=1))
