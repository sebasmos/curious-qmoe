"""Check that the cheap tier saw the AST tier's audio.

1. End to end: for a few clips per condition, build the waveform exactly as
   extract_eff.py does (its Audio class, same CSVs and folders), run the FP32
   AST (MIT/ast-finetuned-audioset-10-10-0.4593 at f826b80, pooler_output,
   CPU) and compare with idea 5 Stage 2's stored fp32__<condition>.npy rows.
2. Every background draw in this stage's bg_plan.json equals Stage 2's.

    python scripts/redesign/idea6/stage2/check_same_audio.py --stage2-emb STAGE2/emb --emb EMB --out check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))

from audio_shift import NoiseMaker, cond_name, index_wavs  # noqa: E402
from extract_eff import Audio, parse_condition  # noqa: E402

E = Path.home() / "Documents" / "DATASETS"
DATA = {
    "esc50": dict(csv=E / "ESC-50/ast_768/esc-50.csv", audio=E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio",
                  bg_csv=E / "Urban8k/ast_768/urbansound8k.csv",
                  bg_audio=E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio",
                  conds=["clean", "bg:10", "pink:5", "white:0"]),
    "urban8k": dict(csv=E / "Urban8k/ast_768/urbansound8k.csv",
                    audio=E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio",
                    bg_csv=E / "ESC-50/ast_768/esc-50.csv",
                    bg_audio=E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio",
                    conds=["clean", "bg:10", "bg:0"]),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage2-emb", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--n-clips", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    from transformers import ASTFeatureExtractor, ASTModel

    name, rev = "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"
    fe = ASTFeatureExtractor.from_pretrained(name, revision=rev)
    model = ASTModel.from_pretrained(name, revision=rev).eval()
    res = {}
    for ds, d in DATA.items():
        stems = pd.read_csv(d["csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_names = pd.read_csv(d["bg_csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(d["bg_audio"])
        audio = Audio(stems, index_wavs(d["audio"]), NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True))
        rows = np.sort(np.random.default_rng(0).choice(len(stems), a.n_clips, replace=False))
        for c in d["conds"]:
            kind, snr = parse_condition(c)
            waves = [audio.get(int(i), kind, snr) for i in rows]
            with torch.no_grad():
                emb = model(**fe(waves, sampling_rate=16000, return_tensors="pt")).pooler_output.numpy()
            ref = np.load(a.stage2_emb / ds / f"fp32__{cond_name(kind, snr)}.npy", mmap_mode="r")[rows]
            diff = float(np.abs(emb - ref).max())
            rel = float((np.linalg.norm(emb - ref, axis=1) / np.linalg.norm(ref, axis=1)).max())
            res[f"{ds}/{cond_name(kind, snr)}"] = {"rows": rows.tolist(), "max_abs_diff": diff, "max_rel_diff": rel}
            print(ds, c, f"max abs diff {diff:.2e}, max relative {rel:.2e}", flush=True)
        mine, theirs = a.emb / ds / "bg_plan.json", a.stage2_emb / ds / "bg_plan.json"
        if mine.exists():
            p, q = json.loads(mine.read_text()), json.loads(theirs.read_text())
            same = sum(p[k] == q.get(k) for k in p)
            res[f"{ds}/bg_plan"] = {"clips": len(p), "equal_to_stage2": same, "stage2_clips": len(q)}
            print(ds, f"bg_plan: {same} of {len(p)} draws equal to Stage 2's ({len(q)} there)", flush=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
