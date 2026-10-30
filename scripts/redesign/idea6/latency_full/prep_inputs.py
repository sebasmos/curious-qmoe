"""Inputs shared by both envs (run once, curious-qmoe env, before any timing).

Writes cache/mn_mel_<ds>.npy, the EfficientAT log mel [1, 1, 128, frames] of
each dataset's timing clip (ESC-50 1-100032-A-0, 5 s; UrbanSound8K
100263-2-0-117, 4 s zero-padded to 4.04 s as in Stage 3b), and
cache/mn_emb_<model>_<ds>_torch.npy, the torch FP32 embedding of mn10 and
mn20 on it, which time_ort.py checks its FP32 sessions against. The EfficientNet
timing image is the first study's cache/timing_esc50_u8.npy (300 x 300 for every
clip). cache/inputs.json records clip names, samples and frames.

    OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
        python scripts/prep_inputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    torch.set_num_threads(2)
    (C.OUT / "cache").mkdir(exist_ok=True)
    info = {"load_start": C.loadavg()}
    models = {n: C.mn_fp32(n) for n in C.MN_FILES}
    for ds in ("esc50", "urban8k"):
        nm, wave = C.timing_clip(ds)
        w = C.mn_wave(ds, wave)
        with torch.no_grad():
            x = C.mn_input(w)
            np.save(C.OUT / f"cache/mn_mel_{ds}.npy", x.numpy().astype(np.float32))
            for n, m in models.items():
                np.save(C.OUT / f"cache/mn_emb_{n}_{ds}_torch.npy", m(x).numpy().astype(np.float32))
        info[f"clip_{ds}"] = {"name": nm, "samples_16k_raw": int(len(wave)), "samples_16k_to_mn": int(len(w)),
                              "mn_mel_shape": list(x.shape)}
        print(ds, nm, len(wave), len(w), tuple(x.shape), flush=True)
    info["load_end"] = C.loadavg()
    (C.OUT / "cache/inputs.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
