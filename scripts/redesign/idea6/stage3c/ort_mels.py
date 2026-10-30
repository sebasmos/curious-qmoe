"""Log-mel inputs of every clip for the real-int8 study (idea 6 Stage 3c, P4): the same audio, noise draws and
front end as extract_cnn.py (Stage 2's audio_shift, QWave.efficientat_mn.log_mel), written once per condition as a
float16 array so the ONNX Runtime env (which has no torchaudio) can read them.

Per condition: <out>/<cond>.npy float16 [N, 128, frames] in the CSV's row order (every clip padded to
--pad-to-samples when given, as the W8A8 extraction), <out>/<cond>.json (names checksum, lengths, pad), and
bg_plan.json for the background draws (must equal the extraction's).

    python scripts/redesign/idea6/stage3c/ort_mels.py --csv ... --audio-dir ... [--bg-audio-dir ... --bg-csv ...] \
        --out OUT --conditions clean bg:10 ... [--pad-to-samples 64587]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--bg-audio-dir", type=Path, default=None)
    p.add_argument("--bg-csv", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--pad-to-samples", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--chunk", type=int, default=256)
    return p.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(ARGS.threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extract_cnn as X  # noqa: E402  (Stage 3c's extractor: Audio, parse_condition, the stage 2 audio imports)
from QWave import efficientat_mn as M  # noqa: E402
from audio_shift import NoiseMaker, cond_name, index_wavs  # noqa: E402


def main():
    a = ARGS
    torch.set_num_threads(a.threads)
    meta = pd.read_csv(a.csv, usecols=["folder", "name", "class_id"])
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    n = len(stems)
    crc = zlib.crc32("\n".join(stems).encode())
    index = index_wavs(a.audio_dir)
    plan = [X.parse_condition(c) for c in a.conditions]
    noise_maker = NoiseMaker([], {}, False)
    if any(k == "bg" for k, _ in plan):
        bg_names = pd.read_csv(a.bg_csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(a.bg_audio_dir)
        noise_maker = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)
    audio = X.Audio(stems, index, noise_maker)
    a.out.mkdir(parents=True, exist_ok=True)
    for kind, snr in plan:
        cname = cond_name(kind, snr)
        if (a.out / f"{cname}.npy").exists():
            continue
        t0 = time.time()
        mel, lens = None, np.zeros(n, np.int64)
        for s in range(0, n, a.chunk):
            idx = list(range(s, min(n, s + a.chunk)))
            waves = [audio.get(i, kind, snr) for i in idx]
            lens[idx] = [len(w) for w in waves]
            if a.pad_to_samples:
                waves = [np.pad(w, (0, max(0, a.pad_to_samples - len(w)))) for w in waves]
            with torch.no_grad():
                groups = M.model_input(waves)
            if len(groups) != 1:
                raise SystemExit(f"{cname}: {len(groups)} input lengths in one chunk; pad to one length")
            x = groups[0][1][:, 0].numpy().astype(np.float16)
            if mel is None:
                mel = np.zeros((n, x.shape[1], x.shape[2]), np.float16)
            mel[idx] = x
            print(f"  {cname}: {idx[-1] + 1}/{n}, {time.time() - t0:.0f}s", flush=True)
        np.save(a.out / f"_{cname}.tmp.npy", mel)
        os.replace(a.out / f"_{cname}.tmp.npy", a.out / f"{cname}.npy")
        (a.out / f"{cname}.json").write_text(json.dumps({
            "condition": cname, "n_clips": n, "names_crc32": crc, "pad_to_samples": a.pad_to_samples,
            "frames": int(mel.shape[2]), "dtype": "float16", "lengths_min": int(lens.min()),
            "lengths_max": int(lens.max()), "silent_clips_left_clean": audio.n_silent, "seconds": time.time() - t0,
            "frontend": {"sr_in": M.SR_IN, "sr": M.SR, "n_fft": M.N_FFT, "win": M.WIN, "hop": M.HOP,
                         "n_mels": M.N_MELS, "f_min": M.F_MIN, "f_max": M.F_MAX}}, indent=1))
        np.save(a.out / f"lengths__{cname}.npy", lens)
        if kind == "bg":
            planf = a.out / "bg_plan.json"
            old = json.loads(planf.read_text()) if planf.exists() else {}
            old.update(noise_maker.plan)
            planf.write_text(json.dumps(old, indent=0))
        print(f"{cname}: {n} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
