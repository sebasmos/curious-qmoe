"""Per-clip audio statistics for idea 7's model-free noise detectors, on CPU.

Every clip is read and corrupted with Stage 2's own code
(scripts/redesign/idea5/audio_shift.py: same reader, resampling to 16 kHz,
per-clip seeded white, pink and background noise, RMS mixing), so the
waveform here is the one the AST feature extractor saw in Stage 2. The
background draws are checked against Stage 2's emb/<dataset>/bg_plan.json.

Statistics (25 ms Hann frames, 10 ms hop, 512-point FFT, power spectrum bins
1 to 256), all computed from the waveform alone, nothing from any model:
  level_db        20 log10 RMS of the clip
  snr_est_db      blind SNR estimate: 10 log10 of the mean of the loudest 10%
                  of frame energies over the mean of the quietest 10%
  floor_db        10th-percentile frame energy over the mean frame energy, dB
  flatness_db     median over frames of spectral flatness (geometric over
                  arithmetic mean of the power spectrum), dB
  ltas_flat_db    spectral flatness of the long-term average spectrum, dB
  zcr             zero crossings per sample
  centroid_khz    mean power-weighted spectral centroid
  rolloff_khz     median 85% spectral roll-off
  hf_ratio_db     energy above 4 kHz over total energy, dB
  crest_db        peak over RMS, dB
plus n_samples (16 kHz samples, for the bits sent to the cloud).

Output: <out>/<condition>.npz with stats [N, 10] (NaN for rows not computed,
in field cells extracted only for the clips the field assignments give
them), names, feature names, n_samples; and bg_check.json.

    python scripts/redesign/idea7/audio_stats.py --csv CSV --audio-dir AUDIO --bg-csv OTHER_CSV \
        --bg-audio-dir OTHER_AUDIO --conditions clean bg:10 bg:0 --field --bg-plan EMB/bg_plan.json --out OUT
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea5"))

from audio_shift import (FIELD_CELLS, FIELD_SEEDS, NoiseMaker, cond_name, field_assignment,  # noqa: E402
                         index_wavs, load_clip, mix)

FEATURES = ["level_db", "snr_est_db", "floor_db", "flatness_db", "ltas_flat_db", "zcr", "centroid_khz",
            "rolloff_khz", "hf_ratio_db", "crest_db"]
SR, WIN, HOP, NFFT = 16000, 400, 160, 512
EPS = 1e-12


def clip_stats(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n_fr = 1 + (len(x) - WIN) // HOP
    fr = np.lib.stride_tricks.as_strided(x, (n_fr, WIN), (x.strides[0] * HOP, x.strides[0]))
    win = np.hanning(WIN)
    P = np.abs(np.fft.rfft(fr * win, n=NFFT, axis=1)) ** 2
    P = P[:, 1:]  # drop DC
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)[1:]
    e = np.mean(fr ** 2, axis=1) + EPS
    es = np.sort(e)
    k = max(1, int(round(0.1 * len(es))))
    rms = np.sqrt(np.mean(x ** 2)) + EPS
    tot = P.sum(1) + EPS
    flat = np.exp(np.mean(np.log(P + EPS), 1)) / (np.mean(P, 1) + EPS)
    ltas = P.mean(0) + EPS
    cum = np.cumsum(P, 1) / tot[:, None]
    roll = freqs[np.minimum((cum < 0.85).sum(1), len(freqs) - 1)]
    return np.array([
        20 * np.log10(rms),
        10 * np.log10(es[-k:].mean() / es[:k].mean()),
        10 * np.log10(np.percentile(e, 10) / e.mean()),
        10 * np.log10(np.median(flat) + EPS),
        10 * np.log10(np.exp(np.mean(np.log(ltas))) / ltas.mean()),
        float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8))))),
        float(np.mean((P * freqs).sum(1) / tot)) / 1000,
        float(np.median(roll)) / 1000,
        10 * np.log10(P[:, freqs > 4000].sum() / tot.sum()),
        20 * np.log10(np.max(np.abs(x)) / rms + EPS),
    ])


def parse_condition(c):
    if c == "clean":
        return "clean", None
    kind, snr = c.split(":")
    return kind, float(snr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--bg-csv", type=Path, required=True)
    ap.add_argument("--bg-audio-dir", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--field", action="store_true")
    ap.add_argument("--bg-plan", type=Path, default=None, help="Stage 2 bg_plan.json to check the draws against")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.csv, usecols=["folder", "name", "class_id"])
    if args.limit:
        meta = meta.iloc[:args.limit]
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    n = len(stems)
    index = index_wavs(args.audio_dir)
    bg_names = pd.read_csv(args.bg_csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
    bg_index = index_wavs(args.bg_audio_dir)
    noise = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)

    # rows per condition: full conditions get every row; field cells not asked for in full get the rows the
    # test or validation assignment gives them (Stage 2's extract_stage2.plan_conditions)
    plan = {cond_name(*parse_condition(c)): (parse_condition(c), np.arange(n)) for c in args.conditions}
    if args.field:
        assign = [np.array(field_assignment(n, s)) for s in FIELD_SEEDS.values()]
        for kind, snr in FIELD_CELLS:
            name = cond_name(kind, snr)
            if name not in plan:
                plan[name] = ((kind, float(snr)), np.flatnonzero((assign[0] == name) | (assign[1] == name)))
    need = {i: [] for i in range(n)}
    for name, ((kind, snr), rows) in plan.items():
        for i in rows:
            need[int(i)].append((name, kind, snr))

    S = {name: np.full((n, len(FEATURES)), np.nan) for name in plan}
    n_samples = np.zeros(n, np.int64)
    silent = {name: 0 for name in plan}
    t0 = time.time()
    for i in range(n):
        x = load_clip(index[stems[i]])
        n_samples[i] = len(x)
        noises = {}
        for name, kind, snr in need[i]:
            if kind == "clean":
                y = x
            else:
                if kind not in noises:
                    noises[kind] = noise(stems[i], kind, len(x))
                y, mixed = mix(x, noises[kind], snr)
                silent[name] += not mixed
            S[name][i] = clip_stats(y)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} clips, {time.time() - t0:.0f}s", flush=True)

    for name, ((kind, snr), rows) in plan.items():
        np.savez(args.out / f"{name}.npz", stats=S[name], rows=rows, names=np.array(stems),
                 features=np.array(FEATURES), n_samples=n_samples)
    check = {"n_clips": n, "silent_left_clean": silent, "seconds": time.time() - t0,
             "rows": {k: int(len(v[1])) for k, v in plan.items()}}
    if args.bg_plan is not None and noise.plan:
        ref = json.loads(args.bg_plan.read_text())
        same = [noise.plan[s] == ref.get(s) for s in noise.plan]
        check["bg_plan_checked"] = len(same)
        check["bg_plan_equal"] = int(sum(same))
    (args.out / "bg_check.json").write_text(json.dumps(check, indent=1))
    print(json.dumps(check), flush=True)


if __name__ == "__main__":
    main()
