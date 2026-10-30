"""Idea 7's ten audio statistics on the nine new corrupted conditions (idea 6 final round).

clip_stats is imported unchanged from idea 7 stage 1's audio_stats.py (path and SHA-256 are printed); the waveform is
load_clip + corrupt.corrupt, the one the extractors feed the networks. Before the new conditions, the first
--check clean rows are recomputed and compared with idea 7's stored clean.npz (maximum absolute difference printed,
the run stops above 1e-9). Output <out>/<condition>.npz with stats [N, 10], names, features, n_samples.

    python audio_stats_new.py --csv CSV --audio-dir AUDIO --i7 I7 --dataset esc50|urban --out OUT [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "stage2"))

import corrupt as CR  # noqa: E402
from audio_shift import index_wavs, load_clip  # noqa: E402


def load_i7(i7: Path):
    p = i7 / "scripts" / "audio_stats.py"
    spec = importlib.util.spec_from_file_location("i7_audio_stats", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--i7", type=Path, required=True)
    ap.add_argument("--dataset", required=True, choices=["esc50", "urban"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    mod, sha = load_i7(a.i7)
    print(f"clip_stats from {a.i7 / 'scripts' / 'audio_stats.py'} sha256 {sha}", flush=True)
    stems = pd.read_csv(a.csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
    if a.limit:
        stems = stems[:a.limit]
    index = index_wavs(a.audio_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    ref = np.load(a.i7 / "audio_stats" / a.dataset / "clean.npz", allow_pickle=True)
    if [str(s) for s in ref["names"]][:len(stems)] != stems:
        raise SystemExit("idea 7 clean statistics rows differ from the CSV")
    n_chk = min(a.check, len(stems))
    got = np.stack([mod.clip_stats(load_clip(index[stems[i]])) for i in range(n_chk)])
    d = float(np.abs(got - ref["stats"][:n_chk]).max())
    print(f"clean check on {n_chk} clips: max abs difference {d:.3e}", flush=True)
    if d > 1e-9:
        raise SystemExit("clean statistics differ from idea 7's stored ones")
    conds = [(k, float(lv)) for k in CR.KINDS for lv in CR.LEVELS[k]]
    S = {CR.cond_label(k, lv): np.full((len(stems), len(mod.FEATURES)), np.nan) for k, lv in conds}
    n_samples = np.zeros(len(stems), np.int64)
    t0 = time.time()
    for i, s in enumerate(stems):
        x = load_clip(index[s])
        n_samples[i] = len(x)
        for k, lv in conds:
            S[CR.cond_label(k, lv)][i] = mod.clip_stats(CR.corrupt(x, k, lv))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(stems)} {time.time() - t0:.0f}s", flush=True)
    for name, arr in S.items():
        if not np.isfinite(arr).all():
            raise SystemExit(f"{name}: non-finite statistics")
        np.savez(a.out / f"{name}.npz", stats=arr, names=np.array(stems), features=np.array(mod.FEATURES),
                 n_samples=n_samples)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
