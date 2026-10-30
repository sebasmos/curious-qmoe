"""Per-clip audio statistics for the check 8 baseline gates, in the embedding CSV row order:
RMS loudness (dBFS, over the whole clip), active RMS (dBFS over frames above -60 dBFS), and mean
spectral flatness (geometric over arithmetic mean of the power spectrum, 2048-point frames, hop 512).
Mono mix of the file at its own rate.   python audio_stats.py OUT.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
H = Path.home() / "Documents/DATASETS"
rows = []
for ds, csv, root in [("esc50", H / "ESC-50/efficientnet_1536/esc-50.csv", H / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"),
                      ("urban", H / "Urban8k/efficientnet/urbansound8k.csv", H / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio")]:
    meta = pd.read_csv(csv, usecols=["folder", "name"])
    for i, (fo, nm) in enumerate(zip(meta.folder, meta.name)):
        stem = Path(nm).stem
        p = root / f"{stem}.wav" if ds == "esc50" else root / f"fold{fo}" / f"{stem}.wav"
        x, sr = sf.read(p, dtype="float64", always_2d=True)
        x = x.mean(1)
        rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        n = 2048
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        fr = np.lib.stride_tricks.sliding_window_view(x, n)[::512] * np.hanning(n)
        frms = np.sqrt((fr ** 2).mean(1)) + 1e-12
        act = frms[20 * np.log10(frms) > -60]
        P = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-20
        flat = np.exp(np.log(P).mean(1)) / P.mean(1)
        rows.append({"dataset": ds, "row": i, "name": stem, "sr": sr, "seconds": len(x) / sr,
                     "rms_db": 20 * np.log10(rms), "active_rms_db": 20 * np.log10(np.sqrt((act ** 2).mean())) if len(act) else -120.0,
                     "flatness": float(flat.mean())})
    print(ds, "done", flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
