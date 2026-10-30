"""Check 5, latency view: wall clock of the two front ends on one real ESC-50 clip, CPU, 2 threads.

EfficientNet front end as in the rendering script found in git history (c26abba,
scripts/pre-processing/esc-50_processing/esc50_dataset.py): native 44.1 kHz, STFT n_fft 2048,
hop 32, Hann, 128 Slaney mel bands 0 to 20 kHz, power_to_db(ref=max, top_db=80), a 7 x 7 inch
figure at 300 dpi with the axes off, pcolormesh in magma, saved as PNG with a tight box; then the
extractor's PIL open, RGB, Resize((300, 300)), ToTensor. librosa is not installed here, so the STFT
and the filterbank are written in numpy (same sizes; timing only, no pixel equality is claimed).
AST front end: torchaudio Kaldi fbank (16 kHz, 25 / 10 ms, 128 bins) after resampling, padded to
1024 frames and normalised as the AST feature extractor does.

    python check5_frontend_time.py OUT.json
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

torch.set_num_threads(2)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

CLIP = Path.home() / "Documents/DATASETS/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio/1-100032-A-0.wav"


def hz_to_mel(f):  # Slaney
    f = np.asarray(f, float)
    lin = f / (200.0 / 3)
    return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1e-9) / 1000) / (np.log(6.4) / 27), lin)


def mel_to_hz(m):
    m = np.asarray(m, float)
    return np.where(m >= 15, 1000 * np.exp((np.log(6.4) / 27) * (m - 15)), (200.0 / 3) * m)


def mel_fb(sr, n_fft, n_mels, fmin, fmax):
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0, np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)))
        fb[i] *= 2.0 / (hi - lo)
    return fb


def eff_front(x, sr, fb):
    t = {}
    t0 = time.perf_counter()
    xp = np.pad(x, 1024, mode="constant")
    fr = np.lib.stride_tricks.sliding_window_view(xp, 2048)[::32] * np.hanning(2048)
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    M = fb @ S.T
    db = 10 * np.log10(np.maximum(M, 1e-10)) - 10 * np.log10(M.max())
    db = np.maximum(db, -80)
    t["stft_mel_db_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    fig = plt.figure(figsize=[7, 7])
    ax = plt.gca()
    ax.set_axis_off()
    ax.pcolormesh(db, cmap="magma", shading="auto")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    t["render_png_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    size = img.size
    x_t = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])(img)
    t["decode_resize_s"] = time.perf_counter() - t0
    t["frames"] = int(S.shape[0])
    t["png_size"] = list(size)
    return t


def ast_front(x, sr):
    import torchaudio
    t0 = time.perf_counter()
    w = torchaudio.functional.resample(torch.tensor(x, dtype=torch.float32)[None], sr, 16000)
    fb = torchaudio.compliance.kaldi.fbank(w, htk_compat=True, sample_frequency=16000, use_energy=False,
                                           window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
    fb = torch.nn.functional.pad(fb, (0, 0, 0, 1024 - fb.shape[0]))
    fb = (fb + 4.2677393) / (4.5689974 * 2)
    return {"fbank_s": time.perf_counter() - t0, "frames_real": int(w.shape[1] // 160)}


def main(out):
    x, sr = sf.read(CLIP, dtype="float64")
    fb = mel_fb(sr, 2048, 128, 0, 20000)
    eff = [eff_front(x, sr, fb) for _ in range(4)][1:]
    ast = [ast_front(x, sr) for _ in range(4)][1:]
    res = {"clip": str(CLIP), "sr": sr, "efficientnet_front_end": eff, "ast_front_end": ast,
           "eff_total_median_s": float(np.median([sum(v for k, v in e.items() if k.endswith("_s")) for e in eff])),
           "ast_total_median_s": float(np.median([a["fbank_s"] for a in ast]))}
    print(json.dumps(res, indent=1))
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
