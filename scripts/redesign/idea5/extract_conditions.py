"""AST clip embeddings under corrupted audio and fake-quantized backbones.

Idea 5, Stage 1. For every clip of an embedding CSV (rows kept in the CSV's
order) this writes the AST pooler_output for each requested condition, one
float32 .npy per condition, plus a JSON sidecar with timing and settings.

A condition is BACKBONE:NOISE:SNR, for example fp32:clean:-, w4:bg:10,
w4a8:white:0.

Audio. Each clip is read exactly as data/audio_embeddings/extract_ast_embeddings.py
reads it (mono downmix, DC removed, resampled to 16 kHz), then mixed with noise
at an RMS-based SNR over the whole clip:
    y = x + g * n,   g = rms(x) / (rms(n) * 10 ** (snr / 20))
Noise is deterministic per clip: its generator is seeded with
crc32("<clip name>|<noise kind>"), so a clip gets the same noise at every SNR
and under every backbone, and only the gain changes.
  white  standard normal samples
  pink   standard normal samples shaped to a 1/f power spectrum (amplitude
         divided by sqrt(f) in the rFFT domain, DC bin set to the first bin)
  bg     a real recording: one clip drawn from --bg-csv (seeded per clip),
         read the same way, looped if shorter than the clip, then a seeded
         offset window of the clip's length; a window with zero RMS is redrawn
Noise is centred to zero mean before scaling. A clip with zero RMS stays clean.

Backbone. Every nn.Linear inside the transformer encoder (query, key, value,
attention output, intermediate and output dense, 72 layers) gets its weight
replaced by symmetric round-to-nearest per output channel:
    s = max|W_row| / (2^(b-1) - 1),   W_q = clamp(round(W / s)) * s
(b = 2 gives ternary levels {-s, 0, s}). Kept in FP32: the patch-embedding
convolution, the [CLS]/[DIST] tokens, position embeddings, every LayerNorm,
every bias, the attention score and attention-times-value matmuls, softmax and
GELU. The w4a8 backbone also fake-quantizes the input of every encoder Linear
per token, symmetric 8-bit with a dynamic scale max|x_token| / 127.

Bit operations per clip are counted with forward hooks on a real forward pass:
MACs x weight bits x activation bits, Linear layers at (b, 32) or (4, 8), the
patch convolution and the two attention matmuls per layer at (32, 32).

    python scripts/redesign/idea5/extract_conditions.py --csv CLEAN_AST_CSV \
        --audio-dir AUDIO --bg-audio-dir OTHER_AUDIO --bg-csv OTHER_CSV \
        --out-dir OUT --conditions fp32:clean:- w4:bg:10 [--limit 64]
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
    p.add_argument("--csv", type=Path, required=True, help="clean AST CSV: rows, order and names")
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--bg-audio-dir", type=Path, default=None, help="audio searched for background clips")
    p.add_argument("--bg-csv", type=Path, default=None, help="CSV whose names are the background pool")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=None, help="first N rows only (sanity checks)")
    p.add_argument("--model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--revision", default="f826b80d28226b62986cc218e5cec390b1096902")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cache-audio", action="store_true", help="keep clean and background waves in memory")
    p.add_argument("--cache-bg", action="store_true", help="keep background waves in memory")
    return p.parse_args(argv)


ARGS = parse_args()
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = str(ARGS.threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402
from torch import nn  # noqa: E402
from transformers import ASTFeatureExtractor, ASTModel  # noqa: E402

torch.set_num_threads(ARGS.threads)
TARGET_SR = 16000
BACKBONES = {"fp32": (32, 32), "w8": (8, 32), "w4": (4, 32), "w3": (3, 32), "w2": (2, 32), "w4a8": (4, 8)}
NOISES = ("clean", "white", "pink", "bg")


# ── audio (same reading as extract_ast_embeddings.load_clip) ──────────────

def index_wavs(audio_dir):
    index = {}
    for path in sorted(audio_dir.rglob("*.wav")):
        if path.stem in index:
            raise SystemExit(f"duplicate clip name {path.stem}: {index[path.stem]} and {path}")
        index[path.stem] = path
    return index


def load_clip(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.mean(axis=1))
    wav = wav - wav.mean()
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    return wav.numpy()


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def clip_seed(name, kind):
    return zlib.crc32(f"{name}|{kind}".encode())


def pink(rng, n):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = f[1] if len(f) > 1 else 1.0
    return np.fft.irfft(spec / np.sqrt(f), n=n)


class NoiseMaker:
    def __init__(self, bg_names, bg_index, cache):
        self.bg_names, self.bg_index, self.cache = bg_names, bg_index, cache
        self.bg_cache, self.plan = {}, {}

    def bg_wave(self, j):
        if j not in self.bg_cache:
            w = load_clip(self.bg_index[self.bg_names[j]])
            if not self.cache:
                return w
            self.bg_cache[j] = w
        return self.bg_cache[j]

    def __call__(self, name, kind, n):
        rng = np.random.default_rng(clip_seed(name, kind))
        if kind == "white":
            noise = rng.standard_normal(n)
        elif kind == "pink":
            noise = pink(rng, n)
        elif kind == "bg":
            for _ in range(100):
                j = int(rng.integers(len(self.bg_names)))
                w = self.bg_wave(j)
                if len(w) < n:
                    w = np.tile(w, int(np.ceil(n / len(w))))
                off = int(rng.integers(len(w) - n + 1))
                noise = w[off:off + n].astype(np.float64)
                if rms(noise - noise.mean()) > 0:
                    break
            else:
                raise RuntimeError(f"{name}: no non-silent background window in 100 draws")
            self.plan[name] = {"bg_clip": self.bg_names[j], "offset_16k": off}
        else:
            raise ValueError(kind)
        return noise - noise.mean()


def mix(x, noise, snr):
    rx = rms(x)
    if rx == 0:
        return x, False
    g = rx / (rms(noise) * 10 ** (snr / 20))
    return (x.astype(np.float64) + g * noise).astype(np.float32), True


# ── backbone fake quantization and cost ────────────────────────────────────

def quantize_weight_per_channel(w, bits):
    qmax = 2 ** (bits - 1) - 1
    s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    return torch.clamp(torch.round(w / s), -qmax, qmax) * s


def act_quant_hook(bits):
    qmax = 2 ** (bits - 1) - 1

    def hook(module, inputs):
        x = inputs[0]
        s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
        return (torch.clamp(torch.round(x / s), -qmax, qmax) * s,)
    return hook


def build_backbone(name, device):
    model = ASTModel.from_pretrained(ARGS.model, revision=ARGS.revision).eval()
    wb, ab = BACKBONES[name]
    lin = [m for m in model.encoder.modules() if isinstance(m, nn.Linear)]
    assert len(lin) == 72, len(lin)
    with torch.no_grad():
        for m in lin:
            if wb < 32:
                m.weight.copy_(quantize_weight_per_channel(m.weight, wb))
            if ab < 32:
                m.register_forward_pre_hook(act_quant_hook(ab))
    return model.to(device), lin


def count_bops(model, extractor, wb, ab):
    """MACs and bit-ops of one clip, from hooks on one forward pass."""
    macs = {"linear": 0, "attn": 0, "patch": 0}
    hooks = []

    def lin_hook(m, i, o):
        macs["linear"] += i[0].shape[:-1].numel() * m.in_features * m.out_features

    def conv_hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        macs["patch"] += o.shape[2] * o.shape[3] * m.out_channels * k

    def attn_hook(m, i, o):
        T = i[0].shape[1]
        macs["attn"] += 2 * T * T * m.all_head_size  # QK^T and attention x V

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif type(m).__name__.endswith("SelfAttention"):
            hooks.append(m.register_forward_hook(attn_hook))
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(input_values=extractor([np.zeros(TARGET_SR, np.float32)], sampling_rate=TARGET_SR,
                                     return_tensors="pt")["input_values"].to(dev))
    for h in hooks:
        h.remove()
    bops = macs["linear"] * wb * ab + (macs["attn"] + macs["patch"]) * 32 * 32
    return {"macs": macs, "bops": float(bops), "weight_bits": wb, "act_bits": ab}


# ── main ───────────────────────────────────────────────────────────────────

def parse_condition(c):
    b, n, s = c.split(":")
    if b not in BACKBONES or n not in NOISES:
        raise SystemExit(f"bad condition {c}")
    return b, n, (None if n == "clean" else float(s))


def cond_name(b, n, s):
    return f"{b}__{n}" + ("" if s is None else f"__{int(s) if float(s).is_integer() else s}db")


def main():
    t_start = time.time()
    meta = pd.read_csv(ARGS.csv, usecols=["folder", "name", "class_id"])
    if ARGS.limit:
        meta = meta.iloc[:ARGS.limit]
    stems = meta["name"].map(lambda n: Path(n).stem).tolist()
    index = index_wavs(ARGS.audio_dir)
    missing = [s for s in stems if s not in index]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing, e.g. {missing[:5]}")
    conds = [parse_condition(c) for c in ARGS.conditions]
    noise_maker = None
    if any(n == "bg" for _, n, _ in conds):
        bg_names = pd.read_csv(ARGS.bg_csv, usecols=["name"])["name"].map(lambda n: Path(n).stem).tolist()
        bg_index = index_wavs(ARGS.bg_audio_dir)
        bg_names = [b for b in bg_names if b in bg_index]
        noise_maker = NoiseMaker(bg_names, bg_index, ARGS.cache_audio or ARGS.cache_bg)
    else:
        noise_maker = NoiseMaker([], {}, False)

    out_dir = ARGS.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(ARGS.device)
    extractor = ASTFeatureExtractor.from_pretrained(ARGS.model, revision=ARGS.revision)
    clean_cache = {}

    def clean_wave(i):
        if i in clean_cache:
            return clean_cache[i]
        w = load_clip(index[stems[i]])
        if ARGS.cache_audio:
            clean_cache[i] = w
        return w

    for bname in dict.fromkeys(b for b, _, _ in conds):
        todo = [(b, n, s) for b, n, s in conds if b == bname
                and not (out_dir / f"{cond_name(b, n, s)}.npy").exists()]
        if not todo:
            continue
        model, _ = build_backbone(bname, device)
        cost = count_bops(model, extractor, *BACKBONES[bname])
        for b, n, s in todo:
            name = cond_name(b, n, s)
            t0, prep_s, fwd_s = time.time(), 0.0, 0.0
            feats, n_silent = [], 0
            for start in range(0, len(stems), ARGS.batch_size):
                tp = time.time()
                waves = []
                for i in range(start, min(start + ARGS.batch_size, len(stems))):
                    x = clean_wave(i)
                    if n != "clean":
                        x, mixed = mix(x, noise_maker(stems[i], n, len(x)), s)
                        n_silent += not mixed
                    waves.append(x)
                inputs = extractor(waves, sampling_rate=TARGET_SR, return_tensors="pt")
                tf = time.time()
                prep_s += tf - tp
                with torch.no_grad():
                    emb = model(input_values=inputs["input_values"].to(device)).pooler_output
                feats.append(emb.float().cpu().numpy())
                fwd_s += time.time() - tf
            X = np.concatenate(feats).astype(np.float32)
            if not np.isfinite(X).all():
                raise SystemExit(f"{name}: non-finite embeddings")
            np.save(out_dir / f"{name}.npy", X)
            dt = time.time() - t0
            side = {"condition": name, "backbone": b, "noise": n, "snr_db": s, "n_clips": len(stems),
                    "silent_clips_left_clean": n_silent, "seconds": dt, "clips_per_s": len(stems) / dt,
                    "prep_seconds": prep_s, "forward_seconds": fwd_s, "device": str(device),
                    "batch_size": ARGS.batch_size, "csv": str(ARGS.csv), "model": ARGS.model,
                    "revision": ARGS.revision, "backbone_cost_per_clip": cost,
                    "names_crc32": zlib.crc32("\n".join(stems).encode())}
            (out_dir / f"{name}.json").write_text(json.dumps(side, indent=2))
            if n == "bg":
                plan = out_dir / "bg_plan.json"
                old = json.loads(plan.read_text()) if plan.exists() else {}
                old.update(noise_maker.plan)
                plan.write_text(json.dumps(old, indent=0))
            print(f"{name}: {len(stems)} clips {dt:.0f}s ({len(stems) / dt:.2f} clips/s, prep {prep_s:.0f}s, "
                  f"forward {fwd_s:.0f}s), silent {n_silent}", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
