"""Extract frozen Audio Spectrogram Transformer (AST) clip embeddings.

Takes an existing embedding CSV (one row per clip, a ``name`` column such as
``1-100032-A-0.png`` plus metadata columns and numeric feature columns) and
writes a new CSV with the same rows, in the same order, with the same metadata
columns, and with the feature columns replaced by AST embeddings ``0``..``767``.

Each row's audio is found by searching ``audio_dir`` recursively for
``<stem>.wav``, where ``<stem>`` is ``name`` without its extension. A missing
or unreadable clip is a hard error: no row is ever dropped.

Model: ``MIT/ast-finetuned-audioset-10-10-0.4593`` (AST fine-tuned on
AudioSet). Audio is downmixed to mono, DC-centred, resampled to 16 kHz and
turned into the model's own 128-bin Kaldi fbank input (1024 frames, i.e.
10.24 s; shorter clips are zero-padded, longer ones truncated) by its
feature extractor.

Pooling:
  pooler (default)  ``pooler_output``: mean of the [CLS] and [DIST] tokens after
                    the final LayerNorm. This is the vector AST's own
                    classification head reads, so it is the standard AST
                    clip embedding.
  mean              mean of the patch tokens of ``last_hidden_state``.

Usage:
  python extract_ast_embeddings.py AUDIO_DIR EXISTING_CSV OUTPUT_CSV \
      [--device auto|cpu|mps] [--batch-size 8] [--threads 6] [--pooling pooler]
"""

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("audio_dir", type=Path, help="root folder searched recursively for <stem>.wav")
    p.add_argument("csv", type=Path, help="existing embedding CSV whose rows and metadata are copied")
    p.add_argument("out", type=Path, help="output CSV path")
    p.add_argument("--model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--revision", default="f826b80d28226b62986cc218e5cec390b1096902",
                   help="pinned Hugging Face commit of the model")
    p.add_argument("--pooling", choices=["pooler", "mean"], default="pooler")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=6, help="max CPU threads (torch, OpenMP)")
    return p.parse_args()


ARGS = parse_args()
# Thread caps must be set before numpy/torch are imported.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = str(ARGS.threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402
from transformers import ASTFeatureExtractor, ASTModel  # noqa: E402

torch.set_num_threads(ARGS.threads)
TARGET_SR = 16000


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def index_wavs(audio_dir):
    """Map file stem -> path for every .wav under audio_dir; stems must be unique."""
    index = {}
    for path in sorted(audio_dir.rglob("*.wav")):
        if path.stem in index:
            raise SystemExit(f"duplicate clip name {path.stem}: {index[path.stem]} and {path}")
        index[path.stem] = path
    return index


def load_clip(path):
    """Read a wav, downmix to mono, remove the DC offset and resample to 16 kHz."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.mean(axis=1))
    wav = wav - wav.mean()  # same centring as the original AST data loader
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    return wav.numpy()


@torch.no_grad()
def embed_batch(waves, extractor, model, device, pooling):
    inputs = extractor(waves, sampling_rate=TARGET_SR, return_tensors="pt")
    out = model(input_values=inputs["input_values"].to(device))
    if pooling == "pooler":
        emb = out.pooler_output
    else:
        emb = out.last_hidden_state[:, 2:].mean(dim=1)  # skip [CLS] and [DIST]
    return emb.float().cpu().numpy()


def main():
    t0 = time.time()
    header = pd.read_csv(ARGS.csv, nrows=0).columns
    meta_cols = [c for c in header if not c.isdigit()]
    meta = pd.read_csv(ARGS.csv, usecols=meta_cols)[meta_cols]
    if "name" not in meta.columns:
        raise SystemExit("existing CSV has no 'name' column")
    if meta["name"].duplicated().any():
        raise SystemExit("existing CSV has duplicate names")

    index = index_wavs(ARGS.audio_dir)
    stems = meta["name"].map(lambda n: Path(n).stem)
    missing = [s for s in stems if s not in index]
    if missing:
        raise SystemExit(f"{len(missing)} clips not found under {ARGS.audio_dir}, e.g. {missing[:10]}")
    paths = [index[s] for s in stems]

    device = pick_device(ARGS.device)
    extractor = ASTFeatureExtractor.from_pretrained(ARGS.model, revision=ARGS.revision)
    model = ASTModel.from_pretrained(ARGS.model, revision=ARGS.revision).to(device).eval()
    print(f"{len(paths)} clips, device={device}, pooling={ARGS.pooling}, "
          f"max_length={extractor.max_length}", flush=True)

    feats, unreadable = [], []
    for start in range(0, len(paths), ARGS.batch_size):
        waves = []
        for path in paths[start:start + ARGS.batch_size]:
            try:
                waves.append(load_clip(path))
            except Exception as err:  # collect every bad file, then fail
                unreadable.append(f"{path}: {err}")
                waves.append(None)
        good = [w for w in waves if w is not None]
        if good:
            emb = iter(embed_batch(good, extractor, model, device, ARGS.pooling))
            feats.extend(next(emb) if w is not None else None for w in waves)
        else:
            feats.extend([None] * len(waves))
        done = start + len(waves)
        if done % (ARGS.batch_size * 50) == 0 or done == len(paths):
            print(f"  {done}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)

    if unreadable:
        raise SystemExit(f"{len(unreadable)} unreadable clips:\n" + "\n".join(unreadable))

    X = np.stack(feats).astype(np.float32)
    if not np.isfinite(X).all():
        raise SystemExit("non-finite values in embeddings")
    out = pd.concat([meta, pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])], axis=1)
    ARGS.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ARGS.out, index=False, float_format="%.8g")
    print(f"wrote {ARGS.out} {out.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
