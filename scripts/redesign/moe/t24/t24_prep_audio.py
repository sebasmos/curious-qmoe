"""Prepare the clips of one downloaded zip: each listed member is copied to a temporary file, read with T21's reader
(stage 2's load_clip, then the first 163,840 samples) and written as a 16 kHz float32 WAV named after the CSV's `name`
under OUT_DIR (a directory named audio16k_t24). Archive content is only read as data, never run.

    python -I t24_prep_audio.py ZIP CSV PART OUT_DIR TMP_DIR      (rows of CSV whose `part` equals PART; member = `src`)
"""
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import zipfile

MAX_SAMPLES, TARGET_SR = 163840, 16000


def load_clip(path):  # stage 2 audio_shift.load_clip, verbatim (equality checked by t24_check_prep.py)
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.mean(axis=1))
    wav = wav - wav.mean()
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    return wav.numpy()


zp, cp, part, od, tmp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
assert od.name == "audio16k_t24" or "audio16k_t24" in od.parts
od.mkdir(parents=True, exist_ok=True)
tmp.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
want = {r["src"]: r["name"] for r in csv.DictReader(cp.open()) if r["part"] == part}
got = 0
with zipfile.ZipFile(zp) as z:
    for m in z.infolist():
        b = Path(m.filename).name
        if m.is_dir() or b not in want:
            continue
        dst = od / want[b]
        if not dst.exists():
            t = tmp / ("src_" + b)
            with z.open(m) as src, open(t, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
            x = load_clip(t)[:MAX_SAMPLES].astype(np.float32)
            sf.write(str(dst) + ".part.wav", x, TARGET_SR, subtype="FLOAT")
            Path(str(dst) + ".part.wav").rename(dst)
            t.unlink()
        got += 1
print(f"{zp.name}: {got} of {len(want)} listed {part} clips prepared in {od}", flush=True)
