"""Check, before any T24 audio, that t24_prep_audio.py's reader plus the T24 reader return exactly what T21's reader
returns: N UrbanSound8K files (already on disk, 44.1 or 48 kHz, mono and stereo) are prepared to a scratch
audio16k_t24 directory and read back through t24_common.load_clip_t24, and compared with t21_common.load_clip_t21.

    python t24_check_prep.py SCRATCH_DIR [N]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t24_common as C  # noqa: E402

scr, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 24
src = sorted(C.C21.URB_AUDIO.rglob("*.wav"))
pick = [src[i] for i in np.random.default_rng(0).choice(len(src), n, replace=False)]
import csv, zipfile  # noqa: E401,E402
scr.mkdir(parents=True, exist_ok=True)
zp = scr / "check.zip"
with zipfile.ZipFile(zp, "w") as z:
    for p in pick:
        z.write(p, "x/" + p.name)
with open(scr / "check.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["part", "src", "name"])
    for p in pick:
        w.writerow(["chk", p.name, "c_" + p.name])
od = scr / "audio16k_t24"
subprocess.run([sys.executable, "-I", str(HERE / "t24_prep_audio.py"), str(zp), str(scr / "check.csv"), "chk", str(od), str(scr / "tmp")], check=True)
worst = 0.0
for p in pick:
    a = C.C21.load_clip_t21(p)
    b = C.load_clip_t24(od / ("c_" + p.name))
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape, (p, a.shape, b.shape)
    worst = max(worst, float(np.abs(a - b).max()))
print(f"{n} files: max abs difference {worst}  ->  {'EXACT' if worst == 0 else 'DIFFERENT'}")
