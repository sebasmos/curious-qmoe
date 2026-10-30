"""T25 checks on the new CNN embedding files before any score: row count and names checksum equal to T21's pool,
finite on every row a condition needs (all rows, or the field rows of both assignments for the 5 dB cells),
clip lengths equal to T21's MN10 lengths in every condition, and the background plan equal to T21's.

    python t25_check_emb.py R25 R21 d1
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "idea6/stage2"))
from audio_shift import FIELD_SEEDS, field_assignment  # noqa: E402

R, B, D = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
pool = pd.read_csv(R / "csv" / D / "pool.csv")
n = len(pool)
crc = zlib.crc32("\n".join(pool["name"].map(lambda s: Path(s).stem)).encode())
at, av = (np.array(field_assignment(n, FIELD_SEEDS[k])) for k in ("test", "val"))
d, d21 = R / "emb" / D / "cnn_pool", B / "emb" / D / "mn_pool"
out, bad = {}, 0
for f in sorted(d.glob("*__*.npy")):
    if f.name.startswith(("_", "lengths")):
        continue
    X = np.load(f, mmap_mode="r")
    side = json.loads(f.with_suffix(".json").read_text())
    cond = f.stem.split("__", 1)[1]
    need = np.ones(n, bool) if "5db" not in cond else (at == cond) | (av == cond)
    fin = np.isfinite(np.asarray(X)).all(1)
    ok = X.shape[0] == n and bool(fin[need].all()) and side["names_crc32"] == crc
    out[f.name] = {"rows": int(X.shape[0]), "n_needed": int(need.sum()), "ok": ok}
    bad += not ok
for f in sorted(d.glob("lengths__*.npy")):
    a, b = np.load(f), np.load(d21 / f.name)
    cond = f.stem.split("__", 1)[1]
    need = np.ones(n, bool) if "5db" not in cond else (at == cond) | (av == cond)
    ok = bool(np.array_equal(a[need], b[need]))
    out[f.name] = {"equal_to_t21": ok}
    bad += not ok
p25, p21 = json.loads((d / "bg_plan.json").read_text()), json.loads((d21 / "bg_plan.json").read_text())
out["bg_plan_equal_to_t21"] = p25 == p21
bad += not out["bg_plan_equal_to_t21"]
(R / "logs" / f"check_emb_{D}.json").write_text(json.dumps(out, indent=1))
print(f"{len(out)} checks, {bad} failed", flush=True)
sys.exit(1 if bad else 0)
