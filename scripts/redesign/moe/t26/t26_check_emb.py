"""T26 Part C embedding checks before any score: row count and names checksum equal to the CSV of each set, finite on
every row a condition needs (5 dB cells: the rows of either field assignment over eval.csv), clip lengths equal to
T24's MN10 lengths in every condition, background plan equal to T24's.   python t26_check_emb.py R26 R24 d3"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "idea6/stage2"))
from audio_shift import FIELD_SEEDS, field_assignment  # noqa: E402

R, B, D = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
out, bad = {}, 0
for sub, sub24, csvn in (("cnn_all", "mn_all", "all"), ("cnn_eval", "mn_eval", "eval")):
    meta = pd.read_csv(R / "csv" / D / f"{csvn}.csv")
    n = len(meta)
    crc = zlib.crc32("\n".join(meta["name"].map(lambda s: Path(s).stem)).encode())
    at, av = (np.array(field_assignment(n, FIELD_SEEDS[k])) for k in ("test", "val"))
    d = R / "emb" / D / sub
    for f in sorted(d.glob("*__*.npy")):
        if f.name.startswith(("_", "lengths")):
            continue
        X = np.load(f, mmap_mode="r")
        side = json.loads(f.with_suffix(".json").read_text())
        cond = f.stem.split("__", 1)[1]
        need = np.ones(n, bool) if "5db" not in cond else (at == cond) | (av == cond)
        fin = np.isfinite(np.asarray(X)).all(1)
        ok = X.shape[0] == n and bool(fin[need].all()) and side["names_crc32"] == crc
        out[f"{sub}/{f.name}"] = {"rows": int(X.shape[0]), "n_needed": int(need.sum()), "ok": bool(ok)}
        bad += not ok
    for f in sorted(d.glob("lengths__*.npy")):
        cond = f.stem.split("__", 1)[1]
        need = np.ones(n, bool) if "5db" not in cond else (at == cond) | (av == cond)
        ok = bool(np.array_equal(np.load(f)[need], np.load(B / "emb" / D / sub24 / f.name)[need]))
        out[f"{sub}/{f.name}"] = {"equal_to_t24": ok}
        bad += not ok
p = R / "emb" / D / "cnn_eval" / "bg_plan.json"
if p.exists():
    out["bg_plan_equal_to_t24"] = json.loads(p.read_text()) == json.loads((B / "emb" / D / "mn_eval" / "bg_plan.json").read_text())
    bad += not out["bg_plan_equal_to_t24"]
(R / "logs" / f"check_emb_{D}.json").write_text(json.dumps(out, indent=1))
print(f"{len(out)} checks, {bad} failed", flush=True)
sys.exit(1 if bad else 0)
