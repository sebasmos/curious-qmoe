"""T21 checks on the embedding files of one dataset: row counts, names checksum equal across every file, finite rows
where extracted, the field rows of both assignments present in every cell file, the same clip lengths for both tiers."""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "idea6/stage2"))
from audio_shift import FIELD_SEEDS, field_assignment  # noqa: E402

R, D = Path(sys.argv[1]), sys.argv[2]
pool = pd.read_csv(R / "csv" / D / "pool.csv")
extra = pd.read_csv(R / "csv" / D / "extra.csv")
out = {}
for sub, meta in (("mn_pool", pool), ("ast_pool", pool), ("mn_extra", extra), ("ast_extra", extra), ("mn_int8", pool)):
    d = R / "emb" / D / sub
    if not d.exists():
        continue
    n = len(meta)
    crc = zlib.crc32("\n".join(meta["name"].map(lambda s: Path(s).stem)).encode())
    at, av = (np.array(field_assignment(n, FIELD_SEEDS[k])) for k in ("test", "val"))
    for f in sorted(d.glob("*__*.npy")):
        if f.name.startswith(("_", "lengths")):
            continue
        X = np.load(f, mmap_mode="r")
        side = json.loads(f.with_suffix(".json").read_text()) if f.with_suffix(".json").exists() else {}
        cond = f.stem.split("__", 1)[1]
        need = np.ones(n, bool) if "5db" not in cond else (at == cond) | (av == cond)
        fin = np.isfinite(np.asarray(X)).all(1)
        ok = X.shape[0] == n and fin[need].all() and side.get("names_crc32", crc) == crc
        out[f"{sub}/{f.name}"] = {"rows": int(X.shape[0]), "finite_needed": bool(fin[need].all()), "n_needed": int(need.sum()), "ok": bool(ok)}
        if not ok:
            print("FAIL", sub, f.name, flush=True)
if (R / "emb" / D / "ast_pool" / "lengths16k.npy").exists():
    out["lengths_equal_mn_ast"] = bool(np.array_equal(np.load(R / "emb" / D / "ast_pool" / "lengths16k.npy"), np.load(R / "emb" / D / "mn_pool" / "lengths__clean.npy")))
(R / "logs" / f"check_emb_{D}.json").write_text(json.dumps(out, indent=1))
print(sum(v["ok"] for v in out.values() if isinstance(v, dict)), "of", sum(isinstance(v, dict) for v in out.values()), "files ok;", "lengths equal:", out.get("lengths_equal_mn_ast"))
