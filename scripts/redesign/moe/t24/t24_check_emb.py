"""T24 checks on the embedding files of one dataset (plan C8), run before any score: row counts, names checksum equal
to the CSV of each set, finite rows where needed (5 dB cells: the rows of either field assignment over eval.csv), the
same clip lengths for both tiers.   python t24_check_emb.py R d3"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "idea6/stage2"))
from audio_shift import FIELD_SEEDS, field_assignment  # noqa: E402

R, D = Path(sys.argv[1]), sys.argv[2]
out = {}
for sub, csvn in (("mn_all", "all"), ("ast_all", "all"), ("mn_eval", "eval"), ("ast_eval", "eval")):
    meta = pd.read_csv(R / "csv" / D / f"{csvn}.csv")
    d = R / "emb" / D / sub
    n = len(meta)
    crc = zlib.crc32("\n".join(meta["name"].map(lambda s: Path(s).stem)).encode())
    at, av = (np.array(field_assignment(n, FIELD_SEEDS[k])) for k in ("test", "val"))
    for f in sorted(d.glob("*__*.npy")):
        if f.name.startswith(("_", "lengths")):
            continue
        X = np.load(f, mmap_mode="r")
        side = json.loads(f.with_suffix(".json").read_text())
        cond = f.stem.split("__", 1)[1]
        need = np.ones(n, bool) if "5db" not in cond else (at == cond) | (av == cond)
        fin = np.isfinite(np.asarray(X)).all(1)
        ok = X.shape[0] == n and bool(fin[need].all()) and side.get("names_crc32") == crc
        out[f"{sub}/{f.name}"] = {"rows": int(X.shape[0]), "finite_needed": bool(fin[need].all()), "n_needed": int(need.sum()), "ok": bool(ok)}
        if not ok:
            print("FAIL", sub, f.name, flush=True)
for s in ("all", "eval"):
    a, m = R / "emb" / D / f"ast_{s}" / "lengths16k.npy", R / "emb" / D / f"mn_{s}" / "lengths__clean.npy"
    if a.exists() and m.exists():
        out[f"lengths_equal_mn_ast_{s}"] = bool(np.array_equal(np.load(a), np.load(m)))
(R / "logs" / f"check_emb_{D}.json").write_text(json.dumps(out, indent=1))
print(sum(v["ok"] for v in out.values() if isinstance(v, dict)), "of", sum(isinstance(v, dict) for v in out.values()), "files ok;",
      {k: v for k, v in out.items() if k.startswith("lengths")})
