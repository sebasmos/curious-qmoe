"""Two-window AST embeddings for idea 6 Stage 3b (ESC-50): the mean of the
pooled embedding of frames 0 to 255 (<first>/<config>__<condition>.npy) and of
frames 256 to 511 (<second>/<config>__<condition>.npy, extracted with
--frame-offset 256 into its own folder under the same file names).
Writes <out>/<config>__<condition>.npy and a sidecar whose backbone cost is
the sum of both windows' sidecar costs (two passes; the fbank is shared and
priced once by the analysis). Rows, names checksum and max_length must agree.

    python scripts/redesign/idea6/stage3b/make_mean2.py --first E/esc50_256 --second E/esc50_256o256 --out E/esc50_256x2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first", type=Path, required=True)
    ap.add_argument("--second", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p2 in sorted(a.second.glob("*__*.npy")):
        p1 = a.first / p2.name
        po = a.out / p2.name
        if po.exists() or not p1.exists():
            continue
        s1, s2 = (json.loads(p.with_suffix(".json").read_text()) for p in (p1, p2))
        if (s1["names_crc32"] != s2["names_crc32"] or s1["max_length"] != 256 or s2["max_length"] != 256
                or s1.get("frame_offset", 0) != 0 or s2["frame_offset"] != 256 or s1["config"] != s2["config"]):
            raise SystemExit(f"{p2.name}: windows do not match")
        x1, x2 = np.load(p1), np.load(p2)
        if x1.shape != x2.shape or not np.array_equal(np.isfinite(x1).all(1), np.isfinite(x2).all(1)):
            raise SystemExit(f"{p2.name}: rows differ")
        np.save(po, ((x1.astype(np.float64) + x2) / 2).astype(np.float32))
        cost = {k: (v * 2 if isinstance(v, (int, float)) else v) for k, v in s1["backbone_cost_per_clip"].items()
                if k in ("bitops", "macs")}
        side = {**s1, "config": s1["config"], "windows": [[0, 256], [256, 512]], "frame_offset": None,
                "backbone_cost_per_clip": {**s1["backbone_cost_per_clip"], **cost},
                "window_sidecars": [str(p1.with_suffix(".json")), str(p2.with_suffix(".json"))]}
        po.with_suffix(".json").write_text(json.dumps(side, indent=2))
        n += 1
    print(f"{a.out}: wrote {n} two-window file(s)")


if __name__ == "__main__":
    main()
