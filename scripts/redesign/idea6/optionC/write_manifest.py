"""run_manifest.json for the idea 6 option C results folder: commits, machine,
package versions, inputs read, and SHA-256 of every training CSV and of the
clean embedding arrays.

    python scripts/redesign/idea6/optionC/write_manifest.py R --inputs S3C N S3B S2 I5 I7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    root = Path(__file__).resolve().parents[4]
    return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--inputs", nargs="*", type=Path, default=[])
    a = ap.parse_args(argv)
    import numpy, pandas, scipy, sklearn, torch, torchaudio, torchvision, transformers  # noqa: E401

    R = a.results
    files = {}
    for p in sorted(list(R.glob("csv/*.csv")) + list(R.glob("emb_ast_short/*/*.npy")) + list(R.glob("outputs/*/*/summary.json"))
                    + list(R.glob("scores_*/*/*/*/*__clean.npz"))):
        files[str(p.relative_to(R))] = sha(p)
    m = {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "head": git("rev-parse", "HEAD"),
         "log": git("log", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M", "545bb54..HEAD").splitlines(),
         "machine": {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version},
         "packages": {m_.__name__: m_.__version__ for m_ in (numpy, pandas, scipy, sklearn, torch, torchaudio,
                                                              torchvision, transformers)},
         "inputs_read_only": [str(p) for p in a.inputs], "sha256": files}
    m["ast_model"] = {"name": "MIT/ast-finetuned-audioset-10-10-0.4593", "revision": "f826b80d28226b62986cc218e5cec390b1096902"}
    for p in sorted(R.glob("logs/*.done")) + [R / "logs" / "analysis_plan_commit.txt", R / "logs" / "analysis.done"]:
        if p.exists():
            m.setdefault("logs", {})[p.name] = p.read_text().splitlines()
    (R / "run_manifest.json").write_text(json.dumps(m, indent=1))
    print(f"{R / 'run_manifest.json'}: {len(files)} files hashed")


if __name__ == "__main__":
    main()
