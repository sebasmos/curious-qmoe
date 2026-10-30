"""Write run_manifest.json for an idea 5 results folder.

Records the code commit and branch, machine, package versions, the AST model
and revision, the start time, and the sha256 of every CSV read and of every
embedding .npy written under the folder.

    python scripts/redesign/idea5/write_manifest.py RESULTS_DIR --start ISO_TIME --csv A.csv B.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--start", required=True)
    ap.add_argument("--csv", nargs="+", required=True, type=Path)
    ap.add_argument("--extra", default=None, help="JSON object merged into the manifest")
    a = ap.parse_args()
    import torch

    pkgs = ["torch", "torchaudio", "transformers", "numpy", "pandas", "scikit-learn", "soundfile", "omegaconf"]
    man = {
        "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "machine": {"node": platform.node(), "platform": platform.platform(), "processor": platform.processor(),
                    "python": platform.python_version(), "mps": torch.backends.mps.is_available()},
        "versions": {p: metadata.version(p) for p in pkgs},
        "ast_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "ast_revision": "f826b80d28226b62986cc218e5cec390b1096902",
        "start_time": a.start,
        "csv_sha256": {str(p): sha256(p) for p in a.csv},
        "embeddings_sha256": {str(p.relative_to(a.results)): sha256(p) for p in sorted(a.results.rglob("*.npy"))},
    }
    if a.extra:
        man.update(json.loads(a.extra))
    (a.results / "run_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"wrote {a.results / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
