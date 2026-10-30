"""run_manifest.json for the idea 6 final results folder: commits, machine, package versions, inputs read, SHA-256
of every training CSV, embedding array and audio-statistics file written here, and of the earlier stages' files the
analysis scored against.

    python write_manifest.py R --inputs S2 S3B S3C UNP I5 I7 --weights W
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
    ap.add_argument("--weights", type=Path, required=True)
    a = ap.parse_args(argv)
    import numpy, pandas, scipy, sklearn, torch, torchaudio, torchvision, transformers  # noqa: E401

    R = a.results
    files = {}
    for pat in ("csv/*.csv", "emb_cnn/*/*.npy", "emb_ast/*/*.npy", "emb_ast/shared512/*/*.npy", "audio_stats/*/*.npz",
                "csv/*.npy"):
        for p in sorted(R.glob(pat)):
            files[str(p.relative_to(R))] = sha(p)
    m = {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "head": git("rev-parse", "HEAD"),
         "log": git("log", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M", "545bb54..HEAD").splitlines(),
         "base_commit": "545bb54 (redesign/idea6-stage3c-unpadded)",
         "machine": {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version,
                     "hardware": "Apple M2 Pro, 12 cores, 32 GB"},
         "packages": {m_.__name__: m_.__version__ for m_ in (numpy, pandas, scipy, sklearn, torch, torchaudio,
                                                              torchvision, transformers)},
         "inputs_read_only": [str(p) for p in a.inputs], "sha256": files,
         "efficientat_weights": {p.name: sha(p) for p in sorted(a.weights.glob("*.pt"))}}
    for p in sorted(R.glob("logs/*.done")) + [R / "logs" / "analysis_plan_commit.txt", R / "logs" / "check_extract.json",
                                              R / "logs" / "check_repro.json"]:
        if p.exists():
            m.setdefault("logs", {})[p.name] = p.read_text().splitlines()
    (R / "run_manifest.json").write_text(json.dumps(m, indent=1))
    print(f"{R / 'run_manifest.json'}: {len(files)} files hashed")


if __name__ == "__main__":
    main()
