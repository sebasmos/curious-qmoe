"""run_manifest.json for the idea 6 option B results folder (Stage 3c's writer with the commit range, file globs and the dataset entry changed): commits, machine,
package versions, inputs read, and SHA-256 of every training CSV and of the
clean embedding arrays.

    python scripts/redesign/idea6/stage3c/write_manifest.py R --inputs S2 I5 I7 --weights W
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    ap.add_argument("--weights", type=Path, required=True, help="folder of the downloaded EfficientAT weights")
    a = ap.parse_args(argv)
    import numpy, pandas, scipy, sklearn, torch, torchaudio, torchvision, transformers  # noqa: E401

    R = a.results
    files = {}
    for p in sorted(list(R.glob("csv/*.csv")) + list(R.glob("emb*/*/*__clean.npy")) + list(R.glob("audio_stats/*/*.npz")) + list(R.glob("scores*/*/*/*.npz")) + list(R.glob("models/*/*.onnx")) + list(R.glob("models/*.onnx"))):
        files[str(p.relative_to(R))] = sha(p)
    m = {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "head": git("rev-parse", "HEAD"),
         "log": git("log", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M", "cbf6be6..HEAD").splitlines(),
         "machine": {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version},
         "packages": {m_.__name__: m_.__version__ for m_ in (numpy, pandas, scipy, sklearn, torch, torchaudio,
                                                              torchvision, transformers)},
         "inputs_read_only": [str(p) for p in a.inputs], "sha256": files}
    w = Path.home() / ".cache/torch/hub/checkpoints/efficientnet_b3_rwightman-b3899882.pth"
    if w.exists():
        m["efficientnet_b3_weights"] = {"path": str(w), "sha256": sha(w)}
    m["efficientat_weights"] = {
        "release": "https://github.com/fschmid56/EfficientAT/releases/tag/v0.0.1",
        "code_commit": "a425fdce92572e602a1d5634799bd9f1f2efa806",
        "published_checksum": None,
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(a.weights.glob("*.pt"))}}
    ds = Path.home() / "Documents/DATASETS/TAU2020Mobile"
    m["dataset"] = {"name": "TAU Urban Acoustic Scenes 2020 Mobile, development dataset", "zenodo": 3819968,
                    "doi": "10.5281/zenodo.3819968", "licence": "other-nc (TAU licence, non-commercial research use)",
                    "audio_raw": str(ds / "audio_raw/tau2020mobile_zenodo_3819968"),
                    "md5_zenodo": (ds / "audio_raw/tau2020mobile_zenodo_3819968/md5_zenodo.txt").read_text().splitlines(),
                    "verified": sorted(p.read_text().strip() for p in (ds / "audio_raw/tau2020mobile_zenodo_3819968/logs").glob("*.ok")),
                    "meta_csv_sha256": sha(ds / "ast_768/tau2020mobile.csv"), "folds_json_sha256": sha(ds / "ast_768/tau2020mobile.folds.json")}
    ort_env = Path(os.environ.get("PYO", "python"))
    if ort_env.exists():
        v = subprocess.run([str(ort_env), "-c", "import onnxruntime, onnx, torch; print(onnxruntime.__version__, onnx.__version__, torch.__version__)"],
                           capture_output=True, text=True).stdout.split()
        if len(v) == 3:
            m["ort_env"] = {"python": str(ort_env), "onnxruntime": v[0], "onnx": v[1], "torch": v[2]}
    for p in sorted(R.glob("logs/*.done")) + [R / "logs" / "analysis_plan_commit.txt"]:
        if p.exists():
            m.setdefault("logs", {})[p.name] = p.read_text().splitlines()
    (R / "run_manifest.json").write_text(json.dumps(m, indent=1))
    print(f"{R / 'run_manifest.json'}: {len(files)} files hashed")


if __name__ == "__main__":
    main()
