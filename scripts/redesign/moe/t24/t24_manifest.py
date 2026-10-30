"""run_manifest.json for the T24 results folder: commits, machine, software versions, dataset checksums.
    python t24_manifest.py R"""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

R = Path(sys.argv[1])
W = Path.home() / "Documents/qmoe-work/t24"
DS = Path.home() / "Documents/DATASETS"


def run(*c):
    return subprocess.run(c, capture_output=True, text=True).stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


import numpy, pandas, scipy, sklearn, torch, transformers  # noqa: E402,E401

m = {"experiment": "T24", "branch": run("git", "-C", str(W), "rev-parse", "--abbrev-ref", "HEAD"),
     "head_commit": run("git", "-C", str(W), "rev-parse", "HEAD"), "plan_commit": "9a51cf5",
     "git_log": run("git", "-C", str(W), "log", "--format=%h %ci %s", "7957360..HEAD").splitlines(),
     "machine": {"platform": platform.platform(), "chip": run("sysctl", "-n", "machdep.cpu.brand_string"), "memory_bytes": run("sysctl", "-n", "hw.memsize")},
     "python": sys.version, "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": numpy.__version__,
                                          "pandas": pandas.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
     "datasets": {}, "weights": {"mn10": sha(Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1/mn10_as_mAP_471.pt"),
                                 "ast": "MIT/ast-finetuned-audioset-10-10-0.4593 @ f826b80d28226b62986cc218e5cec390b1096902"},
     "background_noise": str(DS / "Urban8k/ast_768/urbansound8k.csv")}
d3 = DS / "FSDnoisy18k"
if d3.exists():
    m["datasets"]["FSDnoisy18k"] = {"zenodo": "2529934", "licence": "CC BY (dataset); clips CC BY or CC0 (LICENSE-INDIVIDUAL-CLIPS.csv)",
        "md5_verified": {"FSDnoisy18k.audio_train.zip": "34dc1d34ca44622af5bf439ceb6f0d55", "FSDnoisy18k.audio_test.zip": "1ac73d70b4ef3f81900d98c261a832de",
                         "FSDnoisy18k.meta.zip": "96e27a4a63b7a2870522ddcedb5d8296", "FSDnoisy18k.doc.zip": "093a1ca185ec341ca4eac14215e7f96b"},
        "meta_sha256": {p.name: sha(p) for p in sorted((d3 / "meta_x/FSDnoisy18k.meta").glob("*.csv"))},
        "n_prepared_wav": len(list((d3 / "audio16k_t24").glob("*.wav"))) if (d3 / "audio16k_t24").exists() else 0}
d4 = DS / "TUT2017"
if d4.exists():
    m["datasets"]["TUT Acoustic Scenes 2017"] = {"zenodo": "400515 (development), 1040168 (evaluation)", "licence": "academic non-commercial (EULA)",
        "md5_verified": json.loads((d4 / "md5_verified.json").read_text()) if (d4 / "md5_verified.json").exists() else {},
        "n_prepared_wav": len(list((d4 / "audio16k_t24").glob("*.wav"))) if (d4 / "audio16k_t24").exists() else 0}
(R / "run_manifest.json").write_text(json.dumps(m, indent=1))
print("manifest written")
