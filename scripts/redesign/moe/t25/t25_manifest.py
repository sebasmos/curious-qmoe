"""run_manifest.json for the T25 results folder: commits, machine, software versions, inputs and weight checksums."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

R = Path(sys.argv[1])
W = Path.home() / "Documents/qmoe-work/t25"
R21 = Path.home() / "Documents/qmoe-results/2026-10-26_1408_moe-t21_code-731570e"
WD = Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1"


def run(*c):
    return subprocess.run(c, capture_output=True, text=True).stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


import numpy, pandas, scipy, sklearn, torch  # noqa: E402,E401

m = {"experiment": "T25", "branch": run("git", "-C", str(W), "rev-parse", "--abbrev-ref", "HEAD"),
     "head_commit": run("git", "-C", str(W), "rev-parse", "HEAD"), "plan_commit": "cc3b048", "base_commit_moe_t21": "7957360",
     "git_log": run("git", "-C", str(W), "log", "--format=%h %ci %s", "7957360..HEAD").splitlines(),
     "machine": {"platform": platform.platform(), "machine": platform.machine(), "chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
                 "memory_bytes": run("sysctl", "-n", "hw.memsize")},
     "python": sys.version, "versions": {"torch": torch.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
                                          "scipy": scipy.__version__, "sklearn": sklearn.__version__},
     "inputs": {"t21_results": str(R21), "t21_pool_csv_sha256": sha(R21 / "csv/d1/pool.csv"), "t25_pool_csv_sha256": sha(R / "csv/d1/pool.csv"),
                "t21_score_files_sha256": {p.name: sha(p) for p in sorted((R21 / "scores/d1").glob("pool_*_s*.npz")) if "int8" not in p.name},
                "dataset": "FSDKaggle2018 (Zenodo 2552860), ~/Documents/DATASETS/FSDKaggle2018/audio_raw, md5 checked by T21",
                "background": "UrbanSound8K, stage 2 bank"},
     "weights_sha256": {p.name: sha(p) for p in sorted(WD.glob("*.pt"))}}
(R / "run_manifest.json").write_text(json.dumps(m, indent=1))
print("manifest written")
