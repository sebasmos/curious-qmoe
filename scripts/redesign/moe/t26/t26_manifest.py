"""run_manifest.json for the T26 results folder: commits, machine, software versions, inputs and checksums.
    python t26_manifest.py R26 PLAN_COMMIT"""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

R, PLAN = Path(sys.argv[1]), sys.argv[2]
W = Path.home() / "Documents/qmoe-work/t26"
Q = Path.home() / "Documents/qmoe-results"
R21, R25, R24 = Q / "2026-10-26_1408_moe-t21_code-731570e", Q / "2026-10-28_2247_moe-t25_code-cc3b048", Q / "2026-10-28_1827_moe-t24_code-9a51cf5"
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

m = {"experiment": "T26", "branch": run("git", "-C", str(W), "rev-parse", "--abbrev-ref", "HEAD"),
     "head_commit": run("git", "-C", str(W), "rev-parse", "HEAD"), "plan_commit": PLAN, "base_commit_moe_t25": "23175a9",
     "merged_moe_t24": "4631bc9",
     "git_log": run("git", "-C", str(W), "log", "--format=%h %ci %s", "23175a9..HEAD").splitlines(),
     "machine": {"platform": platform.platform(), "machine": platform.machine(), "chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
                 "memory_bytes": run("sysctl", "-n", "hw.memsize")},
     "python": sys.version, "versions": {"torch": torch.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
                                          "scipy": scipy.__version__, "sklearn": sklearn.__version__},
     "inputs": {"t21": str(R21), "t25": str(R25), "t24": str(R24),
                "part_a_score_files_sha256": {**{f"t21/{p.name}": sha(p) for p in sorted((R21 / "scores/d1").glob("pool_*_s*.npz")) if "int8" not in p.name},
                                              **{f"t25/{p.name}": sha(p) for p in sorted((R25 / "scores/d1").glob("pool_*_s*.npz"))}},
                "part_c_t24_score_files_sha256": {f"{d}/{p.name}": sha(p) for d in ("d3", "d4") for p in sorted((R24 / "scores" / d).glob("*.npz"))},
                "part_c_csv_sha256": {f"{d}/{n}": sha(R / "csv" / d / n) for d in ("d3", "d4") for n in ("all.csv", "eval.csv") if (R / "csv" / d / n).exists()},
                "part_c_t24_csv_sha256": {f"{d}/{n}": sha(R24 / "csv" / d / n) for d in ("d3", "d4") for n in ("all.csv", "eval.csv")},
                "part_c_t26_score_files_sha256": {f"{d}/{p.name}": sha(p) for d in ("d3", "d4") for p in sorted((R / "scores" / d).glob("*.npz"))},
                "datasets": "FSDnoisy18k (Zenodo 2529934) and TUT Acoustic Scenes 2017 (Zenodo 400515, 1040168), prepared 16 kHz audio of T24 "
                            "(~/Documents/DATASETS/FSDnoisy18k/audio16k_t24, ~/Documents/DATASETS/TUT2017/audio16k_t24), md5 checked by T24",
                "background": "UrbanSound8K, stage 2 bank"},
     "weights_sha256": {p.name: sha(p) for p in sorted(WD.glob("*.pt"))}}
(R / "run_manifest.json").write_text(json.dumps(m, indent=1))
print("manifest written")
