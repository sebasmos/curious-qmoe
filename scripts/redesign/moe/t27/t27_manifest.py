"""run_manifest.json for the T27 results folder: commits, machine, software versions, inputs and checksums.
    python t27_manifest.py R27 PLAN_COMMIT"""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

R, PLAN = Path(sys.argv[1]), sys.argv[2]
W = Path(__file__).resolve().parents[4]
Q = Path.home() / "Documents/qmoe-results"
R21, R24, R25, R26 = (Q / n for n in ("2026-10-26_1408_moe-t21_code-731570e", "2026-10-28_1827_moe-t24_code-9a51cf5",
                                      "2026-10-28_2247_moe-t25_code-cc3b048", "2026-10-29_0912_moe-t26_code-6ecea17"))


def run(*c):
    return subprocess.run(c, capture_output=True, text=True).stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


import numpy, pandas, scipy, sklearn  # noqa: E402,E401

d1 = [R21 / f"scores/d1/pool_{m}_s{s}.npz" for m in ("mn10_w8a8", "ast_w4g128") for s in range(5)] + \
     [R25 / f"scores/d1/pool_{m}_w8a8_s{s}.npz" for m in ("mn04", "mn05", "mn20", "dymn04") for s in range(5)]
d3 = [R24 / f"scores/d3/{m}_s{s}.npz" for m in ("mn10_w8a8", "ast_w4g128") for s in range(3)] + \
     [R26 / f"scores/d3/{m}_w8a8_s{s}.npz" for m in ("mn04", "mn05", "mn20", "dymn04") for s in range(3)]
m = {"experiment": "T27 (post hoc)", "branch": run("git", "-C", str(W), "rev-parse", "--abbrev-ref", "HEAD"),
     "head_commit": run("git", "-C", str(W), "rev-parse", "HEAD"), "plan_commit": PLAN, "base_commit_moe_t26": "b6b5276",
     "git_log": run("git", "-C", str(W), "log", "--format=%h %ci %s", "b6b5276..HEAD").splitlines(),
     "machine": {"platform": platform.platform(), "machine": platform.machine(), "chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
                 "memory_bytes": run("sysctl", "-n", "hw.memsize")},
     "python": sys.version, "versions": {"numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
                                          "sklearn": sklearn.__version__},
     "inputs": {"t21": str(R21), "t24": str(R24), "t25": str(R25), "t26": str(R26),
                "score_files_sha256": {str(p.relative_to(Q)): sha(p) for p in d1 + d3},
                "t26_reference_sha256": {str(p.relative_to(Q)): sha(p) for p in (R26 / "tables/a/verdict.json", R26 / "tables/a/by_seed_fold.csv",
                                                                                 R26 / "tables/c/d3/verdict.json", R26 / "tables/c/d3/by_seed_fold.csv")}}}
(R / "run_manifest.json").write_text(json.dumps(m, indent=1))
print("manifest written")
