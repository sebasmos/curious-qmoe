"""Writes <results>/run_manifest.json: code commit, machine, software versions, dataset CSV sha256 and a sha256 over the audio file
listing (relative path and size of every file), stored inputs reused, and the commands of every step.

    python t19_manifest.py
"""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t19_common import OPTB, S2, TASKS, results_dir  # noqa: E402

R = results_dir()
W = Path(__file__).resolve().parents[3]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def listing_sha(d):
    h = hashlib.sha256()
    n = 0
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            h.update(f"{p.relative_to(d)}\t{p.stat().st_size}\n".encode())
            n += 1
    return h.hexdigest(), n


def git(*a):
    return subprocess.run(["git", "-C", str(W), *a], capture_output=True, text=True).stdout.strip()


import numpy, scipy, sklearn, torch, transformers  # noqa: E402,E401

data = {}
for t in ("esc50", "urban", "tau"):
    T = TASKS[t]
    lsha, n = listing_sha(T["audio"])
    data[t] = dict(csv=str(T["csv"]), csv_sha256=sha(T["csv"]), audio=str(T["audio"]), audio_listing_sha256=lsha, audio_files=n)
m = dict(
    experiment="T19 precision cascade with low-bit activations (QMOE final round)",
    code=dict(repo=str(W), branch=git("rev-parse", "--abbrev-ref", "HEAD"), commit=git("rev-parse", "HEAD"),
              plan_commit="ae334eb", dirty=bool(git("status", "--porcelain", "--untracked-files=no"))),
    machine=dict(host=platform.node(), platform=platform.platform(), processor=platform.processor(), gpu="Apple M2 Pro (MPS)",
                 python=sys.version.split()[0]),
    software=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=numpy.__version__, scipy=scipy.__version__,
                  sklearn=sklearn.__version__),
    model=dict(name="MIT/ast-finetuned-audioset-10-10-0.4593", revision="f826b80d28226b62986cc218e5cec390b1096902"),
    datasets=data,
    reused_inputs=dict(stage2_scores=str(S2), optionB_scores=str(OPTB)),
    analysis_plan="ANALYSIS_PLAN.md (copy of scripts/redesign/moe/ANALYSIS_PLAN_T19.md, Amendment 1 included)",
)
(R / "run_manifest.json").write_text(json.dumps(m, indent=1))
print(json.dumps(m, indent=1))
