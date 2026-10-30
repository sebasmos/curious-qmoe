"""Writes run_manifest.json of a T3/T5 results folder: environment, commits, script hashes. python t3t5_manifest.py --run DIR"""
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); a = ap.parse_args()
import numpy, pandas, scipy, sklearn, torch
sh = lambda *c: subprocess.run(c, capture_output=True, text=True, cwd=Path(__file__).resolve().parent).stdout.strip()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
m = dict(machine=platform.platform(), processor=platform.processor(), python=platform.python_version(), numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, torch=torch.__version__,
         threads="OMP/MKL/OPENBLAS = 2, threadpool_limits(2), nice -n 10, CPU only", branch=sh("git", "rev-parse", "--abbrev-ref", "HEAD"), head=sh("git", "rev-parse", "HEAD"), log=sh("git", "log", "--format=%h %an <%ae> %s", "-12"),
         scripts={p.name: sha(p) for p in sorted((a.run / "scripts").glob("*")) if p.is_file()}, stored_inputs_read_only=True, pushed=False)
json.dump(m, open(a.run / "run_manifest.json", "w"), indent=1); print(json.dumps(m, indent=1)[:600])
