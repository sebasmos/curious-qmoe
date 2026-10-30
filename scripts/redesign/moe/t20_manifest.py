"""T20 run_manifest.json: commit, machine, software versions, dataset CSV sha256, inputs read.
    python t20_manifest.py --run RESULTS"""
import argparse, json, platform, subprocess, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); a = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sha256
from t20_common import TASKS, T1C, UNP, S2, S3C, OPTB
import numpy, torch, transformers, sklearn, scipy
g = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
m = dict(branch=g("rev-parse", "--abbrev-ref", "HEAD"), commit=g("rev-parse", "HEAD"), plan_commit="ca935d8", machine=platform.platform(), processor=platform.processor(),
         python=sys.version.split()[0], numpy=numpy.__version__, torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
         model="MIT/ast-finetuned-audioset-10-10-0.4593@f826b80d28226b62986cc218e5cec390b1096902",
         csv_sha256={t: {"csv": str(v["csv"]), "sha256": sha256(v["csv"])} for t, v in TASKS.items()},
         inputs_read_only=[str(p) for p in (T1C, UNP, S2, S3C, OPTB)])
(a.run / "run_manifest.json").write_text(json.dumps(m, indent=1)); print(json.dumps(m, indent=1)[:400])
