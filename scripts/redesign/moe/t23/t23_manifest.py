"""run_manifest.json for the T23 results folder: commits, machine, software versions, checkpoint and dataset CSV checksums."""
import hashlib, json, platform, subprocess, sys
from pathlib import Path
R = Path(sys.argv[1]); W = Path.home() / "Documents/qmoe-work/t23"
def run(*c): return subprocess.run(c, capture_output=True, text=True).stdout.strip()
def dig(p, a="sha256"):
    h = hashlib.new(a)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
import numpy, pandas, scipy, sklearn, torch, torchaudio  # noqa: E401,E402
DS = Path.home() / "Documents/DATASETS"; CK = Path.home() / "Documents/qmoe-weights/panns-wavegram/Wavegram_Cnn14_mAP=0.389.pth"
m = {"experiment": "T23", "branch": run("git", "-C", str(W), "rev-parse", "--abbrev-ref", "HEAD"), "head_commit": run("git", "-C", str(W), "rev-parse", "HEAD"),
     "plan_commit": "5b913db", "amendment_commits": {"1 (after P1 only)": "d36d3fe"}, "git_log": run("git", "-C", str(W), "log", "--oneline", "7957360..HEAD").splitlines(),
     "machine": {"platform": platform.platform(), "chip": run("sysctl", "-n", "machdep.cpu.brand_string"), "memory_bytes": run("sysctl", "-n", "hw.memsize")},
     "python": sys.version, "versions": {"torch": torch.__version__, "torchaudio": torchaudio.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
                                          "scipy": scipy.__version__, "sklearn": sklearn.__version__},
     "checkpoint": {"file": str(CK), "zenodo_record": "3987831", "url": "https://zenodo.org/api/records/3987831/files/Wavegram_Cnn14_mAP=0.389.pth/content",
                    "md5_published": "1e3506ab640371e0b5a417b15fd66d21", "md5": dig(CK, "md5"), "sha256": dig(CK), "licence": "CC BY 4.0 (weights), MIT (code, not run)"},
     "dataset_csv_sha256": {k: dig(v) for k, v in {"esc50": DS / "ESC-50/ast_768/esc-50.csv", "urban": DS / "Urban8k/ast_768/urbansound8k.csv",
                                                   "tau": DS / "TAU2020Mobile/ast_768/tau2020mobile.csv"}.items()}}
(R / "run_manifest.json").write_text(json.dumps(m, indent=1)); print("manifest written")
