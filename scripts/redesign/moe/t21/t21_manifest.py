"""run_manifest.json for the T21 results folder: commits, machine, software versions, dataset checksums."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

R = Path(sys.argv[1])
W = Path.home() / "Documents/qmoe-work/t21"


def run(*c):
    return subprocess.run(c, capture_output=True, text=True).stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


import numpy, pandas, scipy, sklearn, torch, transformers  # noqa: E402,E401

D1 = Path.home() / "Documents/DATASETS/FSDKaggle2018"
D2 = Path.home() / "Documents/DATASETS/FSD50K"
m = {"experiment": "T21", "branch": run("git", "-C", str(W), "rev-parse", "--abbrev-ref", "HEAD"),
     "head_commit": run("git", "-C", str(W), "rev-parse", "HEAD"), "plan_commit": "731570e",
     "git_log": run("git", "-C", str(W), "log", "--oneline", "94f6af6..HEAD").splitlines(),
     "machine": {"platform": platform.platform(), "machine": platform.machine(), "chip": run("sysctl", "-n", "machdep.cpu.brand_string"),
                 "memory_bytes": run("sysctl", "-n", "hw.memsize")},
     "python": sys.version, "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": numpy.__version__,
                                          "pandas": pandas.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                                          "onnxruntime": run(os.environ.get("PYO", "python"), "-c", "import onnxruntime; print(onnxruntime.__version__)")},
     "datasets": {"FSDKaggle2018": {"zenodo": "2552860", "md5_verified": {"audio_test.zip": "f85c8665c3fc39311ea1b748e194a81d",
                                                                          "audio_train.zip": "a8b47ff4b52022c178f88fa5c6f080d0",
                                                                          "meta.zip": "f16828ac8be0e5285b9175a12f90d784", "doc.zip": "4f2d1ac88f33a62f9db3108b269ee1b7"},
                                    "licence": "CC BY 4.0 (dataset); clips CC BY, CC0, CC BY-NC (3 clips)",
                                    "meta_sha256": {p.name: sha(p) for p in sorted((D1 / "meta_x/FSDKaggle2018.meta").glob("*.csv"))},
                                    "n_wav": len(list((D1 / "audio_raw").rglob("*.wav")))},
                  "FSD50K": {"zenodo": "4060432", "md5_verified": {
                      "FSD50K.dev_audio.z01": "faa7cf4cc076fc34a44a479a5ed862a3", "FSD50K.dev_audio.z02": "8f9b66153e68571164fb1315d00bc7bc",
                      "FSD50K.dev_audio.z03": "1196ef47d267a993d30fa98af54b7159", "FSD50K.dev_audio.z04": "d088ac4e11ba53daf9f7574c11cccac9",
                      "FSD50K.dev_audio.z05": "81356521aa159accd3c35de22da28c7f", "FSD50K.dev_audio.zip": "c480d119b8f7a7e32fdb58f3ea4d6c5a",
                      "FSD50K.eval_audio.z01": "3090670eaeecc013ca1ff84fe4442aeb", "FSD50K.eval_audio.zip": "6fa47636c3a3ad5c7dfeba99f2637982",
                      "FSD50K.metadata.zip": "b9ea0c829a411c1d42adb9da539ed237", "FSD50K.ground_truth.zip": "ca27382c195e37d2269c4c866dd73485"},
                             "licence": "CC BY 4.0 (dataset); qualifying dev clips CC BY 3.0, CC0, CC BY-NC 3.0, Sampling+ 1.0 (counts in csv/d2/classes.json)",
                             "ground_truth_sha256": {p.name: sha(p) for p in sorted((D2 / "meta_x/FSD50K.ground_truth").glob("*.csv"))},
                             "n_wav_dev_subset": len(list((D2 / "audio_raw/dev").glob("*.wav"))) if (D2 / "audio_raw/dev").exists() else 0,
                             "n_wav_eval_subset": len(list((D2 / "audio_raw/eval").glob("*.wav"))) if (D2 / "audio_raw/eval").exists() else 0},
                  "UrbanSound8K_background": {"csv": str(Path.home() / "Documents/DATASETS/Urban8k/ast_768/urbansound8k.csv")}},
     "weights": {"mn10": sha(Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1/mn10_as_mAP_471.pt"),
                 "ast": "MIT/ast-finetuned-audioset-10-10-0.4593 @ f826b80d28226b62986cc218e5cec390b1096902"}}
(R / "run_manifest.json").write_text(json.dumps(m, indent=1))
print("manifest written")
