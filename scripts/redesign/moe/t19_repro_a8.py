"""T19 B2.4 check: the T19 mode in its A8 reference setting (no smoothing, no clipping, symmetric Linear inputs, 16-bit P) reproduces
the stored idea 6 stage 2 embeddings (ESC-50, 512 frames, w4g128 and w8g64, clean) to 1e-5 on 20 clips; the legacy mode is run too.
Clips 0 to 31 in two batches of 16 (the stored run's batch layout), the first 20 compared. Writes <results>/logs/repro_a8.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t19_common import MODEL, REVISION, S2, TASKS, results_dir  # noqa: E402
from QWave.ast_frames import shorten_ast  # noqa: E402
from QWave.backbone_quant import QuantAST, T19Act, parse_config  # noqa: E402
from audio_shift import index_wavs, load_clip  # noqa: E402
from transformers import ASTFeatureExtractor, ASTModel  # noqa: E402

t = TASKS["esc50"]
names = pd.read_csv(t["csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()[:32]
idx = index_wavs(t["audio"])
ext = ASTFeatureExtractor.from_pretrained(MODEL, revision=REVISION, max_length=512)
feats = ext([load_clip(idx[n]) for n in names], sampling_rate=16000, return_tensors="pt")["input_values"].to("mps")
model = shorten_ast(ASTModel.from_pretrained(MODEL, revision=REVISION), 512).to("mps").eval()
qa = QuantAST(model, p_bits=16)
out = {}
for cfg in ("w4g128", "w8g64"):
    stored = np.load(S2 / f"emb_ast_short/esc50/{cfg}__clean.npy")[:20]
    spec, _ = parse_config(cfg)
    qa.set_t19(spec, T19Act(layer_bits=(8,) * 12, p_bits=16, smooth=None, clip=None, asym=False))
    new = torch.cat([qa(feats[i:i + 16]).float().cpu() for i in (0, 16)]).numpy()[:20]
    qa.set_config(spec, 8)
    leg = torch.cat([qa(feats[i:i + 16]).float().cpu() for i in (0, 16)]).numpy()[:20]
    out[cfg] = {"max_abs_t19_vs_stored": float(np.abs(new - stored).max()), "max_abs_legacy_vs_stored": float(np.abs(leg - stored).max()),
                "max_abs_t19_vs_legacy": float(np.abs(new - leg).max()), "max_abs_stored": float(np.abs(stored).max())}
    out[cfg]["pass_1e-5"] = out[cfg]["max_abs_t19_vs_stored"] <= 1e-5
print(json.dumps(out, indent=1))
(results_dir() / "logs/repro_a8.json").write_text(json.dumps(out, indent=1))
