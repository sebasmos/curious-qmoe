"""T1b step 1: check the stored E2 embeddings and the pilot raw files. python t1b_check_inputs.py --out DIR --pilot PILOTDIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Q, I5, OPTB, sha256, write_json
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--pilot", required=True, type=Path); a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
S2 = Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
URBAN = {"E2a": I5 / "emb/urban8k/fp32__clean.npy", "E2b": I5 / "emb/urban8k/w4g128__clean.npy", "E2c": S2 / "emb_ast_short/urban8k/w4g128__clean.npy"}
TAU = {"E2a": OPTB / "emb_ast/tau/fp32__clean.npy", "E2b": OPTB / "emb_ast/tau/w4g128__clean.npy"}
rep, rows = {}, []
for ds, paths, n, csv in (("urban", URBAN, 8732, "Urban8k"), ("tau", TAU, 23035, "TAU2020Mobile")):
    for k, p in paths.items():
        X = np.load(p); js = json.loads(p.with_suffix(".json").read_text())
        ok = dict(shape=list(X.shape), finite=bool(np.isfinite(X).all()), zero_rows=int((np.abs(X).sum(1) == 0).sum()), config=js.get("config"), max_length=js.get("max_length"),
                  n_clips=js.get("n_clips"), rows_extracted=js.get("rows_extracted"), silent_left=js.get("silent_clips_left_clean"), csv=js.get("csv"))
        assert X.shape == (n, 768) and ok["finite"] and ok["zero_rows"] == 0 and ok["rows_extracted"] == n and csv in ok["csv"], (ds, k, ok)
        rep[f"{ds}_{k}"] = ok; rows.append(dict(ds=ds, name=k, path=str(p), bytes=p.stat().st_size, sha256=sha256(p)))
for ds, sub, nf in (("urban", "t1/raw", 10), ("tau", "tau/raw", 5)):
    fs = sorted((a.pilot / sub).glob("s*_f*.npz")); assert len(fs) == 3 * nf, (ds, len(fs))
    R = np.load(fs[0]); need = {"tr", "te", "z1_oof", "S2", "tier", "fit_p", "iv_p", "tau1", "tau1_oof", "z1_te", "z1_iv", "flag"} | ({"cond"} if ds == "tau" else set())
    assert need <= set(R.files), need - set(R.files)
    rep[f"pilot_raw_{ds}"] = dict(n_files=len(fs), keys_ok=True)
pd.DataFrame(rows).to_csv(a.out / "file_list.csv", index=False); write_json(a.out / "check_inputs.json", rep); print(json.dumps(rep, indent=1))
