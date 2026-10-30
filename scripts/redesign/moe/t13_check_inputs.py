"""T13 step 1: which (task, expert, condition) cells are stored and complete, and do rows, folds and inner-validation indices agree across experts and seeds.
Reads label arrays only to compare them for equality (no accuracy is computed). python t13_check_inputs.py --out DIR"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t13_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
rows, probs, files = [], [], []
for task, T in TASKS.items():
    conds = ["clean", "shift"] if task == "tau" else ["clean"] + NOISE + ["field"] + CORR
    ref = None
    for ex in experts_of(task):
        for cond in conds:
            ok = available(task, ex, cond)
            rows.append(dict(task=task, expert=ex, cond=cond, available=ok))
            if not ok: continue
            for s in T["seeds"]:
                D = load(task, ex, cond, s)
                key = (D["y"].tobytes(), D["fold"].tobytes()); iv = {f: v[0].tobytes() for f, v in D["iv"].items()}
                if task == "tau": key += (D["device"].tobytes(),)
                if ref is None: ref = {"key": key, "n": len(D["y"])}
                if key != ref["key"]: probs.append(f"{task} {ex} {cond} s{s}: y/fold/device rows differ from the reference")
                if s not in ref: ref[s] = iv
                elif iv != ref[s]: probs.append(f"{task} {ex} {cond} s{s}: inner-validation indices differ")
                if cond != "field" or True:
                    if not np.isfinite(D["z"]).all(): probs.append(f"{task} {ex} {cond} s{s}: non-finite test logits")
                if task == "tau" or cond in ("clean",): pass
    # cell files with sha
    for ex in experts_of(task):
        for cond in conds:
            if not available(task, ex, cond) or cond == "field": continue
            for s in T["seeds"]:
                p = tau_file(ex, s) if task == "tau" else (lambda dc: dc[0] / f"{dc[1]}__{cond}.npz")(sdir(task, ex, s, cond in CORR))
                files.append(dict(task=task, expert=ex, cond=cond, seed=s, path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
Av = pd.DataFrame(rows); Av.to_csv(a.out / "availability.csv", index=False); pd.DataFrame(files).to_csv(a.out / "file_list.csv", index=False)
json.dump(dict(problems=probs, n_files=len(files)), open(a.out / "check_inputs.json", "w"), indent=1)
piv = Av.pivot_table(index=["task", "cond"], columns="expert", values="available", aggfunc="first", sort=False)
print(piv.to_string()); print("PROBLEMS:", probs[:20] if probs else "none", "files", len(files))
