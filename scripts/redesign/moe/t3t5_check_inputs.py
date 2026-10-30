"""T3/T5 step 1 (ANALYSIS_PLAN_T3T5.md section 2): are the cheap and strong logits, the cheap embeddings and the audio statistics of every cell stored, finite and row-aligned.
Reads label arrays only to compare them for equality across experts and seeds (no accuracy is computed). python t3t5_check_inputs.py --out DIR"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t3t5_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
pre = json.load(open(T13RUN / "inputs/t13_preselected.json")); probs, gaps, rows = [], [], []
for task in T35:
    partner = [e for e in pre[task]["pair"].split("+") if e != CHEAP]
    if partner != [PAIR[task]] or CHEAP not in pre[task]["pair"].split("+"): probs.append(f"{task}: T13 pair {pre[task]['pair']} differs from the plan's mn10+{PAIR[task]}")
    cells = ["clean"] + (NOISE if task != "tau" else ["shift"]) + (CORR if task != "tau" else [])
    bd = bed(task); N = bd["N"]
    for cell in cells:
        for ex in (CHEAP, PAIR[task]):
            ok = available(task, ex, "clean" if task == "tau" else cell); rows.append(dict(task=task, cell=cell, kind=ex, available=ok))
        if cell in CORR: continue
        ok_e = emb_file(task, cell).exists(); ok_s = stats_file(task, cell).exists()
        if task != "tau" or cell == "clean": rows.append(dict(task=task, cell=cell, kind="emb", available=ok_e)); rows.append(dict(task=task, cell=cell, kind="stats", available=ok_s))
        inpool = cell == "clean" or cell in POOL[task]
        if not (ok_e and ok_s): (probs if inpool else gaps).append(f"{task} {cell}: embedding {ok_e} stats {ok_s}"); continue
        side = json.load(open(emb_file(task, cell).with_suffix(".json")))
        X = emb(task, cell)
        if X.shape != (N, 960) or not np.isfinite(X).all(): probs.append(f"{task} {cell}: embedding shape {X.shape} or non-finite")
        if side["names_crc32"] != names_crc(task): probs.append(f"{task} {cell}: embedding names crc {side['names_crc32']} != csv {names_crc(task)}")
        zs = np.load(stats_file(task, cell), allow_pickle=True)
        if [str(s) for s in zs["names"]] != names(task): probs.append(f"{task} {cell}: audio-statistics names differ from the csv stems")
        S = zs["stats"]
        if S.shape != (N, 10): probs.append(f"{task} {cell}: stats shape {S.shape}")
        if not np.isfinite(S).all(): (probs if inpool else gaps).append(f"{task} {cell}: non-finite statistics ({int((~np.isfinite(S)).any(1).sum())} rows)")
    # logits: rows, folds, inner-validation indices agree between cheap and strong, and across seeds; crc of the score file equals the csv names
    for cell in ["clean"] + (POOL[task] if task != "tau" else ["clean"]):
        for s in TASKS[task]["seeds"]:
            Dc, Ds = cellload(task, CHEAP, cell, s), cellload(task, PAIR[task], cell, s)
            if Dc["y"].shape[0] != N: probs.append(f"{task} {cell} s{s}: rows {Dc['y'].shape[0]} != {N}")
            if not (np.array_equal(Dc["y"], Ds["y"]) and np.array_equal(Dc["fold"], Ds["fold"]) and all(np.array_equal(Dc["iv"][f][0], Ds["iv"][f][0]) for f in Dc["iv"])): probs.append(f"{task} {cell} s{s}: cheap and strong rows/folds/iv differ")
            if not np.array_equal(Dc["y"], bd["y"]): probs.append(f"{task} {cell} s{s}: labels differ from the csv")
            if not np.array_equal(Dc["fold"], bd["fold"]): probs.append(f"{task} {cell} s{s}: folds differ from the csv")
            for nm, D in (("cheap", Dc), ("strong", Ds)):
                if not np.isfinite(D["z"]).all() or not all(np.isfinite(v).all() for _, v in D["iv"].values()): probs.append(f"{task} {cell} s{s} {nm}: non-finite logits")
                if D["crc"] is not None and D["crc"] != names_crc(task): probs.append(f"{task} {cell} s{s} {nm}: score names crc differs")
    if task != "tau":      # noise-detector fit rows: inner-validation clips under the field validation assignment must have finite statistics in their assigned cell
        av = A.field_arrays(bd["N"])[1]; iv = set()
        for s in TASKS[task]["seeds"]:
            for f, (idx, _) in cellload(task, CHEAP, "clean", s)["iv"].items(): iv |= set(idx.tolist())
        iv = np.array(sorted(iv)); bad = [c for c in np.unique(av[iv]) if not np.isfinite(stats(task, c)[iv[av[iv] == c]]).all()]
        if bad: probs.append(f"{task}: non-finite statistics on field-validation rows in cells {bad}")
        rows.append(dict(task=task, cell="field-val-assignment", kind="stats", available=not bad))
    print(task, "checked", flush=True)
Av = pd.DataFrame(rows); Av.to_csv(a.out / "availability.csv", index=False)
json.dump(dict(problems=probs, gaps_outside_pools=gaps), open(a.out / "check_inputs.json", "w"), indent=1)
print("PROBLEMS:", probs[:30] if probs else "none"); print("GAPS outside the pools:", len(gaps))
