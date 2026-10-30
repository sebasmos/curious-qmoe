"""Check 5: folds, grouped inner validation, heads on training rows only, thresholds on clean inner validation, noise and calibration
disjoint from the test data (including Freesound recording overlap between ESC-50 and UrbanSound8K)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_core import R, S2, TASK, NEW, LEG, npz

out = {}
for task, embname in (("esc50", "esc50"), ("urban", "urban8k")):
    t = TASK[task]
    meta = pd.read_csv(t["csv"], usecols=["folder", "name", t["label"]])
    stems = meta["name"].map(lambda s: Path(s).stem).to_numpy()
    rec = np.array([s.split("-")[t["rec_token"]] for s in stems])
    fold = meta["folder"].to_numpy()
    y = meta[t["label"]].to_numpy()
    bad = []
    n_checked = 0
    for cfg_dir in sorted((R / "heads" / task).iterdir()):
        for sd in sorted(cfg_dir.glob("s*")):
            summ = sd.parent.parent.parent / "scores" / task / cfg_dir.name / sd.name / "summary.json"
            S = json.loads(summ.read_text()) if summ.exists() else None
            for fd in sorted(sd.glob("fold_*")):
                sp = json.loads((fd / "split.json").read_text())
                k = sp["fold"]
                te, iv = np.array(sp["test_idx"]), np.array(sp["inner_val_idx"])
                tr = np.flatnonzero(fold != k)
                fit = np.setdiff1d(tr, iv)
                n_checked += 1
                if not np.array_equal(np.sort(te), np.flatnonzero(fold == k)):
                    bad.append((cfg_dir.name, sd.name, k, "test rows differ from the official fold"))
                if np.isin(iv, te).any():
                    bad.append((cfg_dir.name, sd.name, k, "inner validation overlaps test"))
                if np.isin(rec[iv], rec[fit]).any():
                    bad.append((cfg_dir.name, sd.name, k, f"{np.isin(rec[iv], rec[fit]).sum()} iv rows share a recording with fit rows"))
                if S is not None:
                    fs = [f for f in S["folds"] if f["fold"] == k]
                    if fs and fs[0]["n_fit"] + fs[0]["n_iv"] != len(tr):
                        bad.append((cfg_dir.name, sd.name, k, "n_fit + n_iv != training rows"))
                    if fs and fs[0]["n_iv"] != len(iv):
                        bad.append((cfg_dir.name, sd.name, k, "n_iv differs from split"))
                # score file iv rows equal the split, labels equal the CSV
                sc = R / "scores" / task / cfg_dir.name / sd.name / f"{cfg_dir.name.replace('L_', '')}__clean.npz"
                if sc.exists():
                    z = npz(sc)
                    if not np.array_equal(z["iv"][k][0], iv):
                        bad.append((cfg_dir.name, sd.name, k, "score iv rows differ from split"))
                    if not (np.array_equal(z["y"], y) and np.array_equal(z["fold"], fold)):
                        bad.append((cfg_dir.name, sd.name, k, "score labels or folds differ from CSV"))
    # inner validation identical across configurations for the same seed (the cascade pairs tiers on these rows)
    iv_same = True
    for s in t["seeds"]:
        ref = npz(R / "scores" / task / "n4a4" / f"s{s}" / "n4a4__clean.npz")["iv"]
        for c in ["w8a8"] + [f"L_{l}" for l in LEG]:
            stem = c.replace("L_", "")
            o = npz(R / "scores" / task / c / f"s{s}" / f"{stem}__clean.npz")["iv"]
            iv_same &= all(np.array_equal(ref[k][0], o[k][0]) for k in ref)
        st = npz(S2 / f"scores_ast_short/{task}/astshort_{task}_s{s}/esc/w8g64__clean.npz")["iv"]
        iv_same &= all(np.array_equal(ref[k][0], st[k][0]) for k in ref)
    # noise plan
    other = "urban" if task == "esc50" else "esc50"
    om = pd.read_csv(TASK[other]["csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).to_numpy()
    orec = np.array([s.split("-")[TASK[other]["rec_token"]] for s in om])
    plan_t19 = json.loads((R / "emb" / embname / "bg_plan.json").read_text())
    plan_s2 = json.loads((S2 / "emb_ast_short" / embname / "bg_plan.json").read_text())
    same_plan = all(plan_t19.get(k) == v for k, v in plan_s2.items()) and set(plan_t19) == set(plan_s2)
    bg_in_task = sum(v["bg_clip"] in set(stems) for v in plan_t19.values())
    bg_from_other = all(v["bg_clip"] in set(om) for v in plan_t19.values())
    own_rec = 0
    same_fold_rec = 0
    any_rec = 0
    recset = set(rec)
    stem_i = {s: i for i, s in enumerate(stems)}
    for clip, v in plan_t19.items():
        br = v["bg_clip"].split("-")[TASK[other]["rec_token"]]
        i = stem_i[clip]
        own_rec += br == rec[i]
        any_rec += br in recset
        same_fold_rec += br in set(rec[fold == fold[i]])
    # calibration inputs
    cal = json.loads((R / "calib" / f"{task}.json").read_text())
    cclips = [c for inp in cal["inputs"] for c in inp["clips"]]
    cal_in_task = sum(c in set(stems) for c in cclips)
    cal_rec_overlap = sum(c.split("-")[TASK[other]["rec_token"]] in recset for c in cclips)
    out[task] = dict(split_dirs_checked=n_checked, problems=bad[:20], n_problems=len(bad), iv_rows_identical_across_systems=bool(iv_same),
                     bg_plan_equals_stage2=bool(same_plan), bg_clips=len(plan_t19), bg_clip_is_a_task_clip=int(bg_in_task),
                     bg_all_from_other_dataset=bool(bg_from_other), bg_shares_recording_with_its_own_clip=int(own_rec),
                     bg_shares_recording_with_a_clip_of_same_fold=int(same_fold_rec), bg_shares_recording_with_any_task_clip=int(any_rec),
                     calib_clips=len(cclips), calib_clip_is_task_clip=int(cal_in_task), calib_clip_shares_recording_with_task=int(cal_rec_overlap),
                     calib_sources=cal["sources"])
print(json.dumps(out, indent=1))
json.dump(out, open("leakage.json", "w"), indent=1)
