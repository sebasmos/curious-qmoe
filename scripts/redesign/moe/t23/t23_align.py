"""T23 alignment gate (ANALYSIS_PLAN.md C7, amendment 1). Compares our rows, folds, inner-validation splits, training rows, background draws,
lengths and corruptions with the stored arrays, clip by clip. Reads no logits.
  --stage pre  (after the heads, before any scoring): per task, writes R/inputs/alignment_pre_<task>.json; t23_ft_score.py refuses without all_ok.
  --stage post (after scoring, before any analysis): also checks the score files, noise draws and corruption levels the condition pass wrote;
               writes R/inputs/alignment.json; t23_analyze.py and t23_recompute.py refuse to run unless all_ok.
    python t23_align.py --r R --stage pre --tasks esc50"""
import argparse, json, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2"))
from common import Q, S2, S3B, S3C  # noqa: E402
from t13_common import FINAL, sdir, tau_file, NOISE, CORR  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--stage", choices=["pre", "post"], required=True)
ap.add_argument("--tasks", nargs="+", default=["esc50", "urban", "tau"]); a = ap.parse_args()
T15 = Q / "2026-10-25_1415_moe-t15-t18_code-2ef25b0"; DS = Path.home() / "Documents/DATASETS"
META = {"esc50": DS / "ESC-50/ast_768/esc-50.csv", "urban": DS / "Urban8k/ast_768/urbansound8k.csv", "tau": DS / "TAU2020Mobile/ast_768/tau2020mobile.csv"}
SEEDS = [0, 1, 2]; NF = {"esc50": 5, "urban": 10, "tau": 5}
BGREF = {"esc50": [S3B / "emb_cnn/esc50", S3C / "emb_cnn/esc50", S2 / "emb_ast_short/esc50"],
         "urban": [S3B / "emb_cnn/urban8k", S3C / "emb_cnn/urban8k_unpadded", S2 / "emb_ast_short/urban8k"]}
STORED = {"esc50": ["mn10", "astshort", "mn20", "astw4", "astfp32"], "urban": ["mn10", "astshort", "mn20", "astw4", "astfp32"], "tau": ["mn10", "astw4", "mn20", "astfp32"]}
res, problems = {}, []
def bad(task, msg): problems.append(f"{task}: {msg}")
for task in a.tasks:
    r = res[task] = {}
    meta = pd.read_csv(META[task]); stems = meta["name"].map(lambda s: Path(s).stem).tolist(); n = len(stems)
    crc = zlib.crc32("\n".join(stems).encode()); r["n"] = n; r["names_crc32_csv"] = crc
    n_a = int((meta.device == "a").sum()) if task == "tau" else n; folder = meta.folder.to_numpy()
    if task == "tau" and not (meta.device[:n_a] == "a").all(): bad(task, "device-A rows are not the leading block")
    E = a.r / f"emb/{task}"
    L, L15 = np.load(E / "lengths.npy"), np.load(T15 / f"emb/{task}/lengths.npy")
    r["lengths_equal_t15"] = bool(np.array_equal(L, L15)); r["lengths_equal_t15"] or bad(task, "lengths differ from T15's lengths.npy")
    y_csv = (meta["label"] if task == "esc50" else meta["class_id"]).to_numpy(np.int64)
    # our heads and tails, per (seed, fold): test rows, inner validation, training rows of the tail
    ours, leak = {}, []
    for seed in SEEDS:
        fold = np.full(n_a, -1); iv = {}
        for f in range(1, NF[task] + 1):
            run = a.r / f"outputs_ft/wg_{task}_s{seed}_f{f}/esc"; od = a.r / f"ft/{task}/f{f}_s{seed}"
            for p in (run / "summary.json", run / "scaler.pkl", run / "check.npz", run / f"fold_{f}/split.json", od / "tail_q8.pt", od / "train_rows.npy"):
                if not p.exists(): bad(task, f"missing {p.relative_to(a.r)}")
            if not (run / f"fold_{f}/split.json").exists(): continue
            sp = json.loads((run / f"fold_{f}/split.json").read_text()); te = np.asarray(sp["test_idx"]); iv[f] = np.asarray(sp["inner_val_idx"]).astype(int)
            if not np.array_equal(te, np.flatnonzero(folder[:n_a] == f)): bad(task, f"s{seed} f{f}: head test rows are not the fold's rows")
            fold[te] = f
            if (od / "train_rows.npy").exists():
                tr = np.load(od / "train_rows.npy"); rule = np.setdiff1d(np.setdiff1d(np.arange(n_a), np.flatnonzero(folder[:n_a] == f)), iv[f])
                d = dict(seed=seed, fold=f, n_train=len(tr), n_test_in_train=int(np.isin(tr, np.flatnonzero(folder == f)).sum()),
                         n_iv_in_train=int(np.isin(tr, iv[f]).sum()), n_other_device_in_train=int((meta.device.to_numpy()[tr] != "a").sum()) if task == "tau" else 0,
                         equals_rule=bool(np.array_equal(np.sort(tr), rule)))
                leak.append(d)
                if d["n_test_in_train"] or d["n_iv_in_train"] or d["n_other_device_in_train"] or not d["equals_rule"]: bad(task, f"tail training rows s{seed} f{f}: {d}")
        if (fold < 0).any(): bad(task, f"s{seed}: {int((fold < 0).sum())} rows in no test fold")
        ours[seed] = (fold, iv)
    r["tail_training_rows"] = leak
    checked = []
    for e in STORED[task]:
        for seed in SEEDS:
            fold, iv = ours[seed]
            if task == "tau":
                fp = tau_file(e, seed)
                if not fp.exists(): continue
                files = [(fp, np.load(fp))]
            else:
                files = []
                for corrupt in (False, True):
                    if corrupt and e in ("astw4", "astfp32"): continue
                    d, cfg = sdir(task, e, seed, corrupt)
                    fp = d / f"{cfg}__{'clean' if not corrupt else CORR[0]}.npz"
                    if fp.exists(): files.append((fp, np.load(fp)))
            for fp, z in files:
                ok = {"crc": ("names_crc32" not in z.files) or int(z["names_crc32"]) == crc, "y": bool(np.array_equal(z["y"].astype(np.int64), y_csv))}
                if task == "tau":
                    ok["fold"] = bool(np.array_equal(z["fold"].astype(int), folder)); ok["fold_deviceA_split"] = bool(np.array_equal(fold, folder[:n_a]))
                    ok["device"] = bool(np.array_equal(z["device"].astype(str), meta.device.to_numpy().astype(str))); ok["n_a"] = int(z["n_a"]) == n_a
                else:
                    ok["fold"] = bool(np.array_equal(z["test_fold"].astype(int), fold))
                ok["iv"] = len(iv) == NF[task] and all(np.array_equal(z[f"iv_idx_{f}"].astype(int), iv[f]) for f in iv)
                ok["iv_set"] = len(iv) == NF[task] and all(np.array_equal(np.sort(z[f"iv_idx_{f}"].astype(int)), np.sort(iv[f])) for f in iv)
                checked.append(dict(expert=e, seed=seed, file=str(fp.relative_to(Q)), **ok))
                if not all(ok.values()): bad(task, f"{fp.name} s{seed}: {[k for k, v in ok.items() if not v]}")
    r["stored_files_checked"] = len(checked); r["iv_same_order_all"] = bool(checked) and all(c["iv"] for c in checked)
    r["mismatches"] = [c for c in checked if not all(v for k, v in c.items() if k not in ("expert", "seed", "file"))]
    if not checked: bad(task, "no stored file checked")
    if a.stage == "pre": continue
    # post: score files, sidecars, noise draws, corruption levels
    need = ["clean"] + (NOISE + CORR if task != "tau" else [])
    r["missing_conditions"] = [c for c in need if not (E / f"wg__{c}.json").exists()]
    if r["missing_conditions"]: bad(task, f"missing conditions {r['missing_conditions']}")
    nsc = 0
    for c in need:
        if (E / f"wg__{c}.json").exists():
            sd = json.loads((E / f"wg__{c}.json").read_text())
            if sd["names_crc32"] != crc or sd["n_clips"] != n: bad(task, f"wg__{c}.json: names checksum or row count differs")
        for seed in SEEDS:
            fp = a.r / f"scores/{task}/wg_{task}_s{seed}/esc/" / ("wg__all.npz" if task == "tau" else f"wg__{c}.npz")
            if not fp.exists(): bad(task, f"missing score file {fp.relative_to(a.r)}"); continue
            z = np.load(fp); fold, iv = ours[seed]; nsc += 1
            ok = int(z["names_crc32"]) == crc and np.array_equal(z["y"], y_csv) and np.isfinite(z["test_logits"]).all() and z["test_logits"].shape == (n, len(np.unique(y_csv)))
            ok = ok and all(np.array_equal(z[f"iv_idx_{f}"], iv[f]) and np.isfinite(z[f"iv_logits_{f}"]).all() for f in iv)
            ok = ok and (np.array_equal(z["fold"], folder) if task == "tau" else np.array_equal(z["test_fold"].astype(int), fold))
            if not ok: bad(task, f"score file {fp.relative_to(a.r)} differs in rows, folds, splits or has non-finite logits")
    r["score_files_checked"] = nsc
    if task == "tau": continue
    mine = json.loads((E / "bg_plan.json").read_text()) if (E / "bg_plan.json").exists() else {}
    refs = [json.loads((d / "bg_plan.json").read_text()) for d in BGREF[task]]
    r["bg_refs"] = [str(d.relative_to(Q)) for d in BGREF[task]]
    r["bg_refs_agree"] = all(set(ref) == set(refs[0]) and all(ref[k] == v for k, v in refs[0].items()) for ref in refs[1:])
    r["bg_plan_clips"] = len(mine); r["bg_plan_ref_clips"] = len(refs[0])
    r["bg_plan_equal"] = bool(set(mine) == set(refs[0]) and all(refs[0][k] == v for k, v in mine.items()))
    if not (r["bg_refs_agree"] and r["bg_plan_equal"]): bad(task, "bg_plan differs")
    lv = {}
    for c in CORR:
        ours_s = json.loads((E / f"wg__{c}.json").read_text()) if (E / f"wg__{c}.json").exists() else {}
        ref = FINAL / f"emb_cnn/{'esc50' if task == 'esc50' else 'urban8k_unpadded'}/mn10_w8a8__{c}.json"
        if not ref.exists(): ref = FINAL / f"emb_cnn/{'esc50' if task == 'esc50' else 'urban8k'}/mn10_w8a8__{c}.json"
        rv = json.loads(ref.read_text()).get("mean_level_change_db") if ref.exists() else None
        lv[c] = dict(ours=ours_s.get("mean_level_change_db"), final=rv, ref=str(ref.relative_to(Q)) if ref.exists() else None)
        if rv is None or ours_s.get("mean_level_change_db") is None or abs(rv - ours_s["mean_level_change_db"]) > 1e-4: bad(task, f"{c}: level change differs or missing")
    r["corruption_level_change_db"] = lv
for t in res:
    if not res[t].get("iv_same_order_all", False): problems.append(f"{t}: inner-validation order differs")
out = {"stage": a.stage, "all_ok": not problems, "problems": problems, "tasks": res}
(a.r / "inputs").mkdir(exist_ok=True)
if a.stage == "pre":
    for t in res: (a.r / f"inputs/alignment_pre_{t}.json").write_text(json.dumps({"stage": "pre", "all_ok": not [p for p in problems if p.startswith(t + ":")],
                                                                                  "problems": [p for p in problems if p.startswith(t + ":")], "task": res[t]}, indent=1, default=str))
else:
    (a.r / "inputs/alignment.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({"stage": a.stage, "all_ok": out["all_ok"], "problems": problems[:20]}, indent=1))
sys.exit(0 if out["all_ok"] else 1)
