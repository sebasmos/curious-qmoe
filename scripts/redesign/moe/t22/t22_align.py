"""T22 alignment check (ANALYSIS_PLAN.md H), hard, before any scoring. Writes inputs/alignment.json; the scorer refuses to run unless
"ok" is true. Compares the coarse expert's score files with the stored tier-2 and tier-3 arrays clip by clip (names checksum, labels,
folds, devices, inner splits), the background plan with both stored plans for every clip, and the audio path with stored MN10
embeddings recomputed through this code path.     python t22_align.py --r R --tasks esc50 urban tau"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t22_common import *  # noqa
import t22_view as V
import t13_common as T13
import analyze as A
import corrupt as CR
from audio_shift import NoiseMaker, index_wavs, load_clip, mix, field_assignment, FIELD_SEEDS
from QWave import efficientat_mn as M
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--tasks", nargs="+", default=["esc50", "urban", "tau"])
ap.add_argument("--n_audio", type=int, default=64); a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
Q = T13.Q if hasattr(T13, "Q") else Path.home() / "Documents/qmoe-results"
S3B, S3C, I5, OPTB, FINAL = T13.S3B, T13.S3C, T13.I5, T13.OPTB, T13.FINAL
PLANS = {"esc50": {"tier2_stage3b": S3B / "emb_cnn/esc50/bg_plan.json", "tier3_idea5_stage2": I5 / "emb/esc50/bg_plan.json"},
         "urban": {"tier2_stage3c_unpadded": S3C / "emb_cnn/urban8k_unpadded/bg_plan.json", "tier3_idea5_stage2": I5 / "emb/urban8k/bg_plan.json"}}
# stored MN10 embeddings for the same-audio check: (file, padded length or 0, configuration)
EMB = {"esc50": [("clean", S3B / "emb_cnn/esc50/mn10_fp32__clean.npy", 0, "fp32"), ("bg__0db", S3B / "emb_cnn/esc50/mn10_fp32__bg__0db.npy", 0, "fp32"),
                 ("reverb_rt600ms", FINAL / "emb_cnn/esc50/mn10_w8a8__reverb_rt600ms.npy", 0, "w8a8")],
       "urban": [("clean", S3B / "emb_cnn/urban8k/mn10_fp32__clean.npy", 64587, "fp32"), ("bg__0db", S3B / "emb_cnn/urban8k/mn10_fp32__bg__0db.npy", 64587, "fp32"),
                 ("reverb_rt600ms", FINAL / "emb_cnn/urban8k_unpadded/mn10_w8a8__reverb_rt600ms.npy", 0, "w8a8")],
       "tau": [("clean", OPTB / "emb_cnn/tau/mn10_fp32__clean.npy", 0, "fp32")]}
rep, ok = {"tasks": {}}, True


def chk(d, name, cond, detail=None):
    global ok
    d[name] = {"pass": bool(cond)} | ({"detail": detail} if detail is not None else {})
    ok &= bool(cond)


base = M.load_mn("mn10", WEIGHTS)
for task in a.tasks:
    T = TASKS[task]; nf = T["nf"]; d = {}; rep["tasks"][task] = d
    crc = names_crc(task)
    sc = a.r / "scores" / task
    conds = ["all"] if task == "tau" else ["clean"] + NOISE + CORR
    # 1-3: rows, labels, folds, devices, inner splits, names checksum, against tier 2 and tier 3 of every seed
    for s in SEEDS:
        for c in conds:
            p = sc / f"coarse_{task}_s{s}" / f"coarse_w8a8__{c}.npz"
            if not p.exists():
                chk(d, f"s{s}_{c}_file", False, f"missing {p}"); continue
            z = np.load(p)
            yk, fk = ("y", "fold") if task == "tau" else ("y", "test_fold")
            refs = [("mn10", "clean" if task == "tau" else c)]
            st_c = "clean" if task == "tau" else c
            if T13.available(task, "astw4", st_c, [s]): refs.append(("astw4", st_c))
            for ex, rc in refs:
                D = T13.load(task, ex, rc, s)
                chk(d, f"s{s}_{c}_{ex}_crc", int(z["names_crc32"]) == D["crc"] == crc)
                chk(d, f"s{s}_{c}_{ex}_y", np.array_equal(z[yk], D["y"]))
                chk(d, f"s{s}_{c}_{ex}_fold", np.array_equal(z[fk].astype(int), D["fold"]))
                if task == "tau": chk(d, f"s{s}_{c}_{ex}_device", np.array_equal(z["device"].astype(str), D["device"]))
                for f in range(1, nf + 1):
                    chk(d, f"s{s}_{c}_{ex}_iv{f}", np.array_equal(z[f"iv_idx_{f}"], D["iv"][f][0]))
            chk(d, f"s{s}_{c}_finite", np.isfinite(z["test_logits"]).all() and all(np.isfinite(z[f"iv_logits_{f}"]).all() for f in range(1, nf + 1)))
    # 4: background plan, every clip, against both stored plans
    if task in PLANS:
        mine = json.loads((sc / "bg_plan.json").read_text())
        for lab, pf in PLANS[task].items():
            ref = json.loads(Path(pf).read_text())
            keys_eq = set(mine) == set(ref)
            diff = [k for k in mine if k in ref and (mine[k]["bg_clip"] != ref[k]["bg_clip"] or int(mine[k]["offset_16k"]) != int(ref[k]["offset_16k"]))]
            chk(d, f"bg_plan_vs_{lab}", keys_eq and not diff, {"n_mine": len(mine), "n_ref": len(ref), "n_differ": len(diff), "first": diff[:5]})
    # 6: field assignment and the assembled field rows of the coarse expert (stage 2's ScoreDir)
    if task != "tau":
        for s in SEEDS:
            g = A.ScoreDir(sc / f"coarse_{task}_s{s}").get("coarse_w8a8", "field")
            chk(d, f"s{s}_field_assembled", np.isfinite(g["z"]).all())
        at, av = A.field_arrays(len(bed(task)["y"]))
        chk(d, "field_assignment_is_stage2", list(at) == field_assignment(len(at), FIELD_SEEDS["test"]) and list(av) == field_assignment(len(av), FIELD_SEEDS["val"]))
    # 5: same audio: native MN10 through this loader, mixer and corruption code vs stored embeddings on n_audio clips
    st = stems(task); index = index_wavs(T["aud"]); B = bed(task)
    pick = np.linspace(0, B["N"] - 1, a.n_audio).astype(int)
    nm = NoiseMaker([], {}, False)
    if T["bg_csv"] is not None:
        bgn = pd.read_csv(T["bg_csv"], usecols=["name"])["name"].map(lambda s_: Path(s_).stem).tolist(); bgi = index_wavs(T["bg_aud"])
        nm = NoiseMaker([b for b in bgn if b in bgi], bgi, True)
    for c, f_, pad, cfg in EMB[task]:
        ref = np.load(f_, mmap_mode="r")
        net = M.build_backbone(cfg, base, "cpu")
        mx, cos = 0.0, 1.0
        for i in pick:
            x = load_clip(index[st[i]])
            if c != "clean":
                kind, lv = cond_arg(c)
                x = CR.corrupt(x, kind, lv) if kind in CR.KINDS else mix(x, nm(st[i], kind, len(x)), lv)[0]
            if pad: x = np.pad(x, (0, max(0, pad - len(x))))
            with torch.no_grad():
                e = net(M.model_input([x])[0][1]).numpy()[0]
            r = np.asarray(ref[i], np.float64)
            mx = max(mx, float(np.abs(e - r).max() / max(np.abs(r).max(), 1e-12)))
            cos = min(cos, float(e @ r / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-30)))
        bar_ok = (mx <= 1e-4) if cfg == "fp32" else (cos >= 0.9999)
        chk(d, f"same_audio_{c}_{cfg}", bar_ok, {"max_abs_diff_rel_to_max": mx, "min_cosine": cos, "n_clips": len(pick), "stored": str(f_)})
    # 7: the extraction's clean coarse view equals the training cache
    X = np.load(CACHE / task / "clean_mel.npy", mmap_mode="r"); crow = np.load(CACHE / task / "clean_rows.npy"); cp = {int(r): j for j, r in enumerate(crow)}
    mx = 0.0
    for i in pick:
        if int(i) not in cp: continue
        with torch.no_grad():
            m_ = V.logmel(torch.from_numpy(V.pad16(load_clip(index[st[i]]), task))[None])[0, 0].numpy()
        mx = max(mx, float(np.abs(m_ - X[cp[int(i)]]).max()))
    chk(d, "clean_view_equals_training_cache", mx == 0.0, {"max_abs_diff": mx})
    print(task, "ok" if all(v["pass"] for v in d.values()) else "PROBLEM", flush=True)
rep["ok"] = bool(ok); rep["failed"] = [f"{t}:{k}" for t, d in rep["tasks"].items() for k, v in d.items() if not v["pass"]]
(a.r / "inputs").mkdir(exist_ok=True); (a.r / "inputs/alignment.json").write_text(json.dumps(rep, indent=1)); print("ok" if ok else f"FAILED {rep['failed'][:20]}")
