"""T22 extraction (ANALYSIS_PLAN.md B, C1, C2): the W8A8 coarse expert's logits under every condition tier 2 has, on the CPU.
Audio: stage 2's load_clip, NoiseMaker and mix, the final round's corrupt.corrupt, called as the stage 3b and final-round extractors
call them (same CSVs, background CSVs and folders), then zero padding to the task's length and the coarse view. Each clip runs through
the model of its own test fold (test logits) and through the model of every fold whose inner split (of that seed) holds it (iv logits).
Output per seed, in stage 2's format so analyze.ScoreDir and t13_common read it unchanged:
  scores/<task>/coarse_<task>_s<seed>/coarse_w8a8__<cond>.npz  (y, test_fold, test_logits, names_crc32, iv_idx_<f>, iv_logits_<f>)
  TAU: coarse_w8a8__all.npz (y, fold, device, test_logits, names_crc32, n_a, iv_idx_<f>, iv_logits_<f>)
and scores/<task>/bg_plan.json (background clip and offset per clip), lengths16k.npy.
    python t22_extract.py --r R --task esc50 --conds clean bg__20db ...   (default: every condition of the task)"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t22_common import *  # noqa
import t22_view as V, t22_models as TM
import corrupt as CR
from audio_shift import NoiseMaker, index_wavs, load_clip, mix
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--task", required=True)
ap.add_argument("--conds", nargs="+", default=None); ap.add_argument("--chunk", type=int, default=256); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", type=Path, default=None); a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
task = a.task; T = TASKS[task]; K = T["K"]; nf = T["nf"]
conds = a.conds or (["all"] if task == "tau" else ["clean"] + NOISE + CORR)
B = bed(task); st = stems(task); N = B["N"]; crc = names_crc(task)
assert crc == B["crc"], "CSV row order differs from the stored arrays"
index = index_wavs(T["aud"])
nm = NoiseMaker([], {}, False)
if any(c.startswith("bg") for c in conds):
    bgn = pd.read_csv(T["bg_csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist(); bgi = index_wavs(T["bg_aud"])
    nm = NoiseMaker([b for b in bgn if b in bgi], bgi, True)
cache_clean = task != "tau"; clean = {}


def wave(i, c):
    if i not in clean:
        x = load_clip(index[st[i]])
        if not cache_clean:
            return x if c in ("clean", "all") else None
        clean[i] = x
    x = clean[i]
    if c in ("clean", "all"):
        return x
    kind, lv = cond_arg(c)
    if kind in CR.KINDS:
        return CR.corrupt(x, kind, lv)
    y, _ = mix(x, nm(st[i], kind, len(x)), lv)
    return y


rows_all = np.arange(N) if a.limit is None else np.arange(a.limit)
models, need = {}, {}
for s in SEEDS:
    for f in range(1, nf + 1):
        q = V.quantize_w8a8(TM.load_coarse(a.r / "runs" / f"{task}_f{f}_s{s}.pt", K), "cpu")
        iv = inner_split(task, s, f)
        models[(s, f)] = q; need[(s, f)] = (B["fold"] == f, np.isin(np.arange(N), iv), iv)
out = (a.out or a.r / "scores") / task; out.mkdir(parents=True, exist_ok=True)
lens = np.zeros(N, np.int64)
for c in conds:
    files = {s: out / f"coarse_{task}_s{s}" / f"coarse_w8a8__{c}.npz" for s in SEEDS}
    if all(p.exists() for p in files.values()):
        print("skip", c, flush=True); continue
    t0 = time.time()
    Z = {s: np.full((N, K), np.nan, np.float32) for s in SEEDS}
    IV = {(s, f): np.full((len(need[(s, f)][2]), K), np.nan, np.float32) for s in SEEDS for f in range(1, nf + 1)}
    ivpos = {(s, f): {int(r): j for j, r in enumerate(need[(s, f)][2])} for s in SEEDS for f in range(1, nf + 1)}
    for b0 in range(0, len(rows_all), a.chunk):
        rr = rows_all[b0:b0 + a.chunk]
        w = [wave(int(i), c) for i in rr]
        lens[rr] = [len(x) for x in w]
        with torch.no_grad():
            X = V.logmel(torch.from_numpy(np.stack([V.pad16(x, task) for x in w])))[:, 0].numpy()
        for (s, f), q in models.items():
            mt, mi, _ = need[(s, f)]
            sel = np.flatnonzero(mt[rr] | mi[rr])
            if not len(sel): continue
            lg = TM.run(q, X[sel])
            g = rr[sel]
            t_ = mt[g]; Z[s][g[t_]] = lg[t_]
            for j in np.flatnonzero(mi[g]): IV[(s, f)][ivpos[(s, f)][int(g[j])]] = lg[j]
        print(f"  {task} {c}: {b0 + len(rr)}/{len(rows_all)} {time.time() - t0:.0f}s", flush=True)
    for s in SEEDS:
        if a.limit is None:
            assert np.isfinite(Z[s]).all() and all(np.isfinite(IV[(s, f)]).all() for f in range(1, nf + 1)), f"{c} s{s}: missing logits"
        files[s].parent.mkdir(parents=True, exist_ok=True)
        ivd = {}
        for f in range(1, nf + 1):
            ivd[f"iv_idx_{f}"] = need[(s, f)][2]; ivd[f"iv_logits_{f}"] = IV[(s, f)]
        if task == "tau":
            np.savez(files[s], y=B["y"], fold=B["fold"].astype(np.int16), device=B["device"], test_logits=Z[s], names_crc32=np.int64(crc),
                     n_a=np.int64(B["is_a"].sum()), **ivd)
        else:
            np.savez(files[s], y=B["y"], test_fold=B["fold"].astype(np.int16), test_logits=Z[s], names_crc32=np.int64(crc), **ivd)
    np.save(out / f"lengths16k__{c}.npy", lens)
    if c.startswith("bg"):
        pf = out / "bg_plan.json"; old = json.loads(pf.read_text()) if pf.exists() else {}; old.update(nm.plan); pf.write_text(json.dumps(old, indent=0))
    print(f"{task} {c} done in {time.time() - t0:.0f}s", flush=True)
print("extraction done", task, flush=True)
