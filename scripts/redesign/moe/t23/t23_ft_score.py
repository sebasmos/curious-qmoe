"""T23 full run, condition pass and scorer (ANALYSIS_PLAN.md C3, C4, C5 and amendment 1). One GPU process per task and group of conditions.
Per condition: every clip is loaded (stage 2's load_clip, 16 kHz), the condition applied at 16 kHz (stage 2's mixer and noise draws, the final
run's corrupt.py), zero-padded, resampled to 32 kHz and passed through the frozen W8A8 trunk (wavegram to conv_block5, QuantWavegram, batch 16,
kept in float16 in memory, as the fine-tune cache). Clean audio reuses the fine-tune caches (ESC-50 and UrbanSound8K all rows, TAU the
device-A block; TAU's other devices are computed here). Then, for every seed and fold, the fold's W8A8 tail (tail_q8.pt) and protocol 3 head
(scaler.pkl, best_model.pth) score that fold's test rows and that run's inner-validation rows. Logits are written in the stored layout:
  ESC-50, UrbanSound8K: R/scores/<task>/wg_<task>_s<seed>/esc/wg__<cond>.npz with y, test_fold, test_logits, iv_idx_<k>, iv_logits_<k>, names_crc32
  TAU: R/scores/tau/wg_tau_s<seed>/esc/wg__all.npz with y, fold, device, n_a, names_crc32, test_logits, iv_idx_<k>, iv_logits_<k>
No embedding is written. Before any logit is written: inputs/alignment_pre_<task>.json must say all_ok; every head is re-run on its check.npz
(the training CSV's fold-f test rows) and must reproduce summary.json to 1e-6; noise draws must equal stage 3b's bg_plan.json clip by clip;
corruption level changes must equal the final run's MN10 sidecars to 1e-4; clip lengths must equal the frozen extraction's lengths.npy.
Run with python -I (the checkpoint is a download).
    python -I t23_ft_score.py --r R --task esc50 --conditions clean bg:20 ... [--device mps] [--max-minutes 75]"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--task", required=True, choices=["esc50", "urban", "tau"])
ap.add_argument("--conditions", nargs="+", default=["clean"]); ap.add_argument("--device", default="mps"); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--max-minutes", type=float, default=75.0)
ap.add_argument("--dry-out", type=Path, default=None, help="code test: fold 1 only, no alignment gate, everything written under this folder")
a = ap.parse_args(); T_START = time.time()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = str(a.threads)
for p_ in (ROOT, HERE.parent, HERE, ROOT / "scripts/redesign/idea6/stage2", ROOT / "scripts/redesign/idea6/final"): sys.path.insert(0, str(p_))
import numpy as np, pandas as pd, torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from QWave import efficientat_mn as M  # noqa: E402
from QWave.panns_wavegram import QuantWavegram, load_wavegram  # noqa: E402
from audio_shift import NoiseMaker, cond_name, index_wavs, load_clip, mix  # noqa: E402
import corrupt as CR  # noqa: E402
from score_eff import logits_of  # noqa: E402
from scripts.analysis.cascade import load_model  # noqa: E402
from t23_tailq import TailQ8  # noqa: E402
from common import S3B  # noqa: E402
from t13_common import FINAL  # noqa: E402
R = a.r; DS = Path.home() / "Documents/DATASETS"; CKPT = Path.home() / "Documents/qmoe-weights/panns-wavegram/Wavegram_Cnn14_mAP=0.389.pth"
DATA = {"esc50": (DS / "ESC-50/ast_768/esc-50.csv", DS / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio", 80000, 50, "label", 5,
                  DS / "Urban8k/ast_768/urbansound8k.csv", DS / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio", "esc50"),
        "urban": (DS / "Urban8k/ast_768/urbansound8k.csv", DS / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio", 64587, 10, "class_id", 10,
                  DS / "ESC-50/ast_768/esc-50.csv", DS / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio", "urban8k"),
        "tau": (DS / "TAU2020Mobile/ast_768/tau2020mobile.csv", DS / "TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968/TAU-urban-acoustic-scenes-2020-mobile-development/audio",
                160000, 10, "class_id", 5, None, None, None)}
SEEDS = [0, 1, 2]; BS = 16
csv, aud, pad, K, ycol, NF, bg_csv, bg_aud, s3name = DATA[a.task]
FOLDS = [1] if a.dry_out is not None else list(range(1, NF + 1))

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

DRY = a.dry_out is not None; OUT = a.dry_out if DRY else R
if not DRY:
    al = json.loads((R / f"inputs/alignment_pre_{a.task}.json").read_text())
    if not al.get("all_ok"): raise SystemExit(f"alignment_pre_{a.task}.json is not all_ok: refusing to score")
if md5(CKPT) != "1e3506ab640371e0b5a417b15fd66d21": raise SystemExit("checkpoint md5 differs from Zenodo's")
meta = pd.read_csv(csv); stems = meta["name"].map(lambda s: Path(s).stem).tolist(); n = len(stems); crc = zlib.crc32("\n".join(stems).encode())
y = meta[ycol].to_numpy(np.int64); folder = meta.folder.to_numpy()
n_a = int((meta.device == "a").sum()) if a.task == "tau" else n
if a.task == "tau" and not (meta.device[:n_a] == "a").all(): raise SystemExit("TAU device-A rows are not the leading block")
torch.set_num_threads(a.threads); dev = torch.device(a.device)
base = load_wavegram(CKPT); q = QuantWavegram(base, 8, 8, dev).net

# heads: load and re-check every (seed, fold) before anything is written
H, checks = {}, []
for s in SEEDS:
    for f in FOLDS:
        run = R / f"outputs_ft/wg_{a.task}_s{s}_f{f}/esc"; split = json.loads((run / f"fold_{f}/split.json").read_text())
        te = np.asarray(split["test_idx"]); iv = np.asarray(split["inner_val_idx"]).astype(int)
        if not np.array_equal(te, np.flatnonzero(folder[:n_a] == f)): raise SystemExit(f"{run}: test rows are not fold {f}'s rows")
        model, _ = load_model(run, f, 2048, K)
        with open(run / "scaler.pkl", "rb") as fh: scaler = pickle.load(fh)
        ck = np.load(run / "check.npz"); summ = {d["fold"]: d for d in json.loads((run / "summary.json").read_text())["folds"]}
        f1 = f1_score(ck["y_te"], logits_of(model, scaler, ck["X_te"], K).argmax(1), average="macro", zero_division=0)
        if not np.array_equal(ck["te"], te) or abs(f1 - summ[f]["f1_macro"]) > 1e-6: raise SystemExit(f"{run}: head check failed ({f1:.6f} vs {summ[f]['f1_macro']:.6f})")
        H[(s, f)] = (np.flatnonzero(folder == f), iv, scaler, model); checks.append(dict(seed=s, fold=f, f1=f1))
print(f"{a.task}: {len(H)} heads re-checked against summary.json", flush=True)

def trunk(x):
    x = q.wavegram(x)
    for i in range(2, 6): x = getattr(q, f"conv_block{i}")(x)
    return x

index = index_wavs(aud); missing = [s for s in stems if s not in index]
if missing: raise SystemExit(f"{len(missing)} clips missing")
noise_maker = NoiseMaker([], {}, False)
if any(c.startswith("bg:") for c in a.conditions):
    bgn = pd.read_csv(bg_csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist(); bgi = index_wavs(bg_aud)
    noise_maker = NoiseMaker([b for b in bgn if b in bgi], bgi, True)
L_ref = np.load(R / f"emb/{a.task}/lengths.npy")
multi = len(a.conditions) > 1 and n * pad * 4 < 3e9; audio = {}
def clean_wave(i):
    if not multi: return load_clip(index[stems[i]])
    if i not in audio: audio[i] = load_clip(index[stems[i]])
    return audio[i]

def compute(kind, lv, todo):
    """W8A8 trunk output (float16) of rows `todo` under one condition; returns (array for those rows, stats)."""
    st = {"silent": 0, "level_db": [], "len_bad": 0}; chunks = [todo[i:i + 128] for i in range(0, len(todo), 128)]; out = None
    def prepare(idx):
        ws = []
        for i in idx:
            x = clean_wave(i)
            if len(x) != L_ref[i]: st["len_bad"] += 1
            if kind in CR.KINDS:
                yv = CR.corrupt(x, kind, lv); d = CR.level_change_db(x, yv)
                if d is not None: st["level_db"].append(d)
                x = yv
            elif kind != "clean":
                x, mixed = mix(x, noise_maker(stems[i], kind, len(x)), lv); st["silent"] += not mixed
            if len(x) > pad: raise SystemExit(f"{stems[i]}: longer than the pad")
            ws.append(np.pad(x, (0, pad - len(x))))
        with torch.no_grad(): return M.resample_to_32k(torch.from_numpy(np.stack(ws)).float())
    pool = ThreadPoolExecutor(1); nxt = pool.submit(prepare, chunks[0]); pos = 0
    for ci, idx in enumerate(chunks):
        w32 = nxt.result()
        if ci + 1 < len(chunks): nxt = pool.submit(prepare, chunks[ci + 1])
        for b in range(0, len(idx), BS):
            xb = w32[b:b + BS]; nb = len(xb)
            if nb < BS: xb = xb[torch.arange(BS) % nb]
            with torch.no_grad(): o = trunk(xb.to(dev)).half().cpu().numpy()[:nb]
            if out is None: out = np.zeros((len(todo),) + o.shape[1:], np.float16)
            out[pos:pos + nb] = o; pos += nb
        if dev.type == "mps": torch.mps.empty_cache()
    return out, st

def tail_logits(T, s, f, rows_):
    tail = TailQ8(base, R / f"ft/{a.task}/f{f}_s{s}/tail_q8.pt", dev); E = []
    for b in range(0, len(rows_), 128):
        E.append(tail(torch.from_numpy(np.asarray(T[rows_[b:b + 128]], np.float32)).to(dev)).float().cpu().numpy())
    del tail
    if dev.type == "mps": torch.mps.empty_cache()
    return np.concatenate(E)

def batch_embed(fn, T, rows_):
    out = []
    with torch.no_grad():
        for b in range(0, len(rows_), 128): out.append(fn(torch.from_numpy(np.asarray(T[rows_[b:b + 128]], np.float32)).to(dev)).float().cpu().numpy())
    return np.concatenate(out)

ref_plan = json.loads((S3B / f"emb_cnn/{s3name}/bg_plan.json").read_text()) if s3name else {}
E_dir = OUT / f"emb/{a.task}"; E_dir.mkdir(parents=True, exist_ok=True); last = 300.0
NEED = np.unique(np.concatenate([np.concatenate([H[k][0], H[k][1]]) for k in H]))   # all rows unless --dry-out
for c in a.conditions:
    kind, lv = ("clean", None) if c == "clean" else (c.split(":")[0], float(c.split(":")[1]))
    cname = "clean" if kind == "clean" else (CR.cond_label(kind, lv) if kind in CR.KINDS else cond_name(kind, lv))
    fname = "wg__all.npz" if a.task == "tau" else f"wg__{cname}.npz"
    dests = [OUT / f"scores/{a.task}/wg_{a.task}_s{s}/esc/{fname}" for s in SEEDS]
    if all(d.exists() for d in dests): continue
    if (time.time() - T_START) / 60 + 1.5 * last / 60 > a.max_minutes: print("time limit before next condition", flush=True); sys.exit(3)
    t0 = time.time()
    if kind == "clean":
        C = np.load(R / f"ft/{a.task}/trunk_w8a8.npy")
        if len(C) != n_a: raise SystemExit("trunk cache rows differ")
        if a.task == "tau":
            rest, st = compute("clean", None, np.arange(n_a, n)); T = np.concatenate([C, rest]); del rest
        else: T, st = C, {"silent": 0, "level_db": [], "len_bad": 0}
        del C
    else:
        Tn, st = compute(kind, lv, NEED); T = np.zeros((n,) + Tn.shape[1:], np.float16); T[NEED] = Tn; del Tn
    side = {"condition": cname, "kind": kind, "level": lv, "task": a.task, "n_clips": n, "names_crc32": crc, "embeddings_written": False,
            "silent_clips_left_clean": st["silent"], "mean_level_change_db": float(np.mean(st["level_db"])) if st["level_db"] else None,
            "n_level": len(st["level_db"]), "lengths_differing": st["len_bad"], "trunk_batch": BS, "device": str(dev), "torch": torch.__version__}
    if st["len_bad"]: raise SystemExit(f"{cname}: {st['len_bad']} clip lengths differ from lengths.npy")
    if kind == "bg":
        bad = [stems[i] for i in NEED if noise_maker.plan.get(stems[i]) != ref_plan.get(stems[i])]
        if bad: raise SystemExit(f"{cname}: {len(bad)} noise draws differ from stage 3b's bg_plan.json, e.g. {bad[:3]}")
        pf = E_dir / "bg_plan.json"; old = json.loads(pf.read_text()) if pf.exists() else {}; old.update(noise_maker.plan); pf.write_text(json.dumps(old, indent=0))
    if kind in CR.KINDS:
        ref = FINAL / f"emb_cnn/{'esc50' if a.task == 'esc50' else 'urban8k_unpadded'}/mn10_w8a8__{cname}.json"
        if not ref.exists(): ref = FINAL / f"emb_cnn/{'esc50' if a.task == 'esc50' else 'urban8k'}/mn10_w8a8__{cname}.json"
        rv = json.loads(ref.read_text()).get("mean_level_change_db")
        side["final_level_change_db"] = rv
        if not DRY and (rv is None or abs(rv - side["mean_level_change_db"]) > 1e-4): raise SystemExit(f"{cname}: level change {side['mean_level_change_db']} vs final run {rv}")
        side["final_ref"] = str(ref)
    if kind == "clean":   # amendment 1 item 3: float16 cache path against the end-to-end pass, and the compact fold-1 tails against the P1 files
        cc = {}
        froz = batch_embed(lambda x: torch.relu(q.fc1((lambda h: h.max(2)[0] + h.mean(2))(q.conv_block6(x, (1, 1)).mean(3)))), T, np.arange(n_a))
        Eref = np.load(R / f"emb/{a.task}/wg__clean.npy")[:n_a]; dlt = np.abs(froz - Eref)
        cc["frozen_tail_on_cache_vs_end_to_end"] = dict(max_abs=float(dlt.max()), mean_abs=float(dlt.mean()), ref_mean_abs=float(np.abs(Eref).mean()),
                                                        rel_l2=float(np.linalg.norm(froz - Eref) / np.linalg.norm(Eref)), rows=n_a)
        for s in SEEDS:
            p1 = R / f"ft/{a.task}/f1_s{s}/wg__clean.npy"
            if p1.exists():
                e1 = tail_logits(T, s, 1, np.arange(n_a)); r1 = np.load(p1); cc[f"tailq8_f1_s{s}_vs_p1_max_abs"] = float(np.abs(e1 - r1).max())
                if cc[f"tailq8_f1_s{s}_vs_p1_max_abs"] > 1e-4 * float(np.abs(r1).max()): raise SystemExit(f"compact tail f1 s{s} differs from the P1 file: {cc}")
        (OUT / "inputs").mkdir(exist_ok=True); (OUT / f"inputs/cache_check_{a.task}.json").write_text(json.dumps(cc, indent=1)); print(cc, flush=True)
    for s, dest in zip(SEEDS, dests):
        res = {"y": y, "names_crc32": np.int64(crc), "test_logits": np.full((n, K), np.nan, np.float32)}
        if a.task == "tau": res.update(fold=folder.astype(np.int16), device=meta.device.to_numpy().astype(str), n_a=np.int64(n_a))
        else: res["test_fold"] = np.full(n, -1, np.int16)
        for f in FOLDS:
            te, iv, scaler, model = H[(s, f)]; rr = np.concatenate([te, iv]); E = tail_logits(T, s, f, rr)
            if not np.isfinite(E).all(): raise SystemExit(f"{cname} s{s} f{f}: non-finite embeddings")
            Z = logits_of(model, scaler, E, K); res["test_logits"][te] = Z[:len(te)]; res[f"iv_idx_{f}"], res[f"iv_logits_{f}"] = iv, Z[len(te):]
            if a.task != "tau": res["test_fold"][te] = f
        if not DRY and not np.isfinite(res["test_logits"]).all(): raise SystemExit(f"{cname} s{s}: rows left unscored")
        dest.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(dest, **res)
    side["seconds"] = time.time() - t0; (E_dir / f"wg__{cname}.json").write_text(json.dumps(side, indent=1)); del T
    last = time.time() - t0; print(f"{a.task} {cname}: scored {len(SEEDS)} seeds x {len(FOLDS)} folds in {last:.0f}s", flush=True)
print(f"done in {time.time() - T_START:.0f}s", flush=True)
