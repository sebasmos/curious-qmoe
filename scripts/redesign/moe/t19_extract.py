"""T19 extraction (steps 19.5, 19.6): AST pooled embeddings of T19 configurations on one task, on the GPU.

Audio, noise and field rows follow idea 6 stage 2's extract_ast_short.py (the same audio_shift mixer, the same background lists, so the
background plan is reproduced; the final round's corrupt.py for reverb, clip and lowpass). For each condition the clips are read and turned
into the task's fbank once per chunk, then every requested configuration embeds the chunk (QuantAST.set_t19 re-quantizes from the smoothed
master). Output <out>/<config>__<condition>.npy, float32 [N, 768] in the CSV's row order, NaN where a row was not extracted; a row is
extracted once and later runs fill only missing rows (P1 runs without fold 1's test rows, the full run fills them). A JSON sidecar records
the rows extracted, timing and costs. --max-minutes stops cleanly at a chunk boundary (the GPU is shared), the arrays are saved every
--save-every chunks and at exit.

    python t19_extract.py --task esc50 --configs w8a8 n4a4 --conditions clean [--field] [--exclude-test-fold 1] [--max-minutes 80]
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t19_common import CONFIGS, MODEL, REVISION, TASKS, backbone_bitops, layer_bits, results_dir  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "idea6/final"))
import corrupt as CR  # noqa: E402
from QWave.ast_frames import shorten_ast  # noqa: E402
from QWave.backbone_quant import QuantAST, T19Act, parse_config  # noqa: E402
from audio_shift import FIELD_CELLS, FIELD_SEEDS, NoiseMaker, cond_name, field_assignment, index_wavs, load_clip, mix  # noqa: E402

BG_OF = {"esc50": "urban", "urban": "esc50"}


def parse_condition(c):
    if c == "clean":
        return "clean", None
    kind, lv = c.split(":")
    if kind not in ("bg", "pink", "white", *CR.KINDS):
        raise SystemExit(f"bad condition {c}")
    return kind, float(lv)


def cname_of(kind, lv):
    return CR.cond_label(kind, lv) if kind in CR.KINDS else cond_name(kind, lv)


def plan_conditions(conditions, field, n):
    plan = [(k, s, np.arange(n)) for k, s in map(parse_condition, conditions)]
    if field:
        full = {cname_of(k, s) for k, s, _ in plan}
        assign = [np.array(field_assignment(n, seed)) for seed in FIELD_SEEDS.values()]
        if "clean" not in full:
            plan.insert(0, ("clean", None, np.arange(n)))
        for kind, snr in FIELD_CELLS:
            name = cond_name(kind, snr)
            if name not in full:
                rows = np.flatnonzero((assign[0] == name) | (assign[1] == name))
                plan.append((kind, float(snr), rows))
    return plan


def act_of(cfg, cal, calinfo):
    wname, a, sr = CONFIGS[cfg]
    bits = layer_bits(cfg, calinfo)
    clip = {b: (None if cal["choice"][b] is None else cal["clip"][cal["choice"][b]]) for b in set(bits)}
    return parse_config(wname)[0], T19Act(layer_bits=bits, p_bits=8, smooth=cal["smooth"], clip=clip, asym=True, sr_seed=sr)


def save_atomic(path, arr):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["esc50", "urban", "tau"])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="*", default=["clean"])
    ap.add_argument("--field", action="store_true")
    ap.add_argument("--exclude-test-fold", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--save-every", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--calib-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default <results>/emb/<task>")
    a = ap.parse_args()
    t_start = time.time()
    T = TASKS[a.task]
    R = results_dir()
    out = a.out or R / "emb" / T["emb"]
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(T["csv"], usecols=lambda c: c in ("folder", "name", "class_id", "device"))
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    n = len(stems) if a.limit is None else a.limit
    crc = zlib.crc32("\n".join(stems).encode())
    index = index_wavs(T["audio"])
    missing = [s for s in stems[:n] if s not in index]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing, e.g. {missing[:3]}")
    keep = np.ones(n, bool)
    if a.exclude_test_fold is not None:
        keep &= meta["folder"].to_numpy()[:n] != a.exclude_test_fold
    plan = plan_conditions(a.conditions, a.field, n)
    noise_maker = NoiseMaker([], {}, False)
    if any(k == "bg" for k, _, _ in plan):
        B = TASKS[BG_OF[a.task]]
        bg_names = pd.read_csv(B["csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(B["audio"])
        noise_maker = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, a.task == "urban")  # cache the 2,000 ESC-50 backgrounds only
    cd = a.calib_dir or R / "calib"
    cal = torch.load(cd / f"{a.task}.pt", weights_only=False)
    calinfo = json.loads((cd / f"{a.task}.json").read_text())
    from transformers import ASTFeatureExtractor, ASTModel

    dev = torch.device(a.device)
    L = T["length"]
    ext = ASTFeatureExtractor.from_pretrained(MODEL, revision=REVISION, max_length=L)
    model = shorten_ast(ASTModel.from_pretrained(MODEL, revision=REVISION), L).to(dev).eval()
    qa = QuantAST(model, p_bits=8)
    acts = {c: act_of(c, cal, calinfo) for c in a.configs}
    current, stopped = None, False
    for kind, lv, rows in plan:
        cname = cname_of(kind, lv)
        rows = rows[keep[rows]]
        arrs, todo = {}, {}
        for c in a.configs:
            f = out / f"{c}__{cname}.npy"
            X = np.load(f) if f.exists() else np.full((n, 768), np.nan, np.float32)
            need = rows[~np.isfinite(X[rows]).all(1)]
            if len(need):
                arrs[c], todo[c] = X, need
        if not todo:
            continue
        union = np.unique(np.concatenate(list(todo.values())))
        print(f"{a.task} {cname}: {len(union)} rows for {sorted(todo)}", flush=True)
        t0, prep_s, fwd_s, n_done, level_db, n_silent = time.time(), 0.0, {c: 0.0 for c in todo}, 0, [], 0
        for ci, start in enumerate(range(0, len(union), a.chunk)):
            idx = union[start:start + a.chunk]
            tp = time.time()
            waves = []
            for i in idx:
                x = load_clip(index[stems[i]])
                if kind in CR.KINDS:
                    x0, x = x, CR.corrupt(x, kind, lv)
                    d = CR.level_change_db(x0, x)
                    if d is not None:
                        level_db.append(d)
                elif kind != "clean":
                    x, mixed = mix(x, noise_maker(stems[i], kind, len(x)), lv)
                    n_silent += not mixed
                waves.append(x)
            feats = ext(waves, sampling_rate=16000, return_tensors="pt")["input_values"].contiguous().to(dev)
            prep_s += time.time() - tp
            for c in todo:
                sel = np.isin(idx, todo[c])
                if not sel.any():
                    continue
                if current != c:
                    qa.set_t19(*acts[c])
                    current = c
                tf = time.time()
                # fixed batch shape: full batches of the chunk, the SR seed sequence depends on the chunk layout (recorded)
                outs = [qa(feats[b:b + a.batch_size]).float().cpu().numpy() for b in range(0, len(idx), a.batch_size)]
                E = np.concatenate(outs)
                arrs[c][idx[sel]] = E[sel]
                fwd_s[c] += time.time() - tf
            n_done += len(idx)
            el = time.time() - t0
            print(f"  {cname}: {n_done}/{len(union)} rows, {el:.0f}s, {n_done * len(todo) / max(el, 1e-9):.2f} clip-configs/s", flush=True)
            last = start + a.chunk >= len(union)
            over = (time.time() - t_start) / 60 > a.max_minutes
            if (ci + 1) % a.save_every == 0 or last or over:
                for c in todo:
                    save_atomic(out / f"{c}__{cname}.npy", arrs[c])
            if over and not last:
                stopped = True
                break
        for c in todo:
            X = arrs[c]
            if not np.isfinite(X[np.isfinite(X).any(1)]).all():
                raise SystemExit(f"{c}__{cname}: non-finite values in extracted rows")
            sf = out / f"{c}__{cname}.json"
            side = json.loads(sf.read_text()) if sf.exists() else {"runs": []}
            side.update({"config": c, "task": a.task, "condition": cname, "kind": kind, "level": lv, "max_length": L,
                         "n_clips": n, "rows_extracted_total": int(np.isfinite(X).all(1).sum()), "names_crc32": crc,
                         "csv": str(T["csv"]), "model": MODEL, "revision": REVISION, "device": str(dev),
                         "batch_size": a.batch_size, "chunk": a.chunk,
                         "layer_bits": list(acts[c][1].layer_bits), "weight": acts[c][0].name, "p_bits": 8,
                         "clip_choice": {str(k): v for k, v in cal["choice"].items()}, "sr_seed": acts[c][1].sr_seed,
                         "backbone_bitops_priced": backbone_bitops(a.task, c, True) if c in CONFIGS else None,
                         "backbone_bitops_unpriced": backbone_bitops(a.task, c, False) if c in CONFIGS else None})
            side["runs"].append({"rows_requested": int(len(todo[c])), "forward_seconds": fwd_s[c], "prep_seconds": prep_s,
                                 "condition_seconds": time.time() - t0, "configs_in_pass": sorted(todo),
                                 "exclude_test_fold": a.exclude_test_fold, "silent_left_clean": n_silent,
                                 "mean_level_change_db": float(np.mean(level_db)) if level_db else None,
                                 "stopped_early": stopped, "when": time.strftime("%F %T")})
            sf.write_text(json.dumps(side, indent=1))
        if kind == "bg":
            pf = out / "bg_plan.json"
            old = json.loads(pf.read_text()) if pf.exists() else {}
            old.update(noise_maker.plan)
            pf.write_text(json.dumps(old, indent=0))
        if stopped:
            print(f"time limit {a.max_minutes} min reached, stopping (resume later)", flush=True)
            sys.exit(3)
    print(f"all done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
