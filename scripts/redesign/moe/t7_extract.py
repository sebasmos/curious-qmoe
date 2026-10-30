"""T7 masked-view AST embeddings (ANALYSIS_PLAN_T7.md section 3). AST w4g128 (QuantAST) at a fixed input length, keeping k patches chosen by a cheap rule.
Output per (mask, condition): <out>/w4g128__<sel><rho>__<cond>.npy float32 [N, 768] (rows not extracted are NaN), keep__<sel><rho>__<cond>.npy uint16 [N, k], a JSON sidecar.
--check: the C1 check of the plan (all patches kept vs the stored full-view embedding, selector sanity), writes check_c1.json and stops.
    python t7_extract.py --csv CSV --audio-dir AUDIO --bg-audio-dir BG --bg-csv BGCSV --out-dir OUT --max-length 400 --conditions clean --field-test [--check STORED_NPY]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2"))
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--bg-audio-dir", type=Path, default=None)
    p.add_argument("--bg-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--masks", nargs="+", default=["energy25", "random25", "crop25", "energy50", "random50", "crop50"])
    p.add_argument("--conditions", nargs="*", default=["clean"], help="clean or KIND:SNR (bg, pink, white), full rows")
    p.add_argument("--field-test", action="store_true", help="add the nine field cells for the rows the TEST assignment (seed 5001) gives them")
    p.add_argument("--check", type=Path, default=None, help="stored full-view embedding .npy: run C1 on 64 rows and stop")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--revision", default="f826b80d28226b62986cc218e5cec390b1096902")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--save-every", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cache-bg", action="store_true")
    return p.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(ARGS.threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from QWave.ast_frames import shorten_ast  # noqa: E402
from QWave.backbone_quant import QuantAST, parse_config  # noqa: E402
from audio_shift import (FIELD_CELLS, FIELD_SEEDS, NoiseMaker, cond_name, field_assignment,  # noqa: E402
                         index_wavs, load_clip, mix)
import t7_mask as M  # noqa: E402

TARGET_SR = 16000


def parse_condition(c):
    if c == "clean":
        return "clean", None
    kind, snr = c.split(":")
    if kind not in ("bg", "pink", "white"):
        raise SystemExit(f"bad condition {c}")
    return kind, float(snr)


def plan_conditions(conditions, field_test, n):
    plan = [(k, s, np.arange(n)) for k, s in map(parse_condition, conditions)]
    if field_test:
        full = {cond_name(k, s) for k, s, _ in plan}
        assign = np.array(field_assignment(n, FIELD_SEEDS["test"]))
        for kind, snr in FIELD_CELLS:
            name = cond_name(kind, snr)
            if name not in full:
                plan.append((kind, float(snr), np.flatnonzero(assign == name)))
    return plan


def mask_spec(m):
    return m.rstrip("0123456789"), int(m[len(m.rstrip("0123456789")):])


def main():
    a = ARGS
    t_start = time.time()
    meta = pd.read_csv(a.csv, usecols=["folder", "name", "class_id"])
    if a.limit:
        meta = meta.iloc[:a.limit]
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    n = len(stems)
    index = index_wavs(a.audio_dir)
    missing = [s for s in stems if s not in index]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing, e.g. {missing[:5]}")
    plan = plan_conditions(a.conditions, a.field_test, n)
    noise_maker = NoiseMaker([], {}, False)
    if any(k == "bg" for k, _, _ in plan):
        bg_names = pd.read_csv(a.bg_csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(a.bg_audio_dir)
        noise_maker = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, a.cache_bg)

    from transformers import ASTFeatureExtractor, ASTModel

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    torch.set_num_threads(a.threads)
    L = a.max_length
    extractor = ASTFeatureExtractor.from_pretrained(a.model, revision=a.revision, max_length=L)
    model = shorten_ast(ASTModel.from_pretrained(a.model, revision=a.revision), L).to(device).eval()
    if extractor.max_length != L:
        raise SystemExit("feature extractor ignored max_length")
    qast = QuantAST(model)
    qast.set_config(*parse_config("w4g128"))
    masks = [mask_spec(m) for m in a.masks]

    def embed(feats_cpu, n_samp, st, sel, rho):
        """keep indices (CPU np) and embeddings for one mask on one chunk."""
        keep = M.select(sel, rho, feats_cpu, n_samp, st, L)
        kt = torch.as_tensor(keep, dtype=torch.long)
        outs = []
        for b in range(0, len(st), a.batch_size):
            outs.append(M.masked_forward(qast, feats_cpu[b:b + a.batch_size].to(device), kt[b:b + a.batch_size].to(device)).float().cpu().numpy())
        return keep, np.concatenate(outs)

    if a.check is not None:
        rows = np.arange(0, n, max(1, n // 64))[:64]
        waves = [load_clip(index[stems[i]]) for i in rows]
        n_samp = np.array([len(w) for w in waves])
        feats = extractor(waves, sampling_rate=TARGET_SR, return_tensors="pt")["input_values"].contiguous()
        st = [stems[i] for i in rows]
        stored = np.load(a.check)[rows]
        full = np.concatenate([M.masked_forward(qast, feats[b:b + a.batch_size].to(device), None).float().cpu().numpy() for b in range(0, len(rows), a.batch_size)])
        ref = np.concatenate([qast(feats[b:b + a.batch_size].to(device)).float().cpu().numpy() for b in range(0, len(rows), a.batch_size)])
        rel = lambda x, y: np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1)
        res = {"rows": rows.tolist(), "full_vs_stored_rel_l2_median": float(np.median(rel(full, stored))), "full_vs_stored_rel_l2_max": float(rel(full, stored).max()),
               "full_vs_qast_forward_rel_l2_max": float(rel(full, ref).max()), "ref_vs_stored_rel_l2_max": float(rel(ref, stored).max())}
        ks, sets = {}, {}
        for sel, rho in masks:
            keep, _ = embed(feats, n_samp, st, sel, rho)
            k = M.k_keep(L, rho)
            n_v = np.array([M.valid_frames(int(x), L) for x in n_samp])
            valid = np.stack([M.patch_valid(int(v), L) for v in n_v])
            uniq = all(len(set(r)) == k and r.max() < M.n_patches(L) for r in keep)
            nvalid_in = [int(valid[i, keep[i]].sum()) for i in range(len(rows))]
            ok_first = all(nvalid_in[i] == min(k, int(valid[i].sum())) for i in range(len(rows)))
            ks[f"{sel}{rho}"] = {"k": k, "unique_and_in_range": bool(uniq), "valid_first": bool(ok_first), "mean_valid_kept": float(np.mean(nvalid_in))}
            sets[(sel, rho)] = keep
        for rho in sorted({r for _, r in masks}):
            if ("energy", rho) in sets and ("random", rho) in sets:
                kk = M.k_keep(L, rho)
                big = [i for i in range(len(rows)) if int(valid[i].sum()) > kk]      # amendment 1: only clips with more valid patches than k can differ
                diff = np.mean([not np.array_equal(sets[("energy", rho)][i], sets[("random", rho)][i]) for i in big])
                ks[f"energy{rho}_differs_from_random{rho}_on_clips_with_more_than_k_valid_patches"] = float(diff)
                ks[f"n_clips_with_at_most_k_valid_patches_{rho}"] = int(len(rows) - len(big))
        res["selectors"] = ks
        res["pass"] = bool(res["full_vs_stored_rel_l2_median"] <= 0.02 and res["full_vs_stored_rel_l2_max"] <= 0.10
                           and all(v["unique_and_in_range"] and v["valid_first"] for k_, v in ks.items() if isinstance(v, dict))
                           and all(v > 0.9 for k_, v in ks.items() if "differs" in k_))
        (out / "check_c1.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return 0 if res["pass"] else 2

    for kind, snr, rows in plan:
        cname = cond_name(kind, snr)
        todo = [m for m in masks if not (out / f"w4g128__{m[0]}{m[1]}__{cname}.npy").exists()]
        if not todo:
            continue
        part = out / f"_partial__{cname}.npz"
        names = [f"{s}{r}" for s, r in todo]
        X = {m: np.full((n, 768), np.nan, np.float32) for m in names}
        K = {m: np.zeros((n, M.k_keep(L, r)), np.uint16) for m, (s, r) in zip(names, todo)}
        done = np.zeros(n, bool)
        if part.exists():
            d = np.load(part)
            if sorted(d["masks"].tolist()) == sorted(names):
                done = d["done"]
                for m in names:
                    X[m], K[m] = d[f"X_{m}"], d[f"K_{m}"]
                print(f"{cname}: resuming, {int(done.sum())} rows done", flush=True)
        prep_s, fwd_s, n_silent = 0.0, {m: 0.0 for m in names}, 0
        t0 = time.time()
        left = rows[~done[rows]]
        for ci, start in enumerate(range(0, len(left), a.chunk)):
            idx = left[start:start + a.chunk]
            tp = time.time()
            waves = []
            for i in idx:
                x = load_clip(index[stems[i]])
                if kind != "clean":
                    x, mixed = mix(x, noise_maker(stems[i], kind, len(x)), snr)
                    n_silent += not mixed
                waves.append(x)
            n_samp = np.array([len(w) for w in waves])
            feats = extractor(waves, sampling_rate=TARGET_SR, return_tensors="pt")["input_values"].contiguous()
            st = [stems[i] for i in idx]
            prep_s += time.time() - tp
            for (sel, rho), m in zip(todo, names):
                tf = time.time()
                keep, emb = embed(feats, n_samp, st, sel, rho)
                X[m][idx], K[m][idx] = emb, keep.astype(np.uint16)
                fwd_s[m] += time.time() - tf
            done[idx] = True
            if (ci + 1) % a.save_every == 0:
                np.savez(part, masks=np.array(names), done=done, **{f"X_{m}": X[m] for m in names}, **{f"K_{m}": K[m] for m in names})
            el = time.time() - t0
            print(f"  {cname}: {int(done[rows].sum())}/{len(rows)} rows, {el:.0f}s", flush=True)
        dt = time.time() - t0
        for (sel, rho), m in zip(todo, names):
            if not np.isfinite(X[m][rows]).all():
                raise SystemExit(f"{m}__{cname}: non-finite embeddings")
            np.save(out / f"w4g128__{m}__{cname}.npy", X[m])
            np.save(out / f"keep__{m}__{cname}.npy", K[m])
            side = {"config": "w4g128", "selector": sel, "rho_keep_pct": rho, "k": M.k_keep(L, rho), "tokens": M.k_keep(L, rho) + 2, "max_length": L,
                    "condition": cname, "noise": kind, "snr_db": snr, "n_clips": n, "rows_extracted": int(len(rows)), "silent_clips_left_clean": n_silent,
                    "forward_seconds": fwd_s[m], "condition_seconds": dt, "prep_seconds": prep_s, "device": str(device), "batch_size": a.batch_size,
                    "chunk": a.chunk, "csv": str(a.csv), "model": a.model, "revision": a.revision,
                    "priced_cost_per_clip": M.priced_cost(sel, rho, L), "priced_cost_no_front": M.priced_cost(sel, rho, L, front=False),
                    "names_crc32": zlib.crc32("\n".join(stems).encode())}
            (out / f"w4g128__{m}__{cname}.json").write_text(json.dumps(side, indent=2))
        if kind == "bg":
            planf = out / "bg_plan.json"
            old = json.loads(planf.read_text()) if planf.exists() else {}
            old.update(noise_maker.plan)
            planf.write_text(json.dumps(old, indent=0))
        if part.exists():
            part.unlink()
        print(f"{cname}: {len(left)} rows x {len(todo)} masks in {dt:.0f}s (prep {prep_s:.0f}s), silent {n_silent}", flush=True)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
