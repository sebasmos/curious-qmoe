"""T21 AST extraction at each clip's own length (ANALYSIS_PLAN.md C3), several configurations, several conditions.

Based on the idea 6 final round's extract_ast_short.py (same condition parsing, field cells, chunking, partial saves
and sidecars). Changes: clips are truncated to 10.24 s (t21_common), and each clip is read unpadded at its own number
of time patches t. Clips run in length buckets of BUCKETS frames; inside a bucket every patch beyond the clip's t is
removed from attention as a key (score -1e9 before the softmax) and its value row is zeroed before the per-channel
value quantization, so the [CLS] and [DIST] outputs equal an unpadded run up to float rounding (checked by `--check`).
Batches inside a bucket have a fixed size (a short last batch is filled with repeated rows whose outputs are dropped),
so MPS compiles one graph per bucket. Sidecars carry the per-clip time patches and per-clip bit-ops of the config.

    python t21_extract_ast.py --csv META --audio-dir AUDIO --bg-csv U8K_CSV --bg-audio-dir U8K_AUDIO --out-dir OUT \
        --configs fp32 w4g128 --conditions clean bg:10 ... [--field]
    python t21_extract_ast.py --check --csv META --audio-dir AUDIO --out-dir OUT --configs fp32 w4g128 (32 clips)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--bg-audio-dir", type=Path, default=None)
    p.add_argument("--bg-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--conditions", nargs="*", default=[])
    p.add_argument("--field", action="store_true")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--save-every", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-seconds", type=float, default=0, help="stop cleanly after this many seconds (GPU share)")
    p.add_argument("--available-only", action="store_true",
                   help="process only clips whose audio is on disk; a condition stays partial until every row is done")
    p.add_argument("--check", action="store_true", help="masked buckets against single-clip unpadded runs")
    p.add_argument("--check-n", type=int, default=32)
    return p.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(ARGS.threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t21_common as C  # noqa: E402

C.patch_loader()
from audio_shift import FIELD_CELLS, FIELD_SEEDS, NoiseMaker, cond_name, field_assignment, index_wavs, mix  # noqa: E402
import audio_shift as AS  # noqa: E402
import corrupt as CR  # noqa: E402
from QWave.ast_frames import shorten_ast  # noqa: E402
from QWave.backbone_quant import QuantAST, fake_quant_qk_, fake_quant_sym, parse_config, quant_softmax_av  # noqa: E402
import analyze as A2  # noqa: E402  stage 2 analyze: ast_bitops_tok, ast_fbank_frontend_macs

BUCKETS = (128, 256, 384, 512, 640, 768, 896, 1024)
NEG = -1e9


def bucket_of(t):
    for L in BUCKETS:
        if C.time_patches(L) >= t:
            return L
    raise ValueError(t)


class MaskedAST(QuantAST):
    """QuantAST whose forward reads a bucket of L frames and masks every patch at time index >= t of its clip."""

    def __init__(self, ast_model, p_bits=16):
        super().__init__(ast_model, p_bits)
        cfg = ast_model.config
        self.nf = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
        self.nt_full = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
        pos = ast_model.embeddings.position_embeddings.detach()
        self.pos_special = pos[:, :2]
        self.pos_grid = pos[:, 2:].reshape(1, self.nf, self.nt_full, pos.shape[-1])

    def embed(self, x):
        e = self.ast.embeddings
        B, L = x.shape[0], x.shape[1]
        nt = C.time_patches(L)
        p = e.patch_embeddings(x)  # [B, nf * nt, D], frequency first
        pos = torch.cat([self.pos_special, self.pos_grid[:, :, :nt].reshape(1, self.nf * nt, -1)], 1)
        h = torch.cat([e.cls_token.expand(B, -1, -1), e.distillation_token.expand(B, -1, -1), p], 1) + pos
        return h, nt

    def keep_mask(self, t, nt):
        j = torch.arange(nt, device=t.device)
        keep = (j[None, None, :] < t[:, None, None]).expand(-1, self.nf, -1).reshape(len(t), -1)
        return torch.cat([torch.ones(len(t), 2, dtype=torch.bool, device=t.device), keep], 1)

    def _mlayer(self, h, layer, w, b, keep):
        B, T, D = h.shape
        H, dh, a = self.heads, self.head_dim, self.act_bits
        qkv = F.linear(self._aq_(layer.layernorm_before(h)), w["qkv"], b["qkv"])
        if a is not None:
            fake_quant_qk_(qkv, a, H, dh)
        q, k, v = qkv.view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4).unbind(0)
        v = v * keep[:, None, :, None].to(v.dtype)
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores.masked_fill_(~keep[:, None, None, :], NEG)
        if a is None:
            ctx = torch.matmul(torch.softmax(scores.mul_(self.scaling), dim=-1), v)
        else:
            ctx = quant_softmax_av(scores.contiguous(), fake_quant_sym(v, a, dim=-2), self.p_bits, self.scaling)
        ctx = ctx.transpose(1, 2).reshape(B, T, D).contiguous()
        h = h + F.linear(self._aq_(ctx), w["attn_out"], b["attn_out"])
        x = F.gelu(F.linear(self._aq_(layer.layernorm_after(h)), w["fc1"], b["fc1"]))
        return h + F.linear(self._aq_(x), w["fc2"], b["fc2"])

    @torch.no_grad()
    def forward_masked(self, x, t):
        h, nt = self.embed(x)
        keep = self.keep_mask(t, nt)
        for layer, w, b in zip(self.ast.encoder.layer, self.weights, self.bias):
            h = self._mlayer(h, layer, w, b, keep)
        h = self.ast.layernorm(h)
        return (h[:, 0] + h[:, 1]) / 2


def clip_bitops(cfg_name, t, n16):
    P = 12 * t
    return A2.ast_bitops_tok(cfg_name, 1024.0, P + 2, P), A2.ast_fbank_frontend_macs(int(n16)) * 1024.0


def parse_condition(c):
    if c == "clean":
        return "clean", None
    kind, snr = c.split(":")
    if kind not in ("bg", "pink", "white", *CR.KINDS):
        raise SystemExit(f"bad condition {c}")
    return kind, float(snr)


def cname_of(kind, level):
    return CR.cond_label(kind, level) if kind in CR.KINDS else cond_name(kind, level)


def plan_conditions(conditions, field, n):
    plan = [(k, s, np.arange(n)) for k, s in map(parse_condition, conditions)]
    if field:
        full = {cname_of(k, s) for k, s, _ in plan}
        assign = [np.array(field_assignment(n, seed)) for seed in FIELD_SEEDS.values()]
        for kind, snr in FIELD_CELLS:
            name = cond_name(kind, snr)
            if name not in full:
                rows = np.flatnonzero((assign[0] == name) | (assign[1] == name))
                plan.append((kind, float(snr), rows))
    return plan


def load_model(device):
    from transformers import ASTFeatureExtractor, ASTModel

    extractor = ASTFeatureExtractor.from_pretrained(C.AST_MODEL, revision=C.AST_REV, max_length=1024)
    model = ASTModel.from_pretrained(C.AST_MODEL, revision=C.AST_REV).to(device).eval()
    return extractor, model


def check(args):
    """Masked bucket runs against single-clip runs of shorten_ast at the clip's own fbank length."""
    import copy

    device = torch.device(args.device)
    meta = pd.read_csv(args.csv, usecols=["name"])
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    index = index_wavs(args.audio_dir)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(stems), args.check_n, replace=False))
    extractor, model = load_model(device)
    waves = [AS.load_clip(index[stems[i]]) for i in pick]
    feats = extractor(waves, sampling_rate=16000, return_tensors="pt")["input_values"]
    T = np.array([C.fbank_frames(len(w)) for w in waves])
    t = np.array([C.time_patches(x) for x in T])
    m = MaskedAST(model)
    out = {}
    for c in args.configs:
        m.set_config(*parse_config(c))
        got = np.zeros((len(pick), 768), np.float32)
        for L in sorted({bucket_of(x) for x in t}):
            sel = np.flatnonzero([bucket_of(x) == L for x in t])
            got[sel] = m.forward_masked(feats[sel, :L].contiguous().to(device),
                                        torch.as_tensor(t[sel], device=device)).float().cpu().numpy()
        ref = np.zeros_like(got)
        ref_noise = np.zeros_like(got)
        gen = torch.Generator().manual_seed(1)
        for k, i in enumerate(pick):
            Lk = 16 + 10 * (t[k] - 1)
            mk = shorten_ast(copy.deepcopy(model).cpu(), Lk).to(device)
            qk = QuantAST(mk)
            qk.set_config(*parse_config(c))
            ref[k] = qk(feats[k:k + 1, :Lk].contiguous().to(device)).float().cpu().numpy()[0]
            xn = feats[k:k + 1, :Lk] + 1e-6 * torch.randn(1, Lk, feats.shape[2], generator=gen)
            ref_noise[k] = qk(xn.contiguous().to(device)).float().cpu().numpy()[0]
            del qk, mk
        rel = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)
        floor = np.linalg.norm(ref_noise - ref, axis=1) / np.linalg.norm(ref, axis=1)
        out[c] = {"rel_mean": float(rel.mean()), "rel_max": float(rel.max()), "n": int(len(pick)),
                  "floor_rel_mean_input_noise_1e-6": float(floor.mean()), "floor_rel_max": float(floor.max()),
                  "pass": bool(rel.mean() < 1e-3 or rel.mean() <= 1.5 * floor.mean()),
                  "frames_min": int(T.min()), "frames_max": int(T.max()), "buckets": sorted({int(bucket_of(x)) for x in t})}
        print(c, out[c], flush=True)
    import torchaudio.compliance.kaldi as K
    kal = [K.fbank(torch.as_tensor(w)[None], sample_frequency=16000, htk_compat=True, window_type="hanning",
                   num_mel_bins=128, dither=0.0, frame_shift=10).shape[0] for w in waves]
    out["fbank_frames_rule_ok"] = bool(np.array_equal(np.array(kal), T))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"mask_check_{args.device}.json").write_text(json.dumps(out, indent=1))


def main():
    args = ARGS
    if args.check:
        return check(args)
    t_start = time.time()
    meta = pd.read_csv(args.csv, usecols=["folder", "name", "class_id"])
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    n = len(stems)
    crc = zlib.crc32("\n".join(stems).encode())
    index = index_wavs(args.audio_dir)
    missing = [s for s in stems if s not in index]
    avail = np.array([s in index for s in stems])
    if missing and not args.available_only:
        raise SystemExit(f"{len(missing)} clips missing, e.g. {missing[:5]}")
    plan = plan_conditions(args.conditions, args.field, n)
    noise_maker = NoiseMaker([], {}, False)
    if any(k == "bg" for k, _, _ in plan):
        bg_names = pd.read_csv(args.bg_csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(args.bg_audio_dir)
        noise_maker = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    extractor, model = load_model(device)
    qast = MaskedAST(model)
    specs = {c: parse_config(c) for c in args.configs}
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    # clean lengths once (noise and mixing keep the length)
    lens_f = out / "lengths16k.npy"
    if lens_f.exists():
        lens = np.load(lens_f)
    else:
        lens = np.array([len(AS.load_clip(index[s])) if s in index else 163840 for s in stems])
        if avail.all():
            np.save(lens_f, lens)
    T = np.array([C.fbank_frames(int(x)) for x in lens])
    tp = np.array([C.time_patches(int(x)) for x in T])
    if (tp < 1).any():
        raise SystemExit("a clip is shorter than one patch")
    bk = np.array([bucket_of(int(x)) for x in tp])
    current = None
    B = args.batch_size
    status = 0
    for kind, snr, rows in plan:
        cname = cname_of(kind, snr)
        todo = [c for c in args.configs if not (out / f"{c}__{cname}.npy").exists()]
        if not todo:
            continue
        if args.max_seconds and time.time() - t_start > args.max_seconds:
            print(f"time budget reached before {cname}; stopping", flush=True)
            status = 3
            break
        part = out / f"_partial__{cname}.npz"
        X = {c: np.full((n, 768), np.nan, np.float32) for c in todo}
        done = np.zeros(n, bool)
        if part.exists():
            d = np.load(part)
            if sorted(d["configs"].tolist()) == sorted(todo):
                done = d["done"]
                for c in todo:
                    X[c] = d[c]
                print(f"{cname}: resuming, {int(done[rows].sum())} rows done", flush=True)
        prep_s, fwd_s, n_silent = 0.0, {c: 0.0 for c in todo}, 0
        t0 = time.time()
        left = rows[~done[rows] & avail[rows]]
        left = left[np.argsort(bk[left], kind="stable")]  # chunks mostly within one bucket
        stopped = False
        for ci, start in enumerate(range(0, len(left), args.chunk)):
            idx = left[start:start + args.chunk]
            tq = time.time()
            waves = []
            for i in idx:
                x = AS.load_clip(index[stems[i]])
                if kind in CR.KINDS:
                    x = CR.corrupt(x, kind, snr)
                elif kind != "clean":
                    x, mixed = mix(x, noise_maker(stems[i], kind, len(x)), snr)
                    n_silent += not mixed
                waves.append(x)
            feats = extractor(waves, sampling_rate=16000, return_tensors="pt")["input_values"]
            prep_s += time.time() - tq
            for c in todo:
                if current != c:
                    qast.set_config(*specs[c])
                    current = c
                tf = time.time()
                for L in sorted(set(bk[idx].tolist())):
                    pos = np.flatnonzero(bk[idx] == L)
                    xb_all = feats[pos, :L].contiguous().to(device)
                    tt_all = torch.as_tensor(tp[idx[pos]], device=device)
                    res = []
                    for b in range(0, len(pos), B):
                        xb, tb = xb_all[b:b + B], tt_all[b:b + B]
                        nb = len(xb)
                        if nb < B:
                            fill = torch.arange(B, device=device) % nb
                            xb, tb = xb[fill], tb[fill]
                        res.append(qast.forward_masked(xb, tb).float().cpu().numpy()[:nb])
                    X[c][idx[pos]] = np.concatenate(res)
                fwd_s[c] += time.time() - tf
            done[idx] = True
            if device.type == "mps":
                torch.mps.empty_cache()
            if (ci + 1) % args.save_every == 0:
                np.savez(part, configs=np.array(todo), done=done, **X)
            el = time.time() - t0
            print(f"  {cname}: {int(done[rows].sum())}/{len(rows)} rows, {el:.0f}s", flush=True)
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                np.savez(part, configs=np.array(todo), done=done, **X)
                print(f"time budget reached inside {cname}; partial saved", flush=True)
                stopped = True
                break
        if stopped:
            status = 3
            break
        if not done[rows].all():
            np.savez(part, configs=np.array(todo), done=done, **X)
            print(f"{cname}: {int(done[rows].sum())}/{len(rows)} rows done (audio missing for the rest); partial saved", flush=True)
            status = max(status, 4)
            continue
        dt = time.time() - t0
        for c in todo:
            if not np.isfinite(X[c][rows]).all():
                raise SystemExit(f"{c}__{cname}: non-finite embeddings")
            bo = np.array([clip_bitops(c, int(t_), int(l_))[0] for t_, l_ in zip(tp, lens)])
            fb = np.array([clip_bitops(c, int(t_), int(l_))[1] for t_, l_ in zip(tp, lens)])
            np.save(out / f"{c}__{cname}.npy", X[c])
            side = {"config": c, "condition": cname, "noise": kind, "snr_db": snr, "n_clips": n,
                    "rows_extracted": int(len(rows)), "silent_clips_left_clean": n_silent, "length_rule": "first 163840 samples at 16 kHz",
                    "buckets": list(BUCKETS), "masked_unpadded": True, "forward_seconds": fwd_s[c], "condition_seconds": dt,
                    "prep_seconds": prep_s, "configs_in_pass": todo, "device": str(device), "batch_size": B,
                    "csv": str(args.csv), "model": C.AST_MODEL, "revision": C.AST_REV, "names_crc32": crc,
                    "time_patches_mean": float(tp.mean()), "backbone_bitops_mean": float(bo.mean()),
                    "fbank_bitops_mean": float(fb.mean()), "price_rescale": 1024.0}
            (out / f"{c}__{cname}.json").write_text(json.dumps(side, indent=2))
        np.save(out / "time_patches.npy", tp)
        if kind == "bg":
            planf = out / "bg_plan.json"
            old = json.loads(planf.read_text()) if planf.exists() else {}
            old.update(noise_maker.plan)
            planf.write_text(json.dumps(old, indent=0))
        if part.exists():
            part.unlink()
        print(f"{cname}: {len(left)} rows x {len(todo)} configs in {dt:.0f}s (prep {prep_s:.0f}s), silent {n_silent}", flush=True)
    print(f"done in {time.time() - t_start:.0f}s, status {status}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
