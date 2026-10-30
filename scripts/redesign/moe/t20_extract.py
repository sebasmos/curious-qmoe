"""T20 extraction (ANALYSIS_PLAN_T20.md 3.4): logits of the fine-tuned AST tails (ft-FP32 and ft-Q4 = LoRA merged, every encoder Linear
w4g128 with 8-bit activations, QWave.backbone_quant) on one fold for all seeds. The frozen blocks 0 to 9 run once per batch in FP32 and
once in w4g128; each seed's tail runs on both. Filter banks are computed from audio in float32 with stage 2's mixer, as the stored statics.
Rows: test clips clean and under the test field assignment; each seed's inner-validation clips clean and under the validation field
assignment; TAU: every test row, each seed's inner-validation rows and 600 sampled non-A training-fold rows (seed 20).
    python t20_extract.py --task urban --fold 1 --runs RUNS_DIR --out SCORES_DIR [--seeds 0 1 2] [--check-only] [--limit N]"""
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--runs", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--tag", default=""); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--check-only", action="store_true"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--device", default="mps"); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = str(a.threads)
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / "idea6/stage2"))
from t17c_ast import load_ast, front, Tail, NL, layer_fwd, _linears, LoRALinear
from t20_common import *
from audio_shift import FIELD_SEEDS, NoiseMaker, field_assignment, index_wavs, load_clip, mix
from QWave.backbone_quant import QuantAST, parse_config, quantize_weight
torch.set_num_threads(a.threads); dev = torch.device(a.device); T = TASKS[a.task]; L, K = T["L"], T["K"]; B = BLOCK
D = bed(a.task); y = D["y"]; N = D["N"]; st = stems(a.task); te = test_rows(a.task, D, a.fold)
index = index_wavs(T["aud"]); nm = NoiseMaker([], {}, False)
if T["bg_csv"] is not None:
    bgn = [Path(s).stem for s in pd.read_csv(T["bg_csv"], usecols=["name"])["name"]]; bgi = index_wavs(T["bg_aud"]); nm = NoiseMaker([b for b in bgn if b in bgi], bgi, True)
    at, av = (np.array(field_assignment(N, FIELD_SEEDS[k])) for k in ("test", "val"))
ext, model = load_ast(L, dev); spec, abits = parse_config("w4g128"); q = QuantAST(model); q.set_config(spec, abits)


def wave(r, cell):
    x = load_clip(index[st[r]])
    if cell != "clean":
        kind, snr = cell.split("__"); x, _ = mix(x, nm(st[r], kind, len(x)), float(snr[:-2]))
    return x


def fbank(jobs): return ext([wave(r, c) for r, c in jobs], sampling_rate=16000, return_tensors="pt")["input_values"].contiguous().to(dev)


@torch.no_grad()
def front_q(x):
    h = model.embeddings(x)
    for i in range(B): h = q._layer(h, model.encoder.layer[i], q.weights[i], q.bias[i])
    return h


def fused(layer):
    """Merged (LoRA folded in) fused weights of one tail layer, in QuantAST's slot layout."""
    g = lambda m, n: (getattr(m, n).merged() if isinstance(getattr(m, n), LoRALinear) else getattr(m, n))
    at_ = layer.attention.attention; qq, kk, vv = g(at_, "query"), g(at_, "key"), g(at_, "value")
    W = {"qkv": torch.cat([qq.weight, kk.weight, vv.weight]), "attn_out": g(layer.attention.output, "dense").weight, "fc1": g(layer.intermediate, "dense").weight, "fc2": g(layer.output, "dense").weight}
    Bv = {"qkv": torch.cat([qq.bias, kk.bias, vv.bias]), "attn_out": g(layer.attention.output, "dense").bias, "fc1": g(layer.intermediate, "dense").bias, "fc2": g(layer.output, "dense").bias}
    return {k: w.detach().clone() for k, w in W.items()}, {k: b.detach().clone() for k, b in Bv.items()}


class SeedTail:
    def __init__(self, path):
        self.t = Tail(model, B, "lora", ncls=K); st_ = torch.load(path, map_location=dev); missing = set(st_) - set(self.t.state_dict()); assert not missing, missing
        self.t.load_state_dict(st_, strict=False); self.t.eval(); self.Wq, self.Bq = [], []
        with torch.no_grad():
            for l in self.t.layers:
                W, Bv = fused(l); self.Wq.append({k: quantize_weight(w, spec) for k, w in W.items()}); self.Bq.append(Bv)
    @torch.no_grad()
    def q4(self, h, pooled=False):
        for j, i in enumerate(range(B, NL)): h = q._layer(h, model.encoder.layer[i], self.Wq[j], self.Bq[j])
        h = self.t.ln(h); p = (h[:, 0] + h[:, 1]) / 2; return p if pooled else self.t.head(p)
    @torch.no_grad()
    def fp32(self, h): return self.t(h)


def rel(x, y_): return np.linalg.norm(x - y_, axis=1) / np.linalg.norm(y_, axis=1)


# ---------- checks (a), (b), (c) ----------
chk = {}
with torch.no_grad():
    sub = te[:: max(1, len(te) // 32)][:32]; cells = ["clean"] * len(sub)
    if T["bg_csv"] is not None:
        nz = te[at[te] != "clean"]; sub_f = nz[:: max(1, len(nz) // 32)][:32]; jobs = [(int(r), "clean") for r in sub] + [(int(r), str(at[r])) for r in sub_f]
    else: jobs = [(int(r), "clean") for r in sub]
    x = fbank(jobs); full = np.concatenate([q(x[i:i + 16]).float().cpu().numpy() for i in range(0, len(jobs), 16)])
    base = SeedTail.__new__(SeedTail); base.t = Tail(model, B, "lora", ncls=K); base.t.eval(); base.Wq = [q.weights[i] for i in range(B, NL)]; base.Bq = [q.bias[i] for i in range(B, NL)]
    split = np.concatenate([base.q4(front_q(x[i:i + 16]), pooled=True).float().cpu().numpy() for i in range(0, len(jobs), 16)])
    chk["a_split_vs_quantast_rel_max"] = float(rel(split, full).max())
    W0, B0 = zip(*[fused(l) for l in base.t.layers]); base.Wq = [{k: quantize_weight(w, spec) for k, w in W.items()} for W in W0]; base.Bq = list(B0)
    split2 = np.concatenate([base.q4(front_q(x[i:i + 16]), pooled=True).float().cpu().numpy() for i in range(0, len(jobs), 16)])
    chk["a_zero_lora_requantized_vs_quantast_rel_max"] = float(rel(split2, full).max())
    if a.task == "tau": stored = np.load(OPTB / "emb_ast/tau/w4g128__clean.npy", mmap_mode="r")
    sd = S2 / "emb_ast_short" / ("urban8k" if a.task == "urban" else "esc50")
    ref = np.stack([np.asarray(np.load(sd / f"w4g128__{c}.npy", mmap_mode="r")[r]) if a.task != "tau" else np.asarray(stored[r]) for r, c in jobs])
    chk["b_vs_stored_rel_median"] = float(np.median(rel(full, ref))); chk["b_vs_stored_rel_max"] = float(rel(full, ref).max()); chk["b_n"] = len(jobs)
    def plain_pooled(xx):
        h = front(model, xx, B)
        for l in model.encoder.layer[B:]: h = layer_fwd(l, h)
        h = model.layernorm(h); return ((h[:, 0] + h[:, 1]) / 2).float().cpu().numpy()
    fp = np.concatenate([base.t.pooled(front(model, x[i:i + 16], B)).float().cpu().numpy() for i in range(0, len(jobs), 16)])
    chk["c_zero_lora_fp32_tail_vs_model_rel_max"] = float(rel(fp, np.concatenate([plain_pooled(x[i:i + 16]) for i in range(0, len(jobs), 16)])).max())
print(json.dumps(chk), flush=True)
assert chk["a_split_vs_quantast_rel_max"] < 1e-5 and chk["a_zero_lora_requantized_vs_quantast_rel_max"] < 1e-5 and chk["b_vs_stored_rel_max"] < 1e-4 and chk["c_zero_lora_fp32_tail_vs_model_rel_max"] < 1e-5, chk
a.out.mkdir(parents=True, exist_ok=True); (a.out / f"check_f{a.fold}.json").write_text(json.dumps(chk, indent=1))
if a.check_only: sys.exit(0)

# ---------- jobs ----------
tails = {s: SeedTail(a.runs / (f"f{a.fold}_s{s}" + (f"_{a.tag}" if a.tag else "") + "_tail.pt")) for s in a.seeds}
ivs = {s: inner_split(a.task, s, a.fold) for s in a.seeds}
need = {}                                   # (row, cell) -> set of seeds; test jobs need every seed
def add(r, c, s): need.setdefault((int(r), str(c)), set()).update(s)
if a.task == "tau":
    oth = np.flatnonzero((D["fold"] != a.fold) & (~D["is_a"])); rs = np.random.default_rng(20)
    siv = np.sort(np.concatenate([rs.choice(oth[D["dev"][oth] == d], size=min(600 // 8, (D["dev"][oth] == d).sum()), replace=False) for d in sorted(set(D["dev"][oth]))]))
    for r in te: add(r, "clean", a.seeds)
    for s in a.seeds:
        for r in ivs[s]: add(r, "clean", [s])
    for r in siv: add(r, "clean", a.seeds)
else:
    for r in te: add(r, "clean", a.seeds); add(r, at[r], a.seeds)
    for s in a.seeds:
        for r in ivs[s]: add(r, "clean", [s]); add(r, av[r], [s])
keys = sorted(need, key=lambda k: (k[1], k[0]))
if a.limit: keys = keys[:a.limit]
Z = {(s, cfg): {} for s in a.seeds for cfg in ("q4", "fp32")}
t0 = time.time(); tp = tf = 0.0
for c0 in range(0, len(keys), 64):
    jobs = keys[c0:c0 + 64]; t1 = time.time(); x = fbank(jobs); tp += time.time() - t1; t1 = time.time()
    for i in range(0, len(jobs), a.bs):
        xb, jb = x[i:i + a.bs], jobs[i:i + a.bs]; hq, hf = front_q(xb), front(model, xb, B)
        for s in a.seeds:
            m = [j for j, k in enumerate(jb) if s in need[k]]
            if not m: continue
            zq, zf = tails[s].q4(hq[m]).float().cpu().numpy(), tails[s].fp32(hf[m]).float().cpu().numpy()
            for j, u, v in zip(m, zq, zf): Z[(s, "q4")][jb[j]] = u; Z[(s, "fp32")][jb[j]] = v
    tf += time.time() - t1
    if (c0 // 64) % 10 == 0: print(f"{c0 + len(jobs)}/{len(keys)} jobs {time.time() - t0:.0f}s (prep {tp:.0f}s, gpu {tf:.0f}s)", flush=True)
if a.limit: print("limit run, nothing saved"); sys.exit(0)
for s in a.seeds:
    out = dict(te=te, iv=ivs[s], y=y, fold=a.fold, seed=s)
    for cfg in ("q4", "fp32"):
        g = lambda rows, cells: np.stack([Z[(s, cfg)][(int(r), str(c))] for r, c in zip(rows, cells)]).astype(np.float32)
        out[f"{cfg}_clean_te"] = g(te, ["clean"] * len(te)); out[f"{cfg}_clean_iv"] = g(ivs[s], ["clean"] * len(ivs[s]))
        if a.task == "tau": out["siv"] = siv; out[f"{cfg}_clean_siv"] = g(siv, ["clean"] * len(siv))
        else: out[f"{cfg}_field_te"] = g(te, at[te]); out[f"{cfg}_field_iv"] = g(ivs[s], av[ivs[s]])
    assert all(np.isfinite(v).all() for k, v in out.items() if k.startswith(("q4", "fp32")))
    np.savez(a.out / f"f{a.fold}_s{s}.npz", **out)
meta = dict(task=a.task, fold=a.fold, seeds=a.seeds, n_jobs=len(keys), seconds=time.time() - t0, prep_seconds=tp, gpu_seconds=tf, checks=chk)
(a.out / f"f{a.fold}_meta.json").write_text(json.dumps(meta, indent=1)); print("done", json.dumps(meta), flush=True)
