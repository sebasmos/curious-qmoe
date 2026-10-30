"""Shared pieces of the idea 6 full latency study: paths, clips, models, timing.

Everything is imported read-only from the idea 6 Stage 3c worktree at cbf6be6
(QWave/effnet_quant.py, QWave/ast_frames.py, QWave/backbone_quant.py,
QWave/efficientat_mn.py, scripts/redesign/idea6/stage2/audio_shift.py,
scripts/analysis/cascade.py). Those files are byte-identical to the Stage 2
worktree at c892c2f that built the EfficientNet and AST models of the first
latency study (git diff c892c2f cbf6be6 on them is empty). Run with
PYTHONDONTWRITEBYTECODE=1 so nothing is written into the worktree.

Models reused from the first latency study (OLD): the int8 EfficientNet FX
state dict and its reference output, the ONNX FP32 and int8 QDQ EfficientNet,
the uint8 timing images. New here: EfficientAT mn10 and mn20 FP32 (torch,
released AudioSet weights) and their ONNX Runtime int8 fold-1 models from
Stage 3c, with the EfficientAT front end (resample to 32 kHz, log mel) timed
as its own step.

Models (batch size 1, CPU):
  effnet fp32       torchvision efficientnet_b3 (IMAGENET1K_V1), classifier
                    nn.Identity, as Stage 2's efficientnet_b3_pooled; its
                    StochasticDepth modules are replaced by nn.Identity, which
                    is exact in eval mode (they return the input unchanged
                    when not training) and keeps FX from wrapping each one in
                    a dequantize/quantize pair.
  effnet int8 fx    static post-training quantization, torch.ao FX graph mode,
                    qnnpack engine: weights int8 symmetric per output channel
                    (PerChannelMinMaxObserver), activations quint8 per tensor
                    (HistogramObserver, reduce_range False), calibrated on
                    clean training clips. Conv + BatchNorm are fused; SiLU has
                    no quantized kernel and runs in FP32 between a dequantize
                    and a quantize.
  ast fp32          transformers ASTModel (MIT/ast-finetuned-audioset-10-10-
                    0.4593 at f826b80) cut to 512 or 400 frames by
                    QWave.ast_frames.shorten_ast; embedding = pooler_output
                    (mean of the two special tokens after the last LayerNorm),
                    the same vector Stage 2 stored.
  ast int8 dynamic  torch.ao.quantization.quantize_dynamic on every nn.Linear
                    (the 48 encoder projections), qnnpack, weights int8 per
                    output channel, activations quint8 per tensor computed at
                    run time; attention products, softmax, LayerNorm, GELU and
                    the patch embedding stay FP32.
  ast n4 kleidiai   the 48 encoder Linear layers run on PyTorch's KleidiAI
                    kernel aten._dyn_quant_matmul_4bit: Stage 2's n4 weight
                    codes exactly (8-bit parent codes per output row, nested
                    to signed 4 bits, scale = parent scale x 16), activations
                    int8 per row (token) quantized at run time by the kernel
                    (asymmetric, where n4 simulated symmetric). The attention
                    products stay FP32 (n4 simulated them at 8 bits).
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

W = Path(os.environ.get("W", "/Users/cajas.sebastian/Desktop/repositories/curious-qmoe-redesign-idea6-stage3c"))  # repository root
R = Path("/Users/cajas.sebastian/Documents/qmoe-results")
S2 = R / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
S3B = R / "2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db"
S3C = R / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1"
OLD = R / "2026-10-22_1621_redesign-idea6-latency_code-c892c2f"
OUT = Path(__file__).resolve().parents[1]
MN_WEIGHTS = Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1"
MN_FILES = {"mn10": "mn10_as_mAP_471.pt", "mn20": "mn20_as_mAP_478.pt"}
URBAN_PAD = 64587  # Stage 3b zero-padded UrbanSound8K clips to 4.04 s before the EfficientAT front end
E = Path.home() / "Documents" / "DATASETS"
ESC_CSV = E / "ESC-50/ast_768/esc-50.csv"
ESC_AUDIO = E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"
US_CSV = E / "Urban8k/ast_768/urbansound8k.csv"
US_AUDIO = E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio"
AST_NAME, AST_REV = "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"
SHORT = {"esc50": 512, "urban8k": 400}
N_CALIB, CALIB_SEED = 160, 0

for p in (str(W), str(W / "scripts/redesign/idea6/stage2")):
    if p not in sys.path:
        sys.path.insert(0, p)


def set_threads(n: int) -> None:
    """Threads for torch (OpenMP and the pthreadpool used by qnnpack and
    KleidiAI). Accelerate's BLAS reads VECLIB_MAXIMUM_THREADS when it loads,
    so the launcher sets it (and OMP_NUM_THREADS) in the environment of a
    fresh process per thread count; this checks that it did."""
    import torch

    for var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(var) != str(n):
            raise SystemExit(f"{var}={os.environ.get(var)} but {n} threads asked: launch with the env set")
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)
    torch.backends.quantized.engine = "qnnpack"


def loadavg():
    return [round(x, 2) for x in os.getloadavg()]


def busy_processes(n=8):
    """Top CPU users (pid, nice, %cpu, command head), to record who else ran."""
    r = subprocess.run(["ps", "-Ao", "pid=,ni=,pcpu=,comm="], capture_output=True, text=True)
    rows = []
    for line in r.stdout.splitlines():
        f = line.split(None, 3)
        if len(f) == 4:
            rows.append((int(f[0]), int(f[1]), float(f[2]), f[3][-60:]))
    rows.sort(key=lambda t: -t[2])
    return rows[:n]


def wait_quiet(max_load: float, tries: int = 720, poll: float = 30.0) -> dict:
    """Wait (up to tries x poll s, 6 h) while the 1-minute load is above max_load, printing the load every
    ten checks.
    The idle load of this machine is 6 to 7.5 (VS Code helpers and an IT agent each hold a
    core), so the gate is 9, and the load before and after every block is recorded."""
    waited = 0
    while os.getloadavg()[0] > max_load and waited < tries:
        if waited % 10 == 0:
            print(f"  waiting, load {loadavg()} above {max_load} ({waited * poll:.0f} s)", flush=True)
        time.sleep(poll)
        waited += 1
    return {"load_before": loadavg(), "waited_s": waited * poll, "still_high": os.getloadavg()[0] > max_load}


def time_fn(fn, warm: int, reps: int) -> np.ndarray:
    """Per-pass wall clock in ms of fn() after warm-up passes."""
    for _ in range(warm):
        fn()
    t = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        t[i] = (time.perf_counter_ns() - t0) / 1e6
    return t


def summarize(t: np.ndarray) -> dict:
    q25, med, q75, p95 = np.percentile(t, [25, 50, 75, 95])
    return {"median_ms": float(med), "q25_ms": float(q25), "q75_ms": float(q75), "iqr_ms": float(q75 - q25),
            "p95_ms": float(p95), "mean_ms": float(t.mean()), "n": int(len(t))}


# ── clips ──────────────────────────────────────────────────────────────────

def esc_meta():
    import pandas as pd

    return pd.read_csv(ESC_CSV, usecols=["folder", "name", "class_id"])


def load_waves(names, audio_dir):
    from audio_shift import index_wavs, load_clip

    index = index_wavs(audio_dir)
    return [load_clip(index[Path(n).stem]) for n in names]


def calibration_rows():
    """N_CALIB ESC-50 rows from folds 2 to 5 (the training folds of fold 1),
    numpy default_rng(CALIB_SEED), sorted."""
    meta = esc_meta()
    pool = np.flatnonzero(meta["folder"].to_numpy() != 1)
    return np.sort(np.random.default_rng(CALIB_SEED).choice(pool, N_CALIB, replace=False))


def fold1_test_rows():
    meta = esc_meta()
    return np.flatnonzero(meta["folder"].to_numpy() == 1)


def timing_clip(ds: str):
    """Stage 2 latency.py's clip: the first full-length clip in CSV order."""
    import pandas as pd

    from audio_shift import index_wavs, load_clip

    csv, audio = (ESC_CSV, ESC_AUDIO) if ds == "esc50" else (US_CSV, US_AUDIO)
    index = index_wavs(audio)
    for nm in pd.read_csv(csv, usecols=["name"])["name"]:
        wave = load_clip(index[Path(nm).stem])
        if len(wave) >= (80000 if ds == "esc50" else 64000):
            return nm, wave
    raise SystemExit("no full-length clip")


def images_u8(waves):
    """Stage 2's mel images as uint8 (they are k / 255 exactly)."""
    import torch

    from QWave.effnet_quant import MelImageSpec, mel_images

    x = mel_images(waves, MelImageSpec())
    u = torch.round(x * 255).to(torch.uint8)
    if not torch.equal(u.float() / 255.0, x):
        raise SystemExit("mel image is not k / 255")
    return u


# ── EfficientNet ───────────────────────────────────────────────────────────

def effnet_fp32():
    from torch import nn
    from torchvision.ops import StochasticDepth

    from QWave.effnet_quant import efficientnet_b3_pooled

    m = efficientnet_b3_pooled().eval()

    def strip(mod):
        for name, child in mod.named_children():
            if isinstance(child, StochasticDepth):
                setattr(mod, name, nn.Identity())
            else:
                strip(child)

    strip(m)
    return m


def effnet_int8_fx(fp32_model, calib_images):
    """FX static PTQ (qnnpack). calib_images: float [N, 3, 300, 300]."""
    import torch
    from torch.ao.quantization import QConfig, get_default_qconfig_mapping
    from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "qnnpack"
    qc = QConfig(activation=HistogramObserver.with_args(reduce_range=False, dtype=torch.quint8),
                 weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
    qmap = get_default_qconfig_mapping("qnnpack").set_global(qc)
    prepared = prepare_fx(copy.deepcopy(fp32_model).eval(), qmap, (calib_images[:1],))
    with torch.no_grad():
        for i in range(len(calib_images)):
            prepared(calib_images[i:i + 1])
    return convert_fx(prepared).eval()


def load_effnet_int8(path, fp32_model, calib_image):
    """Rebuild the converted graph (prepare, one observer pass, convert) and
    load the saved state dict: scales, zero points and packed int8 weights.
    torch.save of the whole quantized GraphModule does not reload."""
    import torch

    q = effnet_int8_fx(fp32_model, calib_image[:1])
    q.load_state_dict(torch.load(path, weights_only=False), strict=True)
    return q.eval()


def float_islands(gm) -> dict:
    """Count what runs where in a converted FX graph: quantized modules and
    functions, and the float modules that sit between a dequantize and a
    quantize."""
    import collections

    mods = dict(gm.named_modules())
    kinds = collections.Counter()
    for n in gm.graph.nodes:
        if n.op == "call_module":
            t = type(mods[n.target])
            kinds[f"module {t.__module__}.{t.__name__}"] += 1
        elif n.op == "call_function":
            name = str(n.target) if "OpOverload" in type(n.target).__name__ else getattr(n.target, "__name__", str(n.target))
            kinds[f"function {name}"] += 1
        elif n.op == "call_method":
            kinds[f"method {n.target}"] += 1
    return dict(sorted(kinds.items()))


# ── AST ────────────────────────────────────────────────────────────────────

def ast_fp32(max_length: int):
    from transformers import ASTModel

    from QWave.ast_frames import shorten_ast

    return shorten_ast(ASTModel.from_pretrained(AST_NAME, revision=AST_REV), max_length).eval()


def ast_extractor(max_length: int):
    from transformers import ASTFeatureExtractor

    fe = ASTFeatureExtractor.from_pretrained(AST_NAME, revision=AST_REV, max_length=max_length)
    if fe.max_length != max_length:
        raise SystemExit("feature extractor ignored max_length")
    return fe


def ast_int8_dynamic(model):
    import torch
    from torch import nn
    from torch.ao.quantization import per_channel_dynamic_qconfig, quantize_dynamic

    torch.backends.quantized.engine = "qnnpack"
    n_lin = sum(isinstance(m, nn.Linear) for m in model.modules())
    q = quantize_dynamic(copy.deepcopy(model), {nn.Linear: per_channel_dynamic_qconfig}, dtype=torch.qint8)
    return q.eval(), n_lin


def n4_codes(w):
    """Stage 2's n4 weight: (signed 4-bit codes [out, in] int8, scale [out])."""
    import torch

    from QWave.backbone_quant import PARENT_BITS, nested_codes, rtn_codes

    codes8, s8 = rtn_codes(w, PARENT_BITS, None)
    c4 = nested_codes(codes8, 4).reshape(w.shape)
    s4 = (s8 * 2 ** (PARENT_BITS - 4)).reshape(w.shape[0])
    return c4.to(torch.int8), s4.float()


def make_kai_linear(lin):
    import torch
    from torch import nn

    class _KaiN4Linear(nn.Module):
        def __init__(self, lin: nn.Linear):
            super().__init__()
            k, n = lin.in_features, lin.out_features
            codes, s = n4_codes(lin.weight.data)
            u = (codes.to(torch.int16) + 8).to(torch.uint8)  # 0..15
            packed = (u[:, 1::2] << 4 | u[:, ::2]).to(torch.uint8).contiguous()
            bias = lin.bias.data.float().contiguous() if lin.bias is not None else None
            self.k, self.n = k, n
            self.register_buffer("packed", torch.ops.aten._dyn_quant_pack_4bit_weight(
                packed, s.reshape(n, 1).contiguous(), bias, k, k, n))

        def forward(self, x):
            shp = x.shape
            y = torch.ops.aten._dyn_quant_matmul_4bit(x.reshape(-1, self.k).contiguous(), self.packed,
                                                      self.k, self.k, self.n)
            return y.reshape(*shp[:-1], self.n)

    return _KaiN4Linear(lin)


def ast_n4_kleidiai(model):
    """A copy of an ASTModel whose nn.Linear layers run on KleidiAI with
    Stage 2's n4 codes. Returns (model, number of layers replaced)."""
    import torch
    from torch import nn

    if not torch.backends.kleidiai.is_available():
        raise SystemExit("KleidiAI not available in this torch build")
    m = copy.deepcopy(model).eval()
    count = 0

    def swap(mod):
        nonlocal count
        for name, child in mod.named_children():
            if isinstance(child, nn.Linear):
                setattr(mod, name, make_kai_linear(child))
                count += 1
            else:
                swap(child)

    swap(m)
    return m, count


# ── EfficientAT ────────────────────────────────────────────────────────────

def mn_wave(ds: str, wave):
    """The waveform the EfficientAT tier ran on in Stage 3b and 3c: UrbanSound8K clips zero-padded to
    URBAN_PAD samples (4.04 s), ESC-50 clips as they are (5 s)."""
    if ds == "urban8k" and len(wave) < URBAN_PAD:
        wave = np.pad(wave, (0, URBAN_PAD - len(wave)))
    return wave


def mn_fp32(name: str):
    from QWave import efficientat_mn as M

    return M.load_mn(name, MN_WEIGHTS / MN_FILES[name])


def mn_input(wave16):
    """EfficientAT's documented front end on one 16 kHz clip: resample to 32 kHz, pre-emphasis, STFT, 128 Kaldi
    mel bands, log, normalization -> [1, 1, 128, frames]."""
    from QWave import efficientat_mn as M

    return M.model_input([wave16])[0][1]


def git_head(path: Path) -> dict:
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    st = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True).stdout
    return {"path": str(path), "head": head, "clean": st.strip() == ""}
