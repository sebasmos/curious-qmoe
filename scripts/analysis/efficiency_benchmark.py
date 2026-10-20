#!/usr/bin/env python3
"""Per-sample inference latency and energy of trained checkpoints.

Run after training, alone on the machine (never alongside training jobs).
Each --arm names one trained fold, NAME=RUN_DIR[@FOLD], where RUN_DIR is
outputs/<tag>/<model_kind> from scripts/benchmark.py. For every arm the
script rebuilds the network exactly as it was scored (checkpoint loaded, PTQ
applied) and feeds it that fold's held-out test samples, scaled with the
fold's fit-portion scaler, one at a time.

Latency: batch size 1, `--warmup` untimed calls then `--reps` timed calls per
arm, with the arms interleaved call by call so drift and thermal state hit
every arm alike. Torch threads are pinned (`--threads`, inter-op 1) and
recorded. Reported: median and IQR in ms, plus the raw timings for Levene's
test in the collector. For MoE arms with a Bayesian router the router alone
is timed inside the same trained model, interleaved with the full forward.

Energy: CodeCarbon around `--energy-inferences` batch-1 calls per arm, arm by
arm, converted to mJ per inference (kWh x 3.6e9 / n) and ug CO2 per inference
(kg x 1e9 / n). The CPU measurement mode, CPU model and power are recorded:
on Apple silicon without powermetrics CodeCarbon runs in constant (TDP) mode,
so energy there is a constant power times the measured time.

Sub-8-bit and BitNet layers are simulated in fp32 (fake quantization), so
their latency is not a hardware measurement; arms containing them are
flagged `simulated_quantization`. Time FP32 (esc), dynamic int8 (qesc) and
MoE configurations; take storage size for every bit-width from summary.json.

    python scripts/analysis/efficiency_benchmark.py \\
        --arm fp32=outputs/esc50_singles_s0/esc --arm int8=outputs/esc50_singles_s0/qesc \\
        --arm moe_pprior=outputs/esc_pprior_a1.0_s0/moe --threads 2 \\
        --out outputs/efficiency/esc.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

KWH_TO_MJ = 3.6e9  # 1 kWh = 3.6e6 J = 3.6e9 mJ
KG_TO_UG = 1e9     # 1 kg = 1e9 ug


def kwh_to_mj(kwh: float) -> float:
    return kwh * KWH_TO_MJ


def kg_to_ug(kg: float) -> float:
    return kg * KG_TO_UG


def per_inference(total: float, n: int) -> float:
    return total / n


def latency_summary(ms) -> dict:
    a = np.asarray(ms, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"median_ms": float(med), "q1_ms": float(q1), "q3_ms": float(q3), "iqr_ms": float(q3 - q1),
            "mean_ms": float(a.mean()), "std_ms": float(a.std()), "p95_ms": float(np.percentile(a, 95)),
            "n": int(a.size)}


def simulated_quantization(model) -> bool:
    from QWave.qmoe_layers import BitLinear
    return any(isinstance(m, BitLinear) and not (isinstance(m.num_bits, int) and m.num_bits >= 32)
               for m in model.modules())


def has_bayesian_router(model) -> bool:
    from QWave.moe import BayesianRouter
    return isinstance(getattr(model, "router", None), BayesianRouter)


def time_interleaved(arms, warmup: int, reps: int) -> dict:
    """arms: {name: (callable taking one (1, D) tensor, X)}; returns ms lists."""
    names = list(arms)
    timings = {n: [] for n in names}
    with torch.no_grad():
        for i in range(warmup):
            for n in names:
                fn, X = arms[n]
                fn(X[i % len(X)].unsqueeze(0))
        for i in range(reps):
            for n in names:
                fn, X = arms[n]
                x = X[i % len(X)].unsqueeze(0)
                t0 = time.perf_counter_ns()
                fn(x)
                timings[n].append((time.perf_counter_ns() - t0) / 1e6)
    return timings


def measure_energy(fn, X, n: int, country_iso: str) -> dict:
    from codecarbon import OfflineEmissionsTracker
    tracker = OfflineEmissionsTracker(country_iso_code=country_iso, save_to_file=False,
                                      log_level="error", measure_power_secs=1)
    cpu_mode = next((getattr(h, "_mode", None) for h in tracker._hardware if type(h).__name__ == "CPU"), None)
    with torch.no_grad():
        tracker.start()
        for i in range(n):
            fn(X[i % len(X)].unsqueeze(0))
        tracker.stop()
    d = tracker.final_emissions_data
    return {
        "n_inferences": n,
        "duration_s": float(d.duration),
        "energy_kwh": float(d.energy_consumed),
        "energy_mj_per_inference": per_inference(kwh_to_mj(d.energy_consumed), n),
        "cpu_energy_mj_per_inference": per_inference(kwh_to_mj(d.cpu_energy), n),
        "ram_energy_mj_per_inference": per_inference(kwh_to_mj(d.ram_energy), n),
        "co2_ug_per_inference": per_inference(kg_to_ug(d.emissions), n),
        "cpu_model": d.cpu_model,
        "cpu_power_w": float(d.cpu_power),
        "ram_power_w": float(d.ram_power),
        "tracking_mode": d.tracking_mode,
        "cpu_measurement_mode": cpu_mode,
        "country_iso_code": country_iso,
        "codecarbon_version": d.codecarbon_version,
    }


def parse_arm(spec: str):
    name, _, target = spec.partition("=")
    run_dir, _, fold = target.partition("@")
    if not name or not run_dir:
        raise argparse.ArgumentTypeError(f"--arm expects NAME=RUN_DIR[@FOLD], got '{spec}'")
    return name, Path(run_dir).resolve(), int(fold) if fold else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", type=parse_arm, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--energy-inferences", type=int, default=5000)
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--country-iso", default="IRL", help="grid used for the CO2 conversion")
    ap.add_argument("--csv", default=None, help="dataset CSV, if it moved since training")
    ap.add_argument("--seed", type=int, default=0, help="seeds MC dropout during timing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # only settable before torch starts inter-op work
        pass
    from scripts.analysis.run_loader import load_trained_fold
    from QWave.memory import storage_bits

    loaded, calls = {}, {}
    for name, run_dir, fold in args.arm:
        tf = load_trained_fold(run_dir, fold, csv=args.csv)
        loaded[name] = (run_dir, tf)
        calls[name] = (tf.model, tf.X_test)
        if has_bayesian_router(tf.model):
            router, unc = tf.model.router, tf.model.use_curiosity
            calls[f"{name}::router"] = (lambda x, r=router, u=unc: r(x, compute_uncertainty=u), tf.X_test)

    torch.manual_seed(args.seed)
    timings = time_interleaved(calls, args.warmup, args.reps)

    results = {
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                    "interop_threads": torch.get_num_interop_threads(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
        "protocol": {"batch_size": 1, "warmup": args.warmup, "reps": args.reps, "interleaved": True,
                     "inputs": "held-out test-fold samples, scaled with the fold's fit-portion scaler",
                     "energy_inferences": None if args.no_energy else args.energy_inferences},
        "arms": {},
    }
    for name, (run_dir, tf) in loaded.items():
        arm = {"run_dir": str(run_dir), "model_kind": tf.model_kind, "fold": tf.fold,
               "seed": int(tf.cfg.experiment.get("seed", 0)),
               "simulated_quantization": simulated_quantization(tf.model),
               "storage_bytes": storage_bits(tf.model) / 8,
               "latency": latency_summary(timings[name]), "latency_raw_ms": timings[name]}
        if f"{name}::router" in timings:
            router = latency_summary(timings[f"{name}::router"])
            arm["router_latency"] = router
            arm["router_share_of_median"] = router["median_ms"] / arm["latency"]["median_ms"]
        if not args.no_energy:
            torch.manual_seed(args.seed)
            arm["energy"] = measure_energy(tf.model, tf.X_test, args.energy_inferences, args.country_iso)
        results["arms"][name] = arm
        lat = arm["latency"]
        print(f"{name:24s} median {lat['median_ms']:.3f} ms  IQR {lat['iqr_ms']:.3f} ms"
              + (f"  energy {arm['energy']['energy_mj_per_inference']:.3f} mJ/inf" if "energy" in arm else ""))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")
    return results


if __name__ == "__main__":
    main()
