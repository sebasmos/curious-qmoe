"""Efficiency benchmark (unit conversions, latency statistics, interleaving,
CodeCarbon energy) and held-out routing analysis on a trained run.

Run: PYTHONPATH=. pytest tests/test_efficiency_and_routing.py -q
"""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.benchmark as bench
from scripts.analysis import efficiency_benchmark as eff
from scripts.analysis import routing_analysis
from scripts.analysis.run_loader import load_trained_fold, list_folds
from test_protocol import synthetic_csv, tiny_cfg, N_ROWS, N_FOLDS


# ── unit conversions ───────────────────────────────────────────────────────

def test_kwh_to_mj():
    assert eff.kwh_to_mj(1.0) == 3.6e9
    # 42.5 W for 2 s is 85 J, i.e. 85,000 mJ
    assert eff.kwh_to_mj(42.5 * 2 / 3.6e6) == pytest.approx(85_000)


def test_kg_to_ug_and_per_inference():
    assert eff.kg_to_ug(1.0) == 1e9
    assert eff.kg_to_ug(2.5e-9) == pytest.approx(2.5)
    assert eff.per_inference(eff.kwh_to_mj(1e-6), 1000) == pytest.approx(3.6)


def test_latency_summary_median_and_iqr():
    s = eff.latency_summary(np.arange(1, 101))
    assert s["median_ms"] == 50.5 and s["q1_ms"] == 25.75 and s["q3_ms"] == 75.25
    assert s["iqr_ms"] == 49.5 and s["n"] == 100


def test_arms_are_timed_interleaved():
    order = []
    arms = {"a": (lambda x: order.append("a"), torch.zeros(3, 2)),
            "b": (lambda x: order.append("b"), torch.zeros(3, 2))}
    t = eff.time_interleaved(arms, warmup=2, reps=5)
    assert len(t["a"]) == len(t["b"]) == 5
    assert order == ["a", "b"] * 7


def test_energy_is_constant_power_times_time_in_mj():
    model = torch.nn.Linear(8, 3).eval()
    e = eff.measure_energy(model, torch.randn(10, 8), n=300, country_iso="IRL")
    expected = (e["cpu_power_w"] + e["ram_power_w"]) * e["duration_s"] * 1000 / 300
    if e["cpu_measurement_mode"] == "constant":
        assert e["energy_mj_per_inference"] == pytest.approx(expected, rel=0.05)
    assert e["energy_mj_per_inference"] == pytest.approx(eff.kwh_to_mj(e["energy_kwh"]) / 300)
    assert {"cpu_model", "tracking_mode", "cpu_power_w"} <= set(e)


# ── on a trained run ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eff")
    csv = tmp / "data.csv"
    synthetic_csv(csv)
    bench.run_cv(str(csv), tiny_cfg(csv, tmp / "run", ["esc", "qesc", "moe"]))
    return tmp / "run"


@pytest.mark.parametrize("kind", ["esc", "qesc", "moe"])
def test_loader_reproduces_the_scored_test_fold(trained, kind):
    summary = json.loads((trained / kind / "summary.json").read_text())
    for f in summary["folds"]:
        tf = load_trained_fold(trained / kind, f["fold"])
        # score through a DataLoader like run_cv: creating its iterator draws
        # once from the global RNG, which shifts the MC-dropout masks
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tf.X_test, tf.y_test),
                                             batch_size=summary["train_recipe"]["batch_size"])
        if kind == "moe":
            _, _, preds = bench.route_test_fold(tf.model, loader, "cpu", summary["seed"])
        else:
            with torch.no_grad():
                preds = tf.model(tf.X_test).argmax(1).numpy()
        f1 = f1_score(tf.y_test.numpy(), preds, average="macro", zero_division=0)
        assert f1 == pytest.approx(f["f1_macro"]), f"{kind} fold {f['fold']}"


def test_efficiency_benchmark_end_to_end(trained, tmp_path):
    out = tmp_path / "eff.json"
    res = eff.main(["--arm", f"fp32={trained / 'esc'}", "--arm", f"int8={trained / 'qesc'}@2",
                    "--arm", f"moe={trained / 'moe'}", "--warmup", "3", "--reps", "20",
                    "--no-energy", "--threads", "1", "--out", str(out)])
    assert json.loads(out.read_text())["protocol"]["batch_size"] == 1
    arms = res["arms"]
    assert arms["int8"]["fold"] == 2 and len(arms["fp32"]["latency_raw_ms"]) == 20
    assert not arms["fp32"]["simulated_quantization"] and not arms["int8"]["simulated_quantization"]
    assert arms["moe"]["simulated_quantization"], "BitNet and k-bit experts are fake-quantized"
    assert 0 < arms["moe"]["router_share_of_median"] < 1
    assert arms["int8"]["storage_bytes"] < arms["fp32"]["storage_bytes"]


def test_routing_analysis_uses_held_out_samples_only(trained, tmp_path):
    args = Namespace(run_dir=str(trained / "moe"), folds=None, csv=None, synthetic=False,
                     output_dir=str(tmp_path), device="cpu", expert_names=None,
                     in_dim=16, num_classes=3, mc_samples=3)
    routing_analysis.run_analysis(args)
    res = json.loads((tmp_path / "routing_results.json").read_text())
    assert res["config"]["folds"] == list_folds(trained / "moe") == list(range(1, N_FOLDS + 1))
    assert sorted(r["idx"] for r in res["routing"]) == list(range(N_ROWS)), "every row once, as a test sample"
    for r in res["routing"]:
        split = json.loads((trained / "moe" / f"fold_{r['fold']}" / "split.json").read_text())
        assert r["idx"] in split["test_idx"]
    assert res["config"]["expert_names"] == ["BitNet", "Q4", "Q8"]
