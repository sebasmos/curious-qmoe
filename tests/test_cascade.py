"""Confidence and uncertainty cascades: thresholds are chosen on the inner
validation split only, never with test labels, and the cascade is scored on
the same (seed, fold) splits as the trained models.

Run: PYTHONPATH=. pytest tests/test_cascade.py -q
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.benchmark as bench
from scripts.analysis import cascade
from test_protocol import synthetic_csv, tiny_cfg


# ── threshold choice ───────────────────────────────────────────────────────

def test_candidates_run_from_all_to_none():
    s = np.array([0.3, 0.1, 0.3, 0.2])
    assert [(s > t).mean() for t in cascade.candidate_thresholds(s)] == [1.0, 0.75, 0.5, 0.0]


def test_max_f1_escalates_exactly_what_q4_gets_wrong():
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    y = np.array([0, 1, 0, 1, 1])
    low, high = np.array([0, 1, 1, 0, 1]), y.copy()
    t = cascade.choose_threshold(scores, y, low, high, "max_f1")
    assert (scores > t).tolist() == [False, False, True, True, False]


def test_matched_rate_and_ties_go_to_fewer_escalations():
    scores = np.linspace(0, 1, 10)
    zeros = np.zeros(10, dtype=int)
    t = cascade.choose_threshold(scores, zeros, zeros, zeros, "match_moe", 0.3)
    assert (scores > t).mean() == pytest.approx(0.3)
    t = cascade.choose_threshold(scores, zeros, zeros, zeros, "max_f1")
    assert not (scores > t).any(), "flat F1: escalate nothing"


# ── on trained runs ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cascade")
    csv = tmp / "data.csv"
    synthetic_csv(csv)
    out = tmp / "outputs"
    bench.run_cv(str(csv), tiny_cfg(csv, out / "syn_singles_s0", ["4", "8", "esc"]))
    bench.run_cv(str(csv), tiny_cfg(csv, out / "syn_pprior_a1.0_s0", ["moe"]))
    cfg = tiny_cfg(csv, out / "syn_sharedtrunk_a1.0_s0", ["moe"])
    cfg.experiment.router.shared_trunk = True
    bench.run_cv(str(csv), cfg)
    return tmp, csv, out


def test_thresholds_are_fit_on_inner_validation_labels(outputs, monkeypatch):
    tmp, csv, out = outputs
    seen, real = [], cascade.choose_threshold

    def spy(scores, y, *a, **k):
        seen.append(np.asarray(y).copy())
        return real(scores, y, *a, **k)

    monkeypatch.setattr(cascade, "choose_threshold", spy)
    cascade.main(["--outputs", str(out), "--out", str(tmp / "casc_spy")])
    labels = pd.read_csv(csv)["class_id"].to_numpy()
    inner = [labels[json.loads((out / "syn_singles_s0" / "4" / f"fold_{f}" / "split.json").read_text())["inner_val_idx"]]
             for f in (1, 2, 3)]
    assert len(seen) == 3 * 8, "8 variants per fold"
    for i, y in enumerate(seen):
        assert np.array_equal(y, inner[i // 8])


def test_changing_test_labels_leaves_thresholds_unchanged(outputs):
    tmp, csv, out = outputs
    res = json.loads(cascade.main(["--outputs", str(out), "--out", str(tmp / "casc_a")])[0].read_text())

    # rewrite the labels of fold 1's test rows only, in a copy of the run tree
    df = pd.read_csv(csv)
    df.loc[df.folder == 1, "class_id"] = 0
    csv_b = tmp / "data_b.csv"
    df.to_csv(csv_b, index=False)
    out_b = tmp / "outputs_b"
    shutil.copytree(out, out_b)
    for c in out_b.glob("*/*/config.yaml"):
        c.write_text(c.read_text().replace(str(csv), str(csv_b)))
    res_b = json.loads(cascade.main(["--outputs", str(out_b), "--out", str(tmp / "casc_b")])[0].read_text())

    a, b = res["folds"][0], res_b["folds"][0]
    assert a["fold"] == b["fold"] == 1
    assert a["moe_inner_val_escalation"] == b["moe_inner_val_escalation"]
    for name, v in a["variants"].items():
        for k in ("threshold", "inner_val_escalation", "inner_val_f1_macro", "escalation_fraction"):
            assert v[k] == b["variants"][name][k], (name, k)
    assert a["singles"]["q4"]["f1_macro"] != b["singles"]["q4"]["f1_macro"], "test labels did change"


def test_records_pair_with_the_moe_runs(outputs):
    tmp, _, out = outputs
    res = json.loads(cascade.main(["--outputs", str(out), "--out", str(tmp / "casc_c")])[0].read_text())
    pprior = {f["fold"]: f for f in json.loads((out / "syn_pprior_a1.0_s0" / "moe" / "summary.json").read_text())["folds"]}
    assert [f["fold"] for f in res["folds"]] == sorted(pprior)
    for f in res["folds"]:
        assert set(f["variants"]) == {f"{s}_{t}_{p}" for s in cascade.SIGNALS for t in cascade.TARGETS
                                      for p in cascade.POINTS}
        assert set(f["moe_arms"]) == {"pprior", "st_pprior"}
        assert f["moe_arms"]["pprior"]["f1_macro"] == pprior[f["fold"]]["f1_macro"]
        for name, s in f["singles"].items():
            assert s["f1_macro"] == pytest.approx(s["summary_f1_macro"], abs=1e-12), "rescoring reproduces the run"
        for v in f["variants"].values():
            extra = 1 + (res["mc_passes"] if v["signal"] == "mc_variance" else 0)
            assert v["compute_rel_q4_macs"] == pytest.approx(extra + v["escalation_fraction"])
            assert v["storage_bytes"] == res["costs"]["q4"]["storage_bytes"] + res["costs"][v["target"]]["storage_bytes"]


def test_moe_cost_counts_every_router_pass():
    from omegaconf import OmegaConf
    from QWave.moe import qMoEModelBatched

    cfg = OmegaConf.load(ROOT / "config" / "esc50.yaml")
    r = cfg.experiment.router
    r.expert_quantizations, r.num_experts, r.use_curiosity, r.mc_samples = ["bitnet", "4", "8"], 3, True, 10
    model = qMoEModelBatched(cfg, 1536, 5, 3, 1)
    router = 1536 * 128 + 128 * 64 + 64 * 3
    expert = 1536 * 640 + 640 * 320 + 320 * 5
    assert cascade.moe_cost(model, [0.0, 1.0, 0.0])[0] == 11 * router + expert
