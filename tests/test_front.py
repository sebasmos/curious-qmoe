"""The accuracy and cost front and the success bar (scripts/redesign/front.py)
on synthetic records.

Run: PYTHONPATH=. pytest tests/test_front.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.redesign import front

SEEDS, FOLDS = range(5), range(1, 6)
# single models: name -> (bit-ops per clip, mean F1)
SINGLES = {"q1": (1e6, 0.60), "q4": (4e6, 0.70), "q8": (8e6, 0.72), "fp32": (100e6, 0.74)}


def records(method, dataset, bitops, f1, is_single, op="default", noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return [{"method": method, "dataset": dataset, "seed": s, "fold": f, "operating_point": op,
             "f1_macro": float(f1 + noise * rng.standard_normal()), "bitops_per_clip": bitops,
             "weight_bytes": bitops / 10, "is_single": is_single} for s in SEEDS for f in FOLDS]


def singles(dataset):
    out = []
    for i, (name, (b, f1)) in enumerate(SINGLES.items()):
        out += records(name, dataset, b, f1, True, seed=100 + i)
    return out


def write(tmp_path, rows, name="r.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def run(tmp_path, rows):
    return front.main([str(write(tmp_path, rows)), "--out", str(tmp_path / "out")])


def test_comparator_is_the_best_single_at_equal_or_lower_cost(tmp_path):
    # 5e6 bit-ops: q8 (better F1) costs more, so the comparator is q4
    rows = singles("esc") + records("m", "esc", 5e6, 0.73, False, seed=1)
    df = front.load_records([write(tmp_path, rows)])
    comp = front.compare(df, front.points(df))
    assert comp.loc[0, "comparator_method"] == "q4"
    # at exactly q8's cost, q8 qualifies
    rows = singles("esc") + records("m", "esc", 8e6, 0.73, False, seed=1)
    df = front.load_records([write(tmp_path, rows)])
    assert front.compare(df, front.points(df)).loc[0, "comparator_method"] == "q8"


def test_clear_win_on_every_dataset_passes(tmp_path):
    rows = singles("esc") + singles("urban")
    rows += records("m", "esc", 5e6, 0.76, False, seed=1) + records("m", "urban", 5e6, 0.76, False, seed=2)
    v = run(tmp_path, rows)
    assert v["family_size"] == 2
    assert v["points"]["m|default"]["verdict"] == "PASS"
    assert v["methods"] == {"m": "PASS"}
    assert all(c["comparator_method"] == "q4" for c in v["comparisons"])


def test_win_on_one_dataset_only_fails_overall(tmp_path):
    rows = singles("esc") + singles("urban")
    rows += records("m", "esc", 5e6, 0.76, False, seed=1)
    rows += records("m", "urban", 5e6, 0.70, False, seed=2)  # level with q4 on urban
    v = run(tmp_path, rows)
    per = v["points"]["m|default"]["per_dataset"]
    assert per == {"esc": "PASS", "urban": "FAIL"}
    assert v["points"]["m|default"]["verdict"] == "FAIL" and v["methods"] == {"m": "FAIL"}


def test_missing_dataset_fails_overall(tmp_path):
    rows = singles("esc") + singles("urban") + records("m", "esc", 5e6, 0.76, False, seed=1)
    v = run(tmp_path, rows)
    assert v["points"]["m|default"]["per_dataset"] == {"esc": "PASS", "urban": "MISSING"}
    assert v["points"]["m|default"]["verdict"] == "FAIL"


def test_cheaper_than_every_single_is_not_a_pass(tmp_path):
    rows = singles("esc") + records("m", "esc", 0.5e6, 0.80, False, seed=1)
    v = run(tmp_path, rows)
    assert v["comparisons"][0]["verdict"] == "NO_SINGLE_AT_COST"
    assert v["points"]["m|default"]["verdict"] == "FAIL"
    assert v["family_size"] == 0


def test_bonferroni_over_the_family(tmp_path):
    # a small gain that clears 0.05 alone but not after correcting over 20
    rows = singles("esc")
    rows += records("m0", "esc", 5e6, 0.706, False, seed=1)
    df = front.load_records([write(tmp_path, rows)])
    alone = front.compare(df, front.points(df)).iloc[0]
    assert alone["family_size"] == 1 and alone["p_ttest"] < 0.05 and alone["verdict"] == "PASS"
    for k in range(1, 20):
        rows += records(f"m{k}", "esc", 5e6, 0.70, False, seed=10 + k)
    df = front.load_records([write(tmp_path, rows)])
    comp = front.compare(df, front.points(df)).set_index("method")
    assert comp.loc["m0", "family_size"] == 20
    assert comp.loc["m0", "p_ttest_bonferroni"] == pytest.approx(min(1.0, 20 * alone["p_ttest"]))
    assert comp.loc["m0", "p_ttest_bonferroni"] >= 0.05 and comp.loc["m0", "verdict"] == "FAIL"


def test_pairs_on_seed_and_fold():
    d = pd.DataFrame(records("q4", "esc", 4e6, 0.70, True, seed=3) + records("m", "esc", 5e6, 0.0, False))
    # the method is q4 plus exactly 0.01 on every (seed, fold), shuffled order
    q4 = d[d.method == "q4"].set_index(["seed", "fold"])["f1_macro"]
    m = d.method == "m"
    d.loc[m, "f1_macro"] = [q4[(s, f)] + 0.01 for s, f in zip(d.seed[m], d.fold[m])]
    d = d.sample(frac=1, random_state=0)
    comp = front.compare(d, front.points(d)).iloc[0]
    assert comp["n_pairs"] == 25 and comp["mean_diff"] == pytest.approx(0.01)


def test_front_marks_dominated_points(tmp_path):
    rows = singles("esc") + records("m", "esc", 5e6, 0.65, False, seed=1)  # beaten by q4
    df = front.load_records([write(tmp_path, rows)])
    pts = front.points(df).set_index("method")
    assert pts["on_front"].to_dict() == {"fp32": True, "m": False, "q1": True, "q4": True, "q8": True}


def test_operating_points_are_judged_separately(tmp_path):
    rows = singles("esc") + singles("urban")
    for ds, s in (("esc", 1), ("urban", 2)):
        rows += records("m", ds, 5e6, 0.76, False, op="tau0.5", seed=s)
        rows += records("m", ds, 5e6, 0.69, False, op="tau0.9", seed=s + 10)
    v = run(tmp_path, rows)
    assert v["points"]["m|tau0.5"]["verdict"] == "PASS" and v["points"]["m|tau0.9"]["verdict"] == "FAIL"
    assert v["methods"] == {"m": "PASS"}
    assert (tmp_path / "out" / "front.csv").exists() and (tmp_path / "out" / "comparisons.csv").exists()


def test_bad_records_are_rejected(tmp_path):
    rows = singles("esc")
    with pytest.raises(ValueError):
        front.load_records([write(tmp_path, rows + rows[:1])])
    bad = dict(rows[0], is_single="yes")
    with pytest.raises(ValueError):
        front.load_records([write(tmp_path, [bad])])
    with pytest.raises(ValueError):
        front.load_records([write(tmp_path, [{k: v for k, v in rows[0].items() if k != "weight_bytes"}])])
    with pytest.raises(ValueError):
        front.load_records([write(tmp_path, rows + [dict(rows[0], is_single=False, operating_point="x")])])
