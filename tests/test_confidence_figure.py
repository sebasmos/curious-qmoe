"""Figure 2 generator on a tiny synthetic input in the routing_analysis.py schema.

Run: PYTHONPATH=. pytest tests/test_confidence_figure.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis import generate_confidence_figure as fig2

NAMES = ["BitNet", "Q4", "Q8"]


def write_routing(out_dir, alpha, counts, q8_shift, seed=0):
    """routing_results.json as routing_analysis.py writes it (fields used here
    plus the rest of a record), with Q8's confidence shifted down by q8_shift."""
    rng = np.random.default_rng(seed)
    recs = []
    for e, n in enumerate(counts):
        conf = np.clip(1 - rng.beta(0.8, 4.0, n) * 0.8 - (q8_shift if e == 2 else 0), 0.2, 1)
        for c in conf:
            recs.append({"idx": len(recs), "expert": e, "expert_name": NAMES[e],
                         "confidence": float(c), "uncertainty": 0.01, "label": 0,
                         "pred": 0, "fold": 1})
    d = Path(out_dir) / f"routing_a{alpha}"
    d.mkdir(parents=True)
    (d / "routing_results.json").write_text(json.dumps({
        "config": {"run_dir": f"/runs/outputs/esc_pprior_a{alpha}_s0/moe", "folds": [1],
                   "expert_names": NAMES, "num_samples": len(recs)},
        "routing": recs, "summaries": {}}))
    return d


def test_two_panel_figure_and_computed_test(tmp_path):
    a03 = write_routing(tmp_path, "0.3", (12, 70, 18), q8_shift=0.25)
    a10 = write_routing(tmp_path, "1.0", (5, 0, 60), q8_shift=0.0, seed=1)
    out = tmp_path / "fig.pdf"
    fig2.main(["--routing-json", str(a03), str(a10 / "routing_results.json"),
               "--output", str(out)])
    assert out.read_bytes()[:4] == b"%PDF"

    p03 = fig2.load_routing(a03)
    assert p03["title"] == "α = 0.3"
    assert [len(v) for v in p03["conf"]] == [12, 70, 18] and p03["total"] == 100
    t = fig2.compare_top(p03)
    rest = np.concatenate(p03["conf"][:2])
    assert t["p"] == stats.mannwhitneyu(p03["conf"][2], rest, alternative="two-sided").pvalue
    assert t["median_top"] < t["median_rest"] and t["p"] < 0.05

    p10 = fig2.load_routing(a10)            # an expert with no samples keeps its row
    assert p10["names"] == NAMES and len(p10["conf"][1]) == 0
