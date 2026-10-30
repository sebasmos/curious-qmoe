"""T19 cost accounting (QWave/costs.py ast_t19_bitops) against the audit's totals and the stage 2 rule.

Run: PYTHONPATH=. pytest tests/test_t19_costs.py -q
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2"))

from QWave.costs import ast_t19_bitops, ast_tokens  # noqa: E402


def test_audit_totals_at_1024():
    w8 = ast_t19_bitops(1024, 8, 8, 16, group=None, asym=False)
    w4 = ast_t19_bitops(1024, 4, 8, 16, group=None, asym=False)
    w3 = ast_t19_bitops(1024, 3, 8, 16, group=None, asym=False)
    assert ast_tokens(1024) == (1212, 1214)
    assert abs(w8["unpriced"] / 1e12 - 9.45) < 0.005
    assert abs(w4["unpriced"] / 1e12 - 6.15) < 0.005
    assert abs(w3["unpriced"] / 1e12 - 5.33) < 0.005
    assert abs(w8["linear"] / 64 / 1e9 - 103.1) < 0.05
    assert abs(w8["qk"] / 64 / 1e9 - 13.6) < 0.05


def test_matches_stage2_rule_with_rescale():
    from analyze import ast_bitops_tok, short_tokens  # stage 2 helper (legacy configs: A8, P16)
    for L in (400, 512, 1024):
        P, T = short_tokens(L)
        for cfg, wb, g in (("w8g64", 8, 64), ("w4g128", 4, 128), ("n4", 4, None), ("w3g64", 3, 64)):
            mine = ast_t19_bitops(L, wb, 8, 16, group=g, asym=False)
            assert abs(mine["total"] - ast_bitops_tok(cfg, 1024.0, T, P)) < 1e-6 * mine["total"]
            assert abs(mine["unpriced"] - ast_bitops_tok(cfg, 0.0, T, P)) < 1e-6 * mine["total"]


def test_mixed_blocks_interpolate():
    a4 = ast_t19_bitops(512, 4, 4, 8)
    a8 = ast_t19_bitops(512, 4, 8, 8)
    mp = ast_t19_bitops(512, 4, [4] * 6 + [8] * 6, 8)
    assert abs(mp["total"] - (a4["total"] + a8["total"]) / 2) < 1e-6 * mp["total"]
