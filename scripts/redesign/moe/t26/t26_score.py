"""T26 Part C scores: T24's t24_score.py unchanged (fixed-test design, both scalers, every condition, clean and field
inner-validation logits, every fold model must reproduce its summary.json F1 first) with the tier table holding the
CNNs extracted in T26.   python t26_score.py --r R26 --d d3 --seeds 0 1 2 --models mn20"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "t24"))
import t24_score as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--r", type=Path, required=True)
ap.add_argument("--d", required=True)
ap.add_argument("--seeds", nargs="+", required=True)
ap.add_argument("--models", nargs="+", required=True)
a = ap.parse_args()
S.TIERS = {f"{m}_w8a8": (m, "cnn_all", "cnn_eval", f"{m}_w8a8__{{}}.npy") for m in a.models}
sys.argv = [sys.argv[0], "--r", str(a.r), "--d", a.d, "--tiers", *S.TIERS, "--seeds", *a.seeds]
S.main()
