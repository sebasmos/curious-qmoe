"""T26 head-training CSVs (T24's t24_make_csvs.py, fixed-test layout) for the CNNs extracted in T26.
    python t26_make_csvs.py --r R26 --d d3 --models mn20"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "t24"))
import t24_make_csvs as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--r", type=Path, required=True)
ap.add_argument("--d", required=True)
ap.add_argument("--models", nargs="+", required=True)
a = ap.parse_args()
M.TIERS = {m: ("cnn_all", f"{m}_fp32__clean.npy") for m in a.models}
sys.argv = [sys.argv[0], "--r", str(a.r), "--d", a.d, "--tiers", *a.models]
M.main()
