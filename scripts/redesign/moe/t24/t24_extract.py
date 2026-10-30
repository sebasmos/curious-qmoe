"""T24 extraction: T21's extractors unchanged (t21_extract_ast.py on MPS, t21_extract_mn.py on the CPU), with the
T24 reader for the prepared audio (t24_common). Arguments after the first are passed to T21's parser.

    python t24_extract.py ast|mn --csv ... (T21 arguments)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "t21"))
kind, argv = sys.argv[1], sys.argv[2:]
import t24_common  # noqa: E402

if kind == "ast":
    import t21_extract_ast as X  # noqa: E402
elif kind == "mn":
    import t21_extract_mn as X  # noqa: E402
else:
    raise SystemExit("first argument: ast or mn")
t24_common.patch_loader()  # after T21's own patch at import
X.ARGS = X.parse_args(argv)
sys.exit(X.main())
