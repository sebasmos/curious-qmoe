"""T26 Part C extraction: T25's CNN extractor (t25_extract_cnn.py: stage 3c loaders, simulated W8A8, T21's reader,
length rule, mixer, field cells, sidecars) with T24's reader for the prepared 16 kHz audio (t24_common.patch_loader),
on T24's all.csv and eval.csv. Arguments are T25's.   python t26_extract.py --csv ... --models mn20 ..."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t25"))
sys.path.insert(0, str(HERE.parent / "t24"))
import t25_extract_cnn as X  # noqa: E402  (applies T21's loader patch at import)
import t24_common  # noqa: E402

t24_common.patch_loader()  # after T21's own patch, as t24_extract.py does
X.ARGS = X.parse_args(sys.argv[1:])
sys.exit(X.main())
