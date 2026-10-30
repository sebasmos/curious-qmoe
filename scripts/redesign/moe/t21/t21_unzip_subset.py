"""Extract only the listed clips (CSV column `name`) from one (merged) FSD50K audio zip; never runs archive content.

    python -I t21_unzip_subset.py ZIP CSV OUT_DIR
"""
import csv
import shutil
import sys
import zipfile
from pathlib import Path

zp, cp, od = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
od.mkdir(parents=True, exist_ok=True)
want = {r["name"] for r in csv.DictReader(cp.open())}
got = 0
with zipfile.ZipFile(zp) as z:
    for m in z.infolist():
        b = Path(m.filename).name
        if m.is_dir() or b not in want:
            continue
        dst = od / b
        if not (dst.exists() and dst.stat().st_size == m.file_size):
            with z.open(m) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
        got += 1
missing = len(want) - got
print(f"{zp.name}: {got} of {len(want)} listed clips extracted to {od}, {missing} not in this archive", flush=True)
