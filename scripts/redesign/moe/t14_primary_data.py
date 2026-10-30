"""T14 stored singles loader (stage 3c static.csv plus final-round non-additive cells for esc50)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t14_pair import S3C, FINAL
FMAP = {"mn20:esc": "cnn:mn20:w8a8:esc", "mn20:8": "cnn:mn20:w8a8:8", "mn20:4": "cnn:mn20:w8a8:4", "mn20:bitnet": "cnn:mn20:w8a8:bitnet", "mn10:esc": "cnn:mn10:w8a8:esc",
        "mn10:8": "cnn:mn10:w8a8:8", "mn10:4": "cnn:mn10:w8a8:4", "mn10:bitnet": "cnn:mn10:w8a8:bitnet", "ast:esc": "ast:w4g128@512+esc", "ast:4": "ast:w4g128@512+4"}
_cache = {}
def static(task):
    if task not in _cache:
        S_ = pd.read_csv(S3C / f"analysis/{task}/static.csv"); S_ = S_[S_.set.isin(["stage3b", "new"])]
        S_ = S_.rename(columns={"condition": "cell"})[["system", "cell", "seed", "fold", "f1", "cost"]]
        if task == "esc50":
            F = pd.read_csv(FINAL / "analysis/b_esc50/static.csv").rename(columns={"condition": "cell"}); F = F[F.cell != "clean"]; F = F[F.system.isin(FMAP)]
            F["system"] = F.system.map(FMAP); S_ = pd.concat([S_, F[["system", "cell", "seed", "fold", "f1", "cost"]]])
        S_["f1"] = 100 * S_.f1; _cache[task] = S_
    return _cache[task]
def single(task, name, cells=None):
    g = static(task); g = g[g.system == name]
    return g[g.cell.isin(cells)] if cells else g
