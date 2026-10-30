#!/usr/bin/env python3
"""Accuracy and cost front, and the redesign's success bar, from per-(seed,
fold) records.

Input: JSON lines, one record per (dataset, method, operating point, seed,
fold):
  {"method", "dataset", "seed", "fold", "f1_macro", "bitops_per_clip",
   "weight_bytes", "operating_point", "is_single"}
A point is one (dataset, method, operating_point); its macro F1, bit-ops per
clip and weight bytes are means over its (seed, fold) records. Single models
(is_single true) are the baselines; every other point is a method point.
Operating points are compared by name across datasets, so a rule that picks
an operating point (a threshold set on inner validation, say) keeps one name.

Front: per dataset, the points that no other point matches or beats on both
axes (bit-ops no higher and F1 no lower, one of them strictly).

Success bar, for every method point:
  comparator  the single point of the same dataset with the highest mean F1
              among those whose mean bit-ops is at most the method's
  test        method minus comparator over the (seed, fold) pairs both have:
              mean difference, two-sided paired t-test and Wilcoxon
              signed-rank (scripts/collect_paper_numbers.paired_test)
  family      every comparison that has a comparator; both p values are
              Bonferroni corrected over it (times its size, capped at 1)
  PASS        mean difference > 0 and both corrected p values < alpha;
              otherwise FAIL, or NO_SINGLE_AT_COST when no single model
              costs as little (not a pass: there is nothing it beat)
Overall, a (method, operating point) passes only when it passes on every
dataset in the input; a dataset where it has no records is a FAIL. A method
passes when one of its operating points passes overall.

Writes front.csv, comparisons.csv and verdict.json to --out.

    python scripts/redesign/front.py records.jsonl [more.jsonl ...] --out DIR [--alpha 0.05]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.collect_paper_numbers import paired_test  # noqa: E402

FIELDS = ["method", "dataset", "seed", "fold", "f1_macro", "bitops_per_clip", "weight_bytes",
          "operating_point", "is_single"]
POINT = ["dataset", "method", "operating_point"]
REL_TOL = 1e-9  # "equal bit-ops" up to float rounding


def load_records(paths) -> pd.DataFrame:
    """Every record of the JSON-lines files, checked: all fields present,
    is_single a bool, a method never both single and not, and no (point,
    seed, fold) twice."""
    rows = []
    for path in paths:
        for i, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            missing = [f for f in FIELDS if f not in r]
            if missing:
                raise ValueError(f"{path}:{i} has no {missing}")
            if not isinstance(r["is_single"], bool):
                raise ValueError(f"{path}:{i} is_single must be true or false")
            rows.append({f: r[f] for f in FIELDS})
    df = pd.DataFrame(rows, columns=FIELDS)
    df["operating_point"] = df["operating_point"].astype(str)
    dup = df.duplicated(POINT + ["seed", "fold"], keep=False)
    if dup.any():
        raise ValueError(f"duplicate (point, seed, fold) records:\n{df[dup][POINT + ['seed', 'fold']]}")
    mixed = df.groupby("method")["is_single"].nunique()
    if (mixed > 1).any():
        raise ValueError(f"methods marked both single and not: {list(mixed[mixed > 1].index)}")
    return df


def points(df: pd.DataFrame) -> pd.DataFrame:
    """One row per point: means over its (seed, fold) records, and on_front."""
    g = df.groupby(POINT, sort=True)
    pts = g.agg(is_single=("is_single", "first"), n=("f1_macro", "size"),
                f1_macro_mean=("f1_macro", "mean"), f1_macro_std=("f1_macro", "std"),
                bitops_per_clip_mean=("bitops_per_clip", "mean"),
                weight_bytes_mean=("weight_bytes", "mean")).reset_index()
    pts["is_single"] = pts["is_single"].astype(bool)
    pts["on_front"] = False
    for ds, sub in pts.groupby("dataset"):
        b, f = sub["bitops_per_clip_mean"].to_numpy(), sub["f1_macro_mean"].to_numpy()
        for i, idx in enumerate(sub.index):
            dominated = (b <= b[i]) & (f >= f[i]) & ((b < b[i]) | (f > f[i]))
            pts.loc[idx, "on_front"] = not dominated.any()
    return pts


def cheaper_or_equal(bitops, limit) -> np.ndarray:
    return np.asarray(bitops) <= limit * (1 + REL_TOL)


def compare(df: pd.DataFrame, pts: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """One row per method point: its comparator, the paired tests with
    Bonferroni correction over the family, and the verdict."""
    rec = {k: dict(zip(zip(g["seed"], g["fold"]), g["f1_macro"])) for k, g in df.groupby(POINT)}
    rows = []
    for _, p in pts[~pts["is_single"]].iterrows():
        row = {k: p[k] for k in POINT}
        row.update(bitops_per_clip=p["bitops_per_clip_mean"], f1_macro=p["f1_macro_mean"])
        singles = pts[(pts["dataset"] == p["dataset"]) & pts["is_single"]]
        ok = singles[cheaper_or_equal(singles["bitops_per_clip_mean"], p["bitops_per_clip_mean"])]
        if ok.empty:
            row["verdict"] = "NO_SINGLE_AT_COST"
            rows.append(row)
            continue
        # highest F1; ties go to the cheaper single
        c = ok.sort_values(["f1_macro_mean", "bitops_per_clip_mean"], ascending=[False, True]).iloc[0]
        row.update(comparator_method=c["method"], comparator_operating_point=c["operating_point"],
                   comparator_bitops_per_clip=c["bitops_per_clip_mean"], comparator_f1_macro=c["f1_macro_mean"])
        res = paired_test(rec[tuple(p[k] for k in POINT)], rec[tuple(c[k] for k in POINT)])
        if res is None:
            row.update(n_pairs=0, verdict="FAIL")
        else:
            row.update(n_pairs=res["n_pairs"], mean_diff=res["mean_diff"], t=res["t"],
                       p_ttest=res["p_ttest"], p_wilcoxon=res["p_wilcoxon"])
        rows.append(row)
    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp
    tested = comp["p_ttest"].notna() if "p_ttest" in comp else pd.Series(False, index=comp.index)
    m = int(tested.sum())
    comp["family_size"] = m
    for key in ("p_ttest", "p_wilcoxon"):
        if key in comp:
            comp[f"{key}_bonferroni"] = (comp[key] * m).clip(upper=1.0)
    passed = tested & (comp.get("mean_diff", 0) > 0) & (comp.get("p_ttest_bonferroni", 1) < alpha) \
        & (comp.get("p_wilcoxon_bonferroni", 1) < alpha)
    comp.loc[tested, "verdict"] = np.where(passed[tested], "PASS", "FAIL")
    return comp


def overall(comp: pd.DataFrame, datasets) -> dict:
    """PASS for a (method, operating point) only when it passes on every
    dataset; a missing dataset is a FAIL."""
    out = {}
    if comp.empty:
        return out
    for (method, op), g in comp.groupby(["method", "operating_point"]):
        per = {ds: "MISSING" for ds in datasets}
        per.update(dict(zip(g["dataset"], g["verdict"])))
        out[f"{method}|{op}"] = {"method": method, "operating_point": op, "per_dataset": per,
                                 "verdict": "PASS" if all(v == "PASS" for v in per.values()) else "FAIL"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", help="JSON-lines files of per-(seed, fold) records")
    ap.add_argument("--out", required=True, help="writes front.csv, comparisons.csv, verdict.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    df = load_records(args.records)
    datasets = sorted(df["dataset"].unique())
    pts = points(df)
    comp = compare(df, pts, args.alpha)
    per_point = overall(comp, datasets)
    methods = {}
    for v in per_point.values():
        methods[v["method"]] = "PASS" if v["verdict"] == "PASS" or methods.get(v["method"]) == "PASS" else "FAIL"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pts.to_csv(out / "front.csv", index=False)
    comp.to_csv(out / "comparisons.csv", index=False)
    verdict = {"inputs": [str(Path(p).resolve()) for p in args.records], "alpha": args.alpha,
               "datasets": datasets, "family_size": int(comp["family_size"].iloc[0]) if len(comp) else 0,
               "rule": "PASS = mean paired F1 difference > 0 and Bonferroni-corrected paired t-test and "
                       "Wilcoxon p < alpha against the best single model with mean bit-ops <= the method's, "
                       "on every dataset",
               "methods": methods, "points": per_point,
               "comparisons": json.loads(comp.to_json(orient="records"))}
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))

    cols = [c for c in ("dataset", "method", "operating_point", "bitops_per_clip", "f1_macro", "comparator_method",
                        "comparator_f1_macro", "n_pairs", "mean_diff", "p_ttest_bonferroni",
                        "p_wilcoxon_bonferroni", "verdict") if c in comp]
    if len(comp):
        print(comp[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    for name, v in per_point.items():
        print(f"{name}: {v['verdict']}  {v['per_dataset']}")
    print(f"front, comparisons and verdict in {out}")
    return verdict


if __name__ == "__main__":
    main()
