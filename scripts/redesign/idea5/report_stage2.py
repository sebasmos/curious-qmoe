"""Run scripts/redesign/front.py on every records family and condition of an
analyze_stage2.py output, and write markdown tables.

Per analysis directory (one dataset):
  front/<family>__<condition>/   front.csv, comparisons.csv, verdict.json
  tables.md                      static front, cascade verdicts, humility
                                 curve, calibration, nested versus direct

    python scripts/redesign/idea5/report_stage2.py ANALYSIS/esc5 [ANALYSIS/urban ...]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.redesign import front  # noqa: E402

KIND_ORDER = {"clean": 0, "bg": 1, "pink": 2, "white": 3, "field": 4}


def cond_key(c):
    m = re.match(r"([a-z]+)(?:__(\d+)db)?$", c)
    return (KIND_ORDER.get(m.group(1), 9), -int(m.group(2)) if m.group(2) else 0)


def cond_label(c):
    m = re.match(r"([a-z]+)(?:__(\d+)db)?$", c)
    return m.group(1) if not m.group(2) else f"{m.group(1)}@{m.group(2)}"


def run_fronts(adir: Path) -> pd.DataFrame:
    rows = []
    for rec in sorted((adir / "records").glob("*.jsonl")):
        fam, cond = rec.stem.split("__", 1)
        out = adir / "front" / rec.stem
        with contextlib.redirect_stdout(io.StringIO()):
            front.main([str(rec), "--out", str(out)])
        comp = pd.read_csv(out / "comparisons.csv") if (out / "comparisons.csv").stat().st_size > 1 else pd.DataFrame()
        if len(comp):
            comp["family"], comp["condition"] = fam, cond
            rows.append(comp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def static_table(adir: Path, cond: str) -> str:
    S = pd.read_csv(adir / "static.csv")
    S = S[S["condition"] == cond]
    g = S.groupby(["config", "head"]).agg(f1=("f1", "mean"), sd=("f1", "std"), n=("f1", "size"),
                                         bitops=("bitops", "mean"), read=("bits_read", "mean"),
                                         store=("storage_bytes", "mean"), ece=("ece", "mean")).reset_index()
    fr = adir / "front" / f"primary__{cond}" / "front.csv"
    on = set()
    if fr.exists():
        F = pd.read_csv(fr)
        on = set(F[F["on_front"] & F["is_single"]]["method"])
    g["front"] = [("yes" if f"{c}+{h}" in on else "") for c, h in zip(g["config"], g["head"])]
    g = g.sort_values("bitops")
    out = pd.DataFrame({"config": g["config"], "head": g["head"], "F1": g["f1"], "sd": g["sd"], "pairs": g["n"],
                        "bit-ops (T)": g["bitops"] / 1e12, "bits read (Mb)": g["read"] / 1e6,
                        "storage (MB)": g["store"] / 1e6, "ECE": g["ece"], "on front": g["front"]})
    return md(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("analysis", nargs="+", type=Path)
    args = ap.parse_args(argv)
    for adir in args.analysis:
        comp = run_fronts(adir)
        comp.to_csv(adir / "front_comparisons.csv", index=False)
        C = pd.read_csv(adir / "cascades.csv")
        rate = (C.groupby(["condition", "cheap", "expensive", "head", "signal", "point"])
                .agg(rate=("rate", "mean"), iv_rate=("iv_rate", "mean"), read=("bits_read", "mean"),
                     store=("storage_bytes", "mean"), ece=("ece", "mean"),
                     reached=("match_reached", lambda s: float(s.dropna().astype(bool).mean()) if s.notna().any()
                              else float("nan")))
                .reset_index())
        conds = sorted(pd.read_csv(adir / "static.csv")["condition"].unique(), key=cond_key)
        parts = [f"# {adir.name}: idea 5 Stage 2 tables", ""]
        # verdict summary
        if len(comp):
            prim = comp[comp["family"] == "primary"]
            parts += ["## Primary cascade points that PASS front.py's bar", "",
                      "FP32 head, margin gate, thresholds from each fold's inner validation split (same condition). "
                      "Family per condition: every primary point; Bonferroni over it.", ""]
            ok = prim[prim["verdict"] == "PASS"].copy()
            if len(ok):
                ok = ok.merge(rate, left_on=["condition", "operating_point"], right_on=["condition", "point"],
                              how="left", suffixes=("", "_r"))
                ok = ok[[m.startswith(f"{c}>{e}+{h}|{s}") for m, c, e, h, s in
                         zip(ok["method"], ok["cheap"], ok["expensive"], ok["head"], ok["signal"])]]
                t = pd.DataFrame({"condition": ok["condition"].map(cond_label), "cascade": ok["method"],
                                  "point": ok["operating_point"], "F1": ok["f1_macro"],
                                  "bit-ops (T)": ok["bitops_per_clip"] / 1e12, "escalated": ok["rate"],
                                  "comparator": ok["comparator_method"], "comparator F1": ok["comparator_f1_macro"],
                                  "diff (pts)": 100 * ok["mean_diff"], "p t (Bonf.)": ok["p_ttest_bonferroni"],
                                  "p W (Bonf.)": ok["p_wilcoxon_bonferroni"]})
                parts += [md(t.sort_values(["condition", "bit-ops (T)"]), "{:.4g}"), ""]
            else:
                parts += ["None.", ""]
            parts += ["## Verdict counts per family and condition", ""]
            vc = comp.groupby(["family", "condition", "verdict"]).size().unstack(fill_value=0).reset_index()
            vc["condition"] = vc["condition"].map(cond_label)
            parts += [md(vc), ""]
        for cond in conds:
            parts += [f"## {cond_label(cond)}", "", "### Static systems", "", static_table(adir, cond), ""]
            sub = comp[(comp["condition"] == cond) & (comp["family"] == "primary")] if len(comp) else comp
            if len(sub):
                r = rate[(rate["condition"] == cond) & (rate["head"] == "esc") & (rate["signal"] == "margin")]
                sub = sub.copy()
                sub["key"] = sub["method"] + "@" + sub["operating_point"]
                r = r.assign(key=r["cheap"] + ">" + r["expensive"] + "+esc|margin@" + r["point"])
                sub = sub.merge(r[["key", "rate", "read", "ece", "reached"]], on="key", how="left")
                t = pd.DataFrame({"cascade": sub["method"].str.replace("+esc|margin", "", regex=False),
                                  "point": sub["operating_point"], "escalated": sub["rate"], "F1": sub["f1_macro"],
                                  "bit-ops (T)": sub["bitops_per_clip"] / 1e12, "bits read (Mb)": sub["read"] / 1e6,
                                  "ECE": sub["ece"], "comparator": sub.get("comparator_method"),
                                  "diff (pts)": 100 * sub.get("mean_diff", float("nan")),
                                  "verdict": sub["verdict"]})
                parts += ["### Primary cascade points (FP32 head, margin)", "", md(t, "{:.4g}"), ""]
        # humility curve
        hum = pd.read_csv(adir / "humility.csv") if (adir / "humility.csv").exists() else pd.DataFrame()
        if len(hum):
            hum = hum[(hum["head"] == "esc") & (hum["signal"] == "margin") & (hum["expensive"] != "fp32")]
            hum["cascade"] = hum["cheap"] + ">" + hum["expensive"] + "@" + hum["point"]
            pv = hum.pivot_table(index="cascade", columns="condition", values="rate")
            pv = pv[sorted(pv.columns, key=cond_key)]
            pv.columns = [cond_label(c) for c in pv.columns]
            parts += ["## Humility curve: test escalation rate under thresholds set on clean inner validation", "",
                      md(pv.reset_index()), ""]
        cal = pd.read_csv(adir / "calibration.csv")
        cal = cal[cal["head"] == "esc"]
        pv = cal.pivot_table(index="config", columns="condition", values="ece")
        pv = pv[sorted(pv.columns, key=cond_key)]
        pv.columns = [cond_label(c) for c in pv.columns]
        parts += ["## Calibration: top-label ECE (15 bins) of static systems, FP32 head", "", md(pv.reset_index()), ""]
        if "conf_gap" in cal:
            pv = cal.pivot_table(index="config", columns="condition", values="conf_gap")
            pv = pv[sorted(pv.columns, key=cond_key)]
            pv.columns = [cond_label(c) for c in pv.columns]
            parts += ["Mean confidence minus accuracy (negative: underconfident), FP32 head", "", md(pv.reset_index()), ""]
        (adir / "tables.md").write_text("\n".join(parts))
        print(f"wrote {adir / 'tables.md'}")


if __name__ == "__main__":
    main()
