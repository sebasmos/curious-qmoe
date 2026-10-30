"""Tables from the output of gate_by_precision.py (fold-level means, paired differences, gate winners).

    python make_report.py --out OUT
Writes tables_means.md, headline_backbone.md, headline_head.md, gate_winners.md/.csv, worst_cells.md, verdict_numbers.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_by_precision import ESC_CONDS, TAU_CONDS, GATES, paired  # noqa: E402

TASK_NAME = {"esc50": "ESC-50, 50 classes", "esc5": "ESC-50, five groups", "urban": "UrbanSound8K", "tau": "TAU 2020 Mobile"}
BB_ORDER = ["FP32", "Q8sim", "INT8", "Q8", "Q6", "Q5", "Q4", "Q4pc", "Q4n", "Q3", "Q3ownhead"]
HEAD_ORDER = ["esc", "8", "4", "bitnet"]
HEAD_NAME = {"esc": "FP32 head", "8": "Q8 head", "4": "Q4 head", "bitnet": "BitNet head"}
TABLE_METRICS = [("f1", "macro F1"), ("ece", "ECE"), ("ece_T", "ECE after temperature scaling"),
                 ("auroc_margin", "margin gate AUROC (model wrong)"), ("caught10", "share of errors caught at 10% escalation"),
                 ("caught20", "share of errors caught at 20% escalation"), ("caught30", "share of errors caught at 30% escalation"),
                 ("rate_b20", "escalation rate, threshold set for 20% on clean inner validation"),
                 ("auroc_entropy", "entropy gate AUROC"), ("auroc_energy", "energy (log-sum-exp) gate AUROC"),
                 ("auroc_tsmax", "temperature-scaled max-probability gate AUROC")]


def conds_of(task):
    return (TAU_CONDS if task == "tau" else ESC_CONDS) + ["ALL"]


def label(r):
    return f"{r.model} {r.backbone}" + (f" [{r.head} head]" if r.head not in ("esc",) else "")


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                     else ("" if isinstance(v, (float, np.floating)) else str(v)) for v in r) + " |")
    return "\n".join(out)


def sort_key(vals, order):
    return [order.index(v) if v in order else len(order) for v in vals]


def fmt_diff(r):
    if r is None or not np.isfinite(r["diff"]):
        return "n/a"
    star = "*" if r["decided"] else ""
    return f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    out = a.out
    sysdf = pd.read_csv(out / "systems.csv").set_index("sid")
    f = pd.read_csv(out / "fold_metrics_with_all.csv.gz")
    st = pd.read_csv(out / "paired_vs_fp32.csv")
    rises = pd.read_csv(out / "fold_rises.csv")
    fm = f.groupby(["sid", "cond"], as_index=False).mean(numeric_only=True)  # fold means
    fm = fm.merge(sysdf.reset_index()[["sid", "task", "model", "backbone", "head", "length", "fam"]], on="sid")
    nfold = f.groupby("sid").fold.nunique()

    # 1. means tables
    lines = ["# Fold means of every cell\n",
             "Each value is the mean over folds of the seed-averaged fold value. Conditions: clean, noise kind (bg, pink, white) "
             "followed by SNR in dB, field mixture; ALL is the mean over clean and the nine cells at 10, 5, 0 dB (TAU: the nine devices, "
             "clean = device A). Empty cell: the score file does not exist (UrbanSound8K AST has no 20 dB files).\n"]
    for task in ["esc50", "esc5", "urban", "tau"]:
        g = fm[fm.task == task]
        lines.append(f"\n## {TASK_NAME[task]}\n")
        cs = conds_of(task)
        # backbone panels: model, head fixed
        for (model, head, length), gg in g.groupby(["model", "head", "length"], sort=False):
            if gg.backbone.nunique() < 2:
                continue
            lines.append(f"\n### Backbone precision, {model} ({length}), {HEAD_NAME[head]}\n")
            for m, mn in TABLE_METRICS:
                piv = gg.pivot_table(index="backbone", columns="cond", values=m)
                piv = piv.reindex(index=sorted(piv.index, key=lambda v: BB_ORDER.index(v) if v in BB_ORDER else 99),
                                  columns=[c for c in cs if c in piv.columns])
                lines.append(f"\n**{mn}**\n\n" + md_table(piv.reset_index().rename(columns={"backbone": "backbone"})))
        # head panels
        for (model, bb, length), gg in g.groupby(["model", "backbone", "length"], sort=False):
            if gg["head"].nunique() < 2:
                continue
            lines.append(f"\n### Head precision, {model} {bb} ({length})\n")
            for m, mn in TABLE_METRICS[:5] + TABLE_METRICS[7:8] + TABLE_METRICS[8:]:
                piv = gg.pivot_table(index="head", columns="cond", values=m)
                piv = piv.reindex(index=[h for h in HEAD_ORDER if h in piv.index], columns=[c for c in cs if c in piv.columns])
                piv.index = [HEAD_NAME[h] for h in piv.index]
                lines.append(f"\n**{mn}**\n\n" + md_table(piv.reset_index().rename(columns={"index": "head"})))
    (out / "tables_means.md").write_text("\n".join(lines) + "\n")

    # 2. headline tables: paired differences against FP32 (backbone) and against the FP32 head (head)
    sidinfo = sysdf
    stx = st.merge(sysdf.reset_index()[["sid", "task", "model", "backbone", "head", "length"]], on="sid")
    get = {(r.panel, r.sid, r.cond, r.metric): r for r in stx.itertuples()}

    def cell(panel, sid, cond, metric):
        r = get.get((panel, sid, cond, metric))
        return None if r is None else r._asdict()

    def headline(panel, name):
        L = [f"# {name}\n",
             "Paired difference against the reference (backbone panel: the FP32 backbone with the same head; head panel: the FP32 head "
             "on the same backbone), over folds (ESC-50: 5, UrbanSound8K: 10, TAU: 5; seeds averaged within folds), with the 95% t-interval "
             "in brackets. A star marks an interval that excludes zero. No correction for the number of cells. "
             "AUROC, F1, caught share and rates are on a 0 to 1 scale.\n"]
        for task in ["esc50", "esc5", "urban", "tau"]:
            sub = stx[(stx.panel == panel) & (stx.task == task)]
            if sub.empty:
                continue
            L.append(f"\n## {TASK_NAME[task]}\n")
            rows = []
            for sid in sub.sid.unique():
                s = sysdf.loc[sid]
                hd_ = s["head"]
                rec = {"system": f"{s.model} {s.backbone} [{hd_}] ({s.length})", "_k": (s.model, s.length, s["head"], BB_ORDER.index(s.backbone) if s.backbone in BB_ORDER else 99)}
                refsid = sub[sub.sid == sid].ref.iloc[0]
                rec["vs"] = refsid.split("|", 2)[2]
                conds = ["clean", "field", "white0", "ALL"] if task != "tau" else ["clean", "shift", "ALL"]
                for c in conds:
                    for m, short in (("f1", "F1"), ("auroc_margin", "AUROC"), ("ece_T", "ECE_T")):
                        r = cell(panel, sid, c, m)
                        base = r["mean"] if r else np.nan
                        rec[f"{short} {c}"] = (f"{base:.3f} ({fmt_diff(r)})" if r else "")
                r = cell(panel, sid, "ALL", "caught20")
                rec["caught20 ALL"] = f"{r['mean']:.3f} ({fmt_diff(r)})" if r else ""
                rname = "rise_b20_shift" if task == "tau" else "rise_b20_mean0"
                r = cell(panel, sid, "rise", rname)
                if r is None or not np.isfinite(r["diff"]):
                    r = cell(panel, sid, "rise", "rise_b20_bg0")  # UrbanSound8K AST files: only bg has full-length 0 dB rows
                rec["rate b20 rise to 0 dB (clean to shift for TAU)"] = f"{r['mean']:+.3f} ({fmt_diff(r)})" if r else ""
                rows.append(rec)
            d = pd.DataFrame(rows).sort_values("_k").drop(columns="_k")
            L.append(md_table(d))
        return "\n".join(L) + "\n"

    (out / "headline_backbone.md").write_text(headline("backbone", "Backbone precision, head fixed"))
    (out / "headline_head.md").write_text(headline("head", "Head precision, backbone fixed"))

    # 2b. compact cross table: ALL-condition means with paired differences, FP32-head systems (INT8 and own-head rows with their own head)
    cl = ["# Compact cross table\n",
          "Each cell: fold mean over the ALL conditions (see tables_means.md), then the paired difference against the reference in "
          "parentheses with its 95% interval over folds; a star marks an interval that excludes zero. Backbone rows are paired with the "
          "FP32 backbone at the same head; head rows with the FP32 head on the same backbone. rise = escalation-rate rise from clean "
          "to the mean of the three 0 dB conditions (TAU: clean to the pooled other devices; UrbanSound8K AST: clean to bg 0 dB), "
          "threshold set for 20% escalation on clean inner validation.\n"]
    for panel, title in (("backbone", "Backbone precision, head fixed"), ("head", "Head precision, backbone fixed")):
        cl.append(f"\n## {title}\n")
        for task in ["esc50", "esc5", "urban", "tau"]:
            sub = stx[(stx.panel == panel) & (stx.task == task)]
            sids = sub.sid.unique()
            rows = []
            for sid in sids:
                s_ = sysdf.loc[sid]
                hd_, bb_ = s_["head"], s_.backbone
                if panel == "backbone" and hd_ != "esc" and bb_ not in ("INT8", "Q3ownhead"):
                    continue
                if panel == "head" and bb_ not in ("FP32", "Q8sim", "Q8", "Q3"):
                    continue
                rec = {"system": f"{s_.model} {bb_} [{hd_}] ({s_.length})",
                       "_k": (s_.model, s_.length, bb_ if panel == "head" else "", hd_ if panel == "head" else "", BB_ORDER.index(bb_) if bb_ in BB_ORDER else 99)}
                for m, short in (("f1", "F1"), ("ece_T", "ECE_T"), ("auroc_margin", "margin AUROC"), ("caught20", "caught@20%")):
                    r = cell(panel, sid, "ALL", m)
                    rec[short] = f"{r['mean']:.3f} ({fmt_diff(r)})" if r else ""
                rname = "rise_b20_shift" if task == "tau" else "rise_b20_mean0"
                r = cell(panel, sid, "rise", rname)
                if r is None or not np.isfinite(r["diff"]):
                    r = cell(panel, sid, "rise", "rise_b20_bg0")
                rec["rise"] = f"{r['mean']:+.3f} ({fmt_diff(r)})" if r else ""
                rows.append(rec)
            if rows:
                cl.append(f"\n### {TASK_NAME[task]}\n")
                cl.append(md_table(pd.DataFrame(rows).sort_values("_k").drop(columns="_k")))
    (out / "cross_compact.md").write_text("\n".join(cl) + "\n")

    # 3. gate winners per system: AUROC of the four gates averaged over conditions, winner, and condition-wise wins
    rows = []
    for sid, g in fm[fm.cond != "ALL"].groupby("sid"):
        s = sysdf.loc[sid]
        A = g.pivot_table(index="cond", values=[f"auroc_{x}" for x in GATES], aggfunc="mean")
        if A.empty:
            continue
        win = A.idxmax(axis=1).str.replace("auroc_", "").value_counts()
        allv = fm[(fm.sid == sid) & (fm.cond == "ALL")]
        rec = {"task": s.task, "model": s.model, "backbone": s.backbone, "head": s["head"], "length": s.length}
        for x in GATES:
            rec[f"AUROC {x}"] = float(allv[f"auroc_{x}"].iloc[0]) if len(allv) else float(A[f"auroc_{x}"].mean())
        means = {x: rec[f"AUROC {x}"] for x in GATES}
        rec["winner (ALL)"] = max(means, key=means.get)
        rec["margin minus best other"] = means["margin"] - max(v for k, v in means.items() if k != "margin")
        for x in GATES:
            rec[f"wins {x}"] = int(win.get(x, 0))
        rec["n conds"] = len(A)
        rows.append(rec)
    gw = pd.DataFrame(rows)
    gw["_b"] = sort_key(gw.backbone, BB_ORDER)
    gw["_h"] = sort_key(gw["head"], HEAD_ORDER)
    gw = gw.sort_values(["task", "model", "length", "_h", "_b"]).drop(columns=["_b", "_h"])
    gw.to_csv(out / "gate_winners.csv", index=False)
    wl = ["# Gate comparison on the same cells\n", "AUROC for 'the model is wrong' (fold means, ALL = mean over the shared conditions); "
          "'wins' counts conditions in which the gate has the highest fold-mean AUROC.\n"]
    for task in ["esc50", "esc5", "urban", "tau"]:
        wl.append(f"\n## {TASK_NAME[task]}\n")
        wl.append(md_table(gw[gw.task == task].drop(columns="task")))
    # winner share per backbone precision label (pooled over systems)
    wl.append("\n## Winner count by backbone precision (systems whose ALL-condition winner is each gate)\n")
    t = gw.groupby(["backbone", "winner (ALL)"]).size().unstack(fill_value=0).reset_index()
    wl.append(md_table(t))
    wl.append("\n## Gate AUROC minus margin AUROC, by backbone precision (mean over systems; a positive value means the gate beats the margin)\n")
    for x in GATES:
        gw[f"{x} minus margin"] = gw[f"AUROC {x}"] - gw["AUROC margin"]
    t2 = gw.groupby("backbone")[[f"{x} minus margin" for x in GATES if x != "margin"]].mean().reset_index()
    t2["systems"] = gw.groupby("backbone").size().values
    t2["_b"] = sort_key(t2.backbone, BB_ORDER)
    wl.append(md_table(t2.sort_values("_b").drop(columns="_b"), "{:+.4f}"))
    (out / "gate_winners.md").write_text("\n".join(wl) + "\n")

    # 4. worst cells for the margin gate, and the counts that carry the verdict
    d = stx[(stx.metric == "auroc_margin") & (stx.cond.isin(conds_of("esc50") + TAU_CONDS)) & stx["diff"].notna()].copy()
    d["cell"] = d.apply(lambda r: f"{r.task} / {r.model} {r.backbone} [{r['head']}] ({r.length}) / {r.cond}", axis=1)
    bb = d[d.panel == "backbone"].sort_values("diff").head(25)
    hd = d[d.panel == "head"].sort_values("diff").head(15)
    wl = ["# Cells where the margin gate loses the most AUROC against its FP32 reference\n",
          "\n## Backbone precision (head fixed), 25 largest losses\n",
          md_table(bb[["cell", "ref_mean", "mean", "diff", "lo", "hi", "n"]].rename(columns={"ref_mean": "FP32 AUROC", "mean": "AUROC", "lo": "CI lo", "hi": "CI hi", "n": "folds"})),
          "\n## Head precision (backbone fixed), 15 largest losses\n",
          md_table(hd[["cell", "ref_mean", "mean", "diff", "lo", "hi", "n"]].rename(columns={"ref_mean": "FP32-head AUROC", "mean": "AUROC", "lo": "CI lo", "hi": "CI hi", "n": "folds"}))]
    (out / "worst_cells.md").write_text("\n".join(wl) + "\n")

    # 5. summary counts over the ALL-condition cells: FP32-head systems (INT8 and own-head rows with their own head)
    summ = []
    xa = stx[(stx.metric == "auroc_margin") & (stx.cond == "ALL")]
    for panel in ["backbone", "head"]:
        x = xa[xa.panel == panel]
        if panel == "backbone":
            x = x[(x["head"] == "esc") | x.backbone.isin(["INT8", "Q3ownhead"])]
        key = "backbone" if panel == "backbone" else "head"
        for lab, g in x.groupby(key):
            summ.append(dict(panel=panel, level=lab, systems=len(g), mean_diff=g["diff"].mean(), min_diff=g["diff"].min(),
                             max_diff=g["diff"].max(), ci_below_zero=int((g.hi < 0).sum()), ci_above_zero=int((g.lo > 0).sum()),
                             ci_spans_zero=int(((g.lo <= 0) & (g.hi >= 0)).sum())))
    sm = pd.DataFrame(summ)
    sm["_b"] = sort_key(sm.level, BB_ORDER + HEAD_ORDER)
    sm = sm.sort_values(["panel", "_b"]).drop(columns="_b")
    sm.to_csv(out / "summary_margin_auroc_by_level.csv", index=False)
    (out / "summary_margin_auroc_by_level.md").write_text(
        "Margin-gate AUROC against the FP32 reference, ALL conditions, one row per precision level; a system is one (task, model, length, head).\n\n"
        + md_table(sm, "{:+.4f}") + "\n")

    # 6. other gates against the margin, paired over folds, ALL conditions
    fa = f[f.cond == "ALL"].merge(sysdf.reset_index()[["sid", "task", "model", "backbone", "head", "length"]], on="sid", suffixes=("", "_s"))
    rows = []
    for sid, g in fa.groupby("sid"):
        g = g.sort_values("fold")
        for x in ("entropy", "energy", "tsmax"):
            r = paired(g[f"auroc_{x}"].to_numpy(), g["auroc_margin"].to_numpy())
            rows.append(dict(sid=sid, task=g.task.iloc[0], model=g.model.iloc[0], backbone=g.backbone.iloc[0], head=g["head"].iloc[0],
                             gate=x, mean_margin=g.auroc_margin.mean(), **r))
    gp = pd.DataFrame(rows)
    gp.to_csv(out / "gate_vs_margin_paired.csv", index=False)
    t = gp.groupby("gate").agg(systems=("sid", "size"), mean_diff=("diff", "mean"), ci_below_zero=("hi", lambda v: int((v < 0).sum())),
                                ci_above_zero=("lo", lambda v: int((v > 0).sum()))).reset_index()
    (out / "gate_vs_margin_paired.md").write_text(
        "Gate AUROC minus margin AUROC (ALL conditions), paired over folds, one row per gate over all systems.\n\n" + md_table(t, "{:+.4f}") + "\n")
    print("report written")


if __name__ == "__main__":
    main()
