#!/usr/bin/env python3
"""Regenerate every paper table from the runs of scripts/reproduce_paper.sh.

Reads outputs/<tag>_s<seed>/<model_kind>/summary.json (protocol version 2
only; older summaries are skipped and listed) and outputs/efficiency/*.json,
then writes one JSON with every number and one booktabs LaTeX snippet per
table. Every reported value traces back to a run directory, a seed and a fold.

Statistics
- Scores are macro F1 on held-out test folds. mean +/- std is over all
  (seed, fold) values of an arm; std is the sample std (ddof=1).
- Arms are compared with a two-sided paired t-test matched on (seed, fold),
  using only pairs present in both arms, with a two-sided Wilcoxon
  signed-rank test as a robustness check. Bonferroni correction is applied
  within each declared family; the family size m is printed and stored.
- Variance is compared with Levene's test (median-centred) on the per-(seed,
  fold) scores of each dataset, never as a ratio of pooled standard
  deviations; the variance ratio is reported alongside, descriptively.
- Latency (from the efficiency benchmark) is compared with a two-sided
  Mann-Whitney U test and Levene's test on the per-call timings.

    python scripts/collect_paper_numbers.py [--outputs outputs] \\
        [--json outputs/paper_numbers.json] [--tex-dir outputs/paper_tables]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROTOCOL_VERSION = 2
DATASETS = {"esc": "ESC-50", "quinn": "Quinn", "urban": "UrbanSound8K"}

# Arm id -> (tag without the _s<seed> suffix, model kind). Tags follow
# scripts/reproduce_paper.sh; {ds} is the dataset key.
SINGLES = {"q1": "1", "q2": "2", "q4": "4", "q8": "8", "q16": "16",
           "fp32": "esc", "int8": "qesc", "bitnet": "bitnet"}
MOE_ALL = {"uniform": "{ds}_uniform_ctrl", "pprior": "{ds}_pprior_a1.0",
           "st_uniform": "{ds}_sharedtrunk_uniform_ctrl", "st_pprior": "{ds}_sharedtrunk_a1.0"}
MOE_ESC = {"pprior_a0.3": "esc_pprior_a0.3", "pprior_a0.5": "esc_pprior_a0.5",
           "kl_a0.3": "esc_kl_divergence_a0.3", "entropy_a0.3": "esc_entropy_regularization_a0.3",
           "sharp_a0.3": "esc_precision_sharp_a0.3", "escalation_a0.3": "esc_escalation_a0.3",
           "soft_escalation_a0.3": "esc_soft_escalation_a0.3",
           "mc1": "esc_pprior_a1.0_mc1", "mc5": "esc_pprior_a1.0_mc5", "mc20": "esc_pprior_a1.0_mc20",
           "gated": "esc_gated_a1.0", "experts_4": "esc_pprior_a1.0_4experts",
           "experts_8_16_ptq": "esc_pprior_a1.0_8_16_qesc", "experts_ptq": "esc_pprior_a1.0_qesc"}
LABELS = {
    "q1": "Q1", "q2": "Q2", "q4": "Q4", "q8": "Q8", "q16": "Q16 (fp16)", "fp32": "FP32",
    "int8": "FP32 + PTQ int8", "bitnet": "BitNet (ternary)",
    "uniform": "MoE BitNet/Q4/Q8, uniform", "pprior": "MoE BitNet/Q4/Q8, precision prior",
    "st_uniform": "Shared trunk, uniform", "st_pprior": "Shared trunk, precision prior",
    "pprior_a0.3": "Precision prior, alpha 0.3", "pprior_a0.5": "Precision prior, alpha 0.5",
    "kl_a0.3": "Polarization only, alpha 0.3", "entropy_a0.3": "Entropy regularization, alpha 0.3",
    "sharp_a0.3": "Precision sharp, alpha 0.3", "escalation_a0.3": "Hard escalation, alpha 0.3",
    "soft_escalation_a0.3": "Soft escalation, alpha 0.3",
    "mc1": "M = 1", "mc5": "M = 5", "mc20": "M = 20", "gated": "Uncertainty gating (0.9)",
    "experts_4": "BitNet/Q4/Q8/Q16", "experts_8_16_ptq": "BitNet/Q8/Q16/PTQ", "experts_ptq": "BitNet/PTQ",
}
PREVIEW = ("q1", "q2", "q4", "q8", "q16", "fp32", "int8", "bitnet", "uniform", "pprior", "st_uniform", "st_pprior")


def arm_key(ds: str, arm: str):
    if arm in SINGLES:
        return f"{ds}_singles", SINGLES[arm]
    if arm in MOE_ALL:
        return MOE_ALL[arm].format(ds=ds), "moe"
    if ds == "esc" and arm in MOE_ESC:
        return MOE_ESC[arm], "moe"
    return None


# ── loading ────────────────────────────────────────────────────────────────

def load_runs(root: Path):
    """{(tag, kind): {"records": {(seed, fold): fold record}, "meta": {...}}}"""
    runs, skipped = defaultdict(lambda: {"records": {}, "meta": {}}), []
    for path in sorted(root.glob("*/*/summary.json")):
        s = json.loads(path.read_text())
        m = re.match(r"(.+)_s(\d+)$", path.parent.parent.name)
        if s.get("protocol_version") != PROTOCOL_VERSION or not m:
            skipped.append(str(path.relative_to(root)))
            continue
        if int(s["seed"]) != int(m.group(2)):
            raise ValueError(f"{path}: summary seed {s['seed']} does not match its directory suffix")
        run = runs[(m.group(1), path.parent.name)]
        for f in s["folds"]:
            key = (int(f["seed"]), int(f["fold"]))
            if key in run["records"]:
                raise ValueError(f"{path}: seed {key[0]} fold {key[1]} appears in two summaries")
            run["records"][key] = f
        run["meta"].update({k: s.get(k) for k in ("parameter_count", "model_size_bytes", "fold_scheme",
                                                   "train_recipe", "primary_metric")})
        run["meta"].setdefault("run_dirs", []).append(str(path.parent))
    return runs, skipped


def values(run, metric="f1_macro"):
    if not run:
        return {}
    if metric == "alpha0_f1_macro":
        return {k: r["alpha0_eval"]["f1_macro"] for k, r in run["records"].items() if "alpha0_eval" in r}
    return {k: r[metric] for k, r in run["records"].items()}


# ── statistics ─────────────────────────────────────────────────────────────

def describe(vals: dict):
    a = np.array(list(vals.values()), dtype=float)
    if a.size == 0:
        return None
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size),
            "seeds": sorted({k[0] for k in vals}), "folds": sorted({k[1] for k in vals})}


def paired_test(a: dict, b: dict):
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return None
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    d = x - y
    if np.allclose(d, 0):
        p_t = p_w = 1.0
        t = 0.0
    else:
        res = stats.ttest_rel(x, y, alternative="two-sided")
        t, p_t = float(res.statistic), float(res.pvalue)
        if not np.isfinite(p_t):  # constant nonzero difference
            p_t = 0.0
        p_w = float(stats.wilcoxon(x, y, alternative="two-sided").pvalue)
    return {"n_pairs": len(keys), "mean_diff": float(d.mean()), "t": t, "p_ttest": p_t, "p_wilcoxon": p_w}


def levene_test(a, b):
    a, b = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    if a.size < 2 or b.size < 2:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va == 0 and vb == 0:
        return {"W": 0.0, "p": 1.0, "var_ratio": float("nan"), "n": [int(a.size), int(b.size)]}
    res = stats.levene(a, b, center="median")
    return {"W": float(res.statistic), "p": float(res.pvalue), "var_ratio": float(va / vb) if vb else float("inf"),
            "n": [int(a.size), int(b.size)]}


def bonferroni(tests: list, p_key: str):
    valid = [t for t in tests if t["result"] is not None and np.isfinite(t["result"][p_key])]
    m = len(valid)
    for t in valid:
        t["result"]["p_bonferroni"] = min(1.0, t["result"][p_key] * m)
    return m


# ── LaTeX ──────────────────────────────────────────────────────────────────

def tex_escape(s: str) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def pm(d, digits=3):
    return "--" if not d else f"{d['mean']:.{digits}f}$\\pm${d['std']:.{digits}f}"


def num(x, fmt="{:.3f}"):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def pval(p):
    if p is None or not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def join_blocks(blocks):
    """Row blocks (one per dataset) separated by a midrule, skipping empty ones."""
    rows = []
    for b in (b for b in blocks if b):
        rows += (["midrule"] if rows else []) + b
    return rows


def write_table(tex_dir: Path, name: str, header: list, rows: list, colspec: str, note: str):
    lines = [f"% {note}", "% Generated by scripts/collect_paper_numbers.py; do not edit by hand.",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == "midrule" else " & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (tex_dir / f"{name}.tex").write_text("\n".join(lines))


# ── main ───────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--json", default=None, help="default: <outputs>/paper_numbers.json (gitignored)")
    ap.add_argument("--tex-dir", default=None, help="default: <outputs>/paper_tables")
    args = ap.parse_args(argv)
    root = Path(args.outputs).resolve()
    json_path = Path(args.json) if args.json else root / "paper_numbers.json"
    tex_dir = Path(args.tex_dir) if args.tex_dir else root / "paper_tables"
    tex_dir.mkdir(parents=True, exist_ok=True)

    runs, skipped = load_runs(root)

    def run(ds, arm):
        k = arm_key(ds, arm)
        return runs.get(k) if k else None

    report = {"source": str(root), "skipped_summaries": skipped,
              "statistics": {"score": "macro F1 on held-out test folds",
                             "spread": "sample std (ddof=1) over all (seed, fold) values of an arm",
                             "paired": "two-sided paired t-test on (seed, fold) pairs, Wilcoxon signed-rank as check",
                             "correction": "Bonferroni within each family",
                             "variance": "Levene (median-centred) on per-(seed, fold) scores within a dataset"},
              "arms": {}, "tables": {}, "tests": {}}

    all_arms = list(SINGLES) + list(MOE_ALL) + list(MOE_ESC)
    desc = {}
    for ds in DATASETS:
        for arm in all_arms:
            r = run(ds, arm)
            if not r:
                continue
            d = describe(values(r))
            d.update({"accuracy": describe(values(r, "accuracy")), "f1_weighted": describe(values(r, "f1_weighted")),
                      "alpha0_f1_macro": describe(values(r, "alpha0_f1_macro")),
                      "parameter_count": r["meta"].get("parameter_count"),
                      "storage_bytes": r["meta"].get("model_size_bytes"),
                      "run_dirs": r["meta"]["run_dirs"], "tag": arm_key(ds, arm)[0], "kind": arm_key(ds, arm)[1]})
            routing = [x["routing"] for x in r["records"].values() if "routing" in x]
            if routing:
                d["routing_overall"] = np.mean([x["overall"] for x in routing], axis=0).tolist()
                if all("by_uncertainty_quartile" in x for x in routing):
                    d["routing_by_quartile"] = np.mean([x["by_uncertainty_quartile"] for x in routing], axis=0).tolist()
            desc[(ds, arm)] = d
            report["arms"][f"{ds}/{arm}"] = {**d, "per_fold": {f"{k[0]}:{k[1]}": v for k, v in values(r).items()}}

    def mparams(ds, arm):
        d = desc.get((ds, arm))
        return num(d["parameter_count"] / 1e6, "{:.2f}") if d and d["parameter_count"] else "--"

    def kb(ds, arm):
        d = desc.get((ds, arm))
        return num(d["storage_bytes"] / 1e3, "{:,.1f}") if d and d.get("storage_bytes") else "--"

    # Table: single-model bit-width, per dataset
    blocks = []
    for ds, dname in DATASETS.items():
        ref, block = desc.get((ds, "q16")), []
        for arm in ("q1", "q2", "q4", "q8", "q16", "bitnet", "fp32", "int8"):
            d = desc.get((ds, arm))
            if not d:
                continue
            rel = num(100 * d["mean"] / ref["mean"], "{:.1f}") if ref else "--"
            block.append([tex_escape(dname), tex_escape(LABELS[arm]), mparams(ds, arm), kb(ds, arm), pm(d), rel])
        blocks.append(block)
    rows = join_blocks(blocks)
    write_table(tex_dir, "singles_bitwidth", ["Dataset", "Model", "Params (M)", "Storage (KB)", "Macro F1", "\\% of Q16"],
                rows, "l l r r c r", "Single models by bit-width; storage at deployed precision.")
    report["tables"]["singles_bitwidth"] = rows

    # Table: cross-dataset, with Avg F1 computed here
    rows, avg = [], {}
    for arm in list(SINGLES) + list(MOE_ALL):
        cells = [desc.get((ds, arm)) for ds in DATASETS]
        if not any(cells):
            continue
        avg[arm] = float(np.mean([c["mean"] for c in cells])) if all(cells) else None
        params = next((mparams(ds, arm) for ds in DATASETS if desc.get((ds, arm))), "--")
        rows.append([tex_escape(LABELS[arm]), params] + [pm(c) for c in cells] + [num(avg[arm])])
    write_table(tex_dir, "cross_dataset", ["Model", "Params (M)"] + [tex_escape(n) for n in DATASETS.values()] + ["Avg F1"],
                rows, "l r c c c c", "Avg F1 is the mean of the three dataset means (only when all three ran).")
    report["tables"]["cross_dataset"] = {"rows": rows, "avg_f1": avg}

    # Table: MoE configurations
    rows = []
    for arm in list(MOE_ALL) + ["experts_4", "experts_8_16_ptq", "experts_ptq", "gated"]:
        cells = [desc.get((ds, arm)) for ds in DATASETS]
        if not any(cells):
            continue
        ds0 = next(ds for ds in DATASETS if desc.get((ds, arm)))
        rows.append([tex_escape(LABELS[arm]), mparams(ds0, arm), kb(ds0, arm)] + [pm(c) for c in cells]
                    + [num(avg.get(arm)) if arm in avg else "--"])
    write_table(tex_dir, "moe_configs", ["Configuration", "Params (M)", "Storage (KB)"]
                + [tex_escape(n) for n in DATASETS.values()] + ["Avg F1"], rows, "l r r c c c c",
                "MoE configurations; expert-set and gating arms run on ESC-50 only.")
    report["tables"]["moe_configs"] = rows

    # Tests and the routing comparison
    families = defaultdict(list)

    def add(family, name, a: dict, b: dict):
        families[family].append({"comparison": name, "result": paired_test(a, b)})

    for ds, dname in DATASETS.items():
        v = {arm: values(run(ds, arm)) for arm in all_arms}
        for arm in ("q1", "q2", "q4", "q8", "q16", "int8", "bitnet"):
            add("singles_vs_fp32", f"{dname}: {LABELS[arm]} vs FP32", v[arm], v["fp32"])
        for arm in ("q1", "q2", "q4", "q8"):
            add("bitwidth_vs_q16", f"{dname}: {LABELS[arm]} vs Q16", v[arm], v["q16"])
        add("routing", f"{dname}: precision prior vs uniform", v["pprior"], v["uniform"])
        add("routing", f"{dname}: shared trunk, prior vs uniform", v["st_pprior"], v["st_uniform"])
        add("routing", f"{dname}: shared trunk vs full experts (prior)", v["st_pprior"], v["pprior"])
        add("routing", f"{dname}: prior on vs off at inference (same weights)", v["pprior"],
            values(run(ds, "pprior"), "alpha0_f1_macro"))
        add("routing", f"{dname}: shared trunk, prior on vs off at inference", v["st_pprior"],
            values(run(ds, "st_pprior"), "alpha0_f1_macro"))
        for arm in ("uniform", "pprior", "st_pprior"):
            add("moe_vs_single", f"{dname}: {LABELS[arm]} vs Q4", v[arm], v["q4"])
    v = {arm: values(run("esc", arm)) for arm in all_arms}
    for arm in ("kl_a0.3", "entropy_a0.3", "sharp_a0.3", "escalation_a0.3", "soft_escalation_a0.3",
                "pprior_a0.3", "pprior_a0.5", "pprior"):
        add("esc_ablations", f"ESC-50: {LABELS[arm]} vs uniform", v[arm], v["uniform"])
    for arm in ("mc1", "mc5", "mc20", "gated", "experts_4", "experts_8_16_ptq", "experts_ptq"):
        add("esc_ablations", f"ESC-50: {LABELS[arm]} vs precision prior (default)", v[arm], v["pprior"])

    variance = []
    for ds, dname in DATASETS.items():
        base = values(run(ds, "uniform"))
        arms = ["pprior", "st_uniform", "st_pprior"] + (
            ["pprior_a0.3", "pprior_a0.5", "kl_a0.3", "entropy_a0.3", "sharp_a0.3", "escalation_a0.3",
             "soft_escalation_a0.3"] if ds == "esc" else [])
        for arm in arms:
            variance.append({"comparison": f"{dname}: {LABELS[arm]} vs uniform",
                             "result": levene_test(values(run(ds, arm)).values(), base.values())})
        variance.append({"comparison": f"{dname}: Q4 single vs uniform MoE",
                         "result": levene_test(values(run(ds, "q4")).values(), base.values())})
    families["variance_levene"] = variance

    for fam, tests in families.items():
        m = bonferroni(tests, "p" if fam == "variance_levene" else "p_ttest")
        report["tests"][fam] = {"family_size": m, "tests": tests}
        print(f"family {fam}: Bonferroni over m = {m} tests")

    # Routing comparison per dataset (uniform vs prior vs shared trunk)
    blocks = []
    for ds, dname in DATASETS.items():
        block = []
        arms = ["uniform", "pprior"] + (["pprior_a0.3", "pprior_a0.5", "kl_a0.3"] if ds == "esc" else []) \
            + ["st_uniform", "st_pprior"]
        for arm in arms:
            d = desc.get((ds, arm))
            if not d:
                continue
            lev = next((t["result"] for t in variance if t["comparison"] == f"{dname}: {LABELS[arm]} vs uniform"), None)
            a0 = d.get("alpha0_f1_macro")
            block.append([tex_escape(dname), tex_escape(LABELS[arm]), pm(d),
                          num(lev["var_ratio"], "{:.2f}") if lev else ("ref." if arm == "uniform" else "--"),
                          pval(lev.get("p_bonferroni")) if lev else "--", pm(a0) if a0 else "--"])
        blocks.append(block)
    rows = join_blocks(blocks)
    write_table(tex_dir, "routing_comparison", ["Dataset", "Routing", "Macro F1", "Var. ratio", "Levene $p_{adj}$",
                                                "F1, prior off at inference"], rows, "l l c r r c",
                "Variance ratio = var(arm) / var(uniform) on per-(seed, fold) scores; Levene p Bonferroni-adjusted.")
    report["tables"]["routing_comparison"] = rows

    # Ablation tables (ESC-50)
    def simple(name, arms, note, extra=None):
        rows = []
        for arm in arms:
            d = desc.get(("esc", arm))
            if d:
                rows.append([tex_escape(LABELS[arm]), pm(d)] + ([extra(arm)] if extra else []))
        write_table(tex_dir, name, ["Configuration", "Macro F1"] + (["Params (M)"] if extra else []), rows,
                    "l c" + (" r" if extra else ""), note)
        report["tables"][name] = rows

    simple("ablation_strategies", ["uniform", "kl_a0.3", "entropy_a0.3", "soft_escalation_a0.3", "pprior_a0.3",
                                   "sharp_a0.3", "escalation_a0.3", "pprior", "st_pprior"],
           "Routing strategies, ESC-50.")
    simple("ablation_alpha", ["uniform", "pprior_a0.3", "pprior_a0.5", "pprior"], "Precision prior strength, ESC-50.")
    simple("ablation_mc", ["mc1", "mc5", "pprior", "mc20"], "Monte Carlo samples (default M = 10), ESC-50.")
    simple("ablation_experts", ["pprior", "experts_4", "experts_8_16_ptq", "experts_ptq"],
           "Expert sets, precision prior alpha 1.0, ESC-50.")
    simple("ablation_arch", ["pprior", "st_pprior", "gated", "q4"], "Architecture variants, ESC-50.",
           extra=lambda arm: mparams("esc", arm))

    # Routing fractions on held-out data
    rows = []
    for ds, dname in DATASETS.items():
        for arm in list(MOE_ALL) + (list(MOE_ESC) if ds == "esc" else []):
            d = desc.get((ds, arm))
            if d and "routing_overall" in d:
                q = d.get("routing_by_quartile")
                rows.append([tex_escape(dname), tex_escape(LABELS[arm]), " / ".join(f"{x:.2f}" for x in d["routing_overall"]),
                             " / ".join(f"{x:.2f}" for x in q[0]) if q else "--",
                             " / ".join(f"{x:.2f}" for x in q[-1]) if q else "--"])
    write_table(tex_dir, "routing_fractions", ["Dataset", "Configuration", "Share per expert", "Least uncertain 25\\%",
                                               "Most uncertain 25\\%"], rows, "l l c c c",
                "Share of held-out samples per expert (expert order as configured), overall and by uncertainty quartile.")
    report["tables"]["routing_fractions"] = rows

    # Model size
    rows = []
    for arm in list(SINGLES) + list(MOE_ALL) + ["experts_4", "experts_8_16_ptq", "experts_ptq"]:
        ds0 = next((ds for ds in DATASETS if desc.get((ds, arm))), None)
        if ds0:
            rows.append([tex_escape(LABELS[arm]), f"{desc[(ds0, arm)]['parameter_count']:,}", kb(ds0, arm)])
    write_table(tex_dir, "model_size", ["Model", "Parameters", "Storage (KB)"], rows, "l r r",
                "Storage at deployed precision: k bits per weight plus per-row scales; biases, norms and heads fp32.")
    report["tables"]["model_size"] = rows

    # Efficiency (latency and energy), from the efficiency benchmark
    eff_rows, lat_tests, eff_report = [], [], {}
    for path in sorted((root / "efficiency").glob("*.json")):
        e = json.loads(path.read_text())
        ds = path.stem
        eff_report[ds] = {"machine": e["machine"], "protocol": e["protocol"], "arms": {}}
        for arm, a in e["arms"].items():
            en = a.get("energy", {})
            eff_report[ds]["arms"][arm] = {k: v for k, v in a.items() if k != "latency_raw_ms"}
            lat = a["latency"]
            eff_rows.append([tex_escape(DATASETS.get(ds, ds)), tex_escape(LABELS.get(arm, arm)) + ("$^*$" if a["simulated_quantization"] else ""),
                             num(a["storage_bytes"] / 1e3, "{:,.1f}"),
                             f"{lat['median_ms']:.3f} [{lat['q1_ms']:.3f}, {lat['q3_ms']:.3f}]",
                             num(a.get("router_share_of_median"), "{:.0%}").replace("%", "\\%"),
                             num(en.get("energy_mj_per_inference")), num(en.get("co2_ug_per_inference"))])
        raw = {arm: a["latency_raw_ms"] for arm, a in e["arms"].items()}
        for arm in raw:
            for ref in ("fp32", "int8"):
                if arm != ref and ref in raw and not (arm == "fp32" and ref == "int8"):
                    mw = stats.mannwhitneyu(raw[arm], raw[ref], alternative="two-sided")
                    lev = levene_test(raw[arm], raw[ref])
                    lat_tests.append({"comparison": f"{DATASETS.get(ds, ds)}: {LABELS.get(arm, arm)} vs {LABELS[ref]}",
                                      "result": {"p": float(mw.pvalue), "U": float(mw.statistic),
                                                 "median_diff_ms": float(np.median(raw[arm]) - np.median(raw[ref])),
                                                 "levene_p": lev["p"], "var_ratio": lev["var_ratio"]}})
        first = next(iter(e["arms"].values()), {})
        if "energy" in first:
            eff_report[ds]["energy_measurement"] = {k: first["energy"][k] for k in
                                                    ("cpu_model", "cpu_power_w", "ram_power_w", "tracking_mode",
                                                     "cpu_measurement_mode", "country_iso_code")}
    m = bonferroni(lat_tests, "p")
    report["tests"]["latency"] = {"family_size": m, "tests": lat_tests}
    print(f"family latency: Bonferroni over m = {m} tests")
    report["efficiency"] = eff_report
    write_table(tex_dir, "efficiency", ["Dataset", "Model", "Storage (KB)", "Latency, ms (median [IQR])",
                                        "Router share", "Energy (mJ/inf.)", "CO$_2$ ($\\mu$g/inf.)"], eff_rows,
                "l l r c r r r", "Batch-1 latency on held-out samples, arms interleaved; *fake-quantized layers "
                "(fp32 arithmetic), so latency is not a hardware measure.")
    report["tables"]["efficiency"] = eff_rows

    # Significance table: every family, with its size
    rows = []
    for fam, block in report["tests"].items():
        for t in block["tests"]:
            r = t["result"]
            if r is None:
                continue
            if fam in ("variance_levene", "latency"):
                rows.append([tex_escape(fam), tex_escape(t["comparison"]), "--", pval(r.get("p")),
                             pval(r.get("p_bonferroni")), "--", str(block["family_size"])])
            else:
                rows.append([tex_escape(fam), tex_escape(t["comparison"]), num(r["mean_diff"], "{:+.3f}"),
                             pval(r["p_ttest"]), pval(r.get("p_bonferroni")), pval(r["p_wilcoxon"]),
                             str(block["family_size"])])
    write_table(tex_dir, "significance", ["Family", "Comparison", "$\\Delta$F1", "p", "$p_{adj}$", "p (Wilcoxon)", "m"],
                rows, "l l r r r r r", "Two-sided tests; paired on (seed, fold) for F1, Levene for variance, "
                "Mann-Whitney U for latency; Bonferroni within each family of size m.")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"\n{len(desc)} arm/dataset cells from {len({d for d, _ in desc})} datasets; {len(skipped)} summaries skipped "
          f"(not protocol {PROTOCOL_VERSION} or no _s<seed> suffix)")
    for (ds, arm), d in sorted(desc.items()):
        if arm in PREVIEW:
            print(f"  {ds:6s} {LABELS[arm]:36s} {d['mean']:.3f} +/- {d['std']:.3f}  (n={d['n']}, seeds {d['seeds']})")
    print(f"\nJSON: {json_path}\nLaTeX tables: {tex_dir}")
    return report


if __name__ == "__main__":
    main()
