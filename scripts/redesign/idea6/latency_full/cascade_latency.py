"""Per-clip latency of the Stage 3c cascade picks from the measured step times
and their escalation rates, against the static short AST alone and the static
CNN alone.

Picks (analysis/<task>/selected.csv of Stage 3c; rate = mean over seeds and
test folds, rate_min and rate_max over seed-fold pairs):
  ESC-50/50     family P3, the Stage 3c pick: mn20 W8A8 + 4-bit head ->
                w4g128 at 512 frames, avg, b05; field 0.175, clean 0.065
  UrbanSound8K  family S4, Stage 3b's P2 pick re-evaluated in Stage 3c: mn10
                W8A8 + FP32 head -> w8g64 at 400 frames, avg, b30; field 0.408,
                clean 0.300
A clip always pays the cheap path (EfficientAT front end on its dataset's
clip, the CNN, the cheap head); an escalated clip also pays the AST path
(fbank at 512 or 400 frames, the short AST, the AST head; the 'avg' rule
adds one average of two softmax vectors, not timed). The picks' AST tiers
(w4g128, w8g64) are fake quantization; the real kernels timed here stand in:
n4 KleidiAI (4-bit weights) for w4g128, int8 dynamic (8-bit weights) for
w8g64, and FP32 for both. The CNN runs as torch FP32, ONNX Runtime FP32 or
ONNX Runtime int8 (W8A8 is fake quantization, not timed).
Blocks: every timed block is kept in raw/ (first run, and reruns tagged _r2
of the blocks that met a load burst from other agents' jobs). Per step,
dataset and thread count the block with the lowest recorded load (the larger
of the 1-minute load before and after it) is used; tables/block_choice.csv
lists every block with its load and which one was taken.
Mean = E[cheap] + rate x E[AST path] from the means of the raw passes.
Median and p95 from 200,000 simulated clips (numpy default_rng(0)): each
step time drawn independently from that step's measured passes, each clip
escalated with probability rate. Static systems: the cheap path alone (the
static CNN) and the AST path alone (the static short AST), per thread count.

    python scripts/cascade_latency.py   -> tables/cascade_latency.csv, tables/steps.csv, tables/rates.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TASK = {  # task -> (dataset clip, AST frames, CNN, cheap head step, AST head step, selected.csv family, closest kernel)
    "esc50": ("esc50", 512, "mn20", "head_cheap_mn20_esc50_4bit", "head_ast_esc50_fp32", "P3", "n4_kleidiai"),
    "urban": ("urban8k", 400, "mn10", "head_cheap_mn10_urban_fp32", "head_ast_urban_fp32", "S4", "int8_dynamic"),
}
CNN = {"torch_fp32": "{cnn}_torch_fp32", "ort_fp32": "{cnn}_ort_fp32", "ort_int8": "{cnn}_ort_int8"}
AST = ("fp32", "n4_kleidiai", "int8_dynamic")
N_SIM = 200_000


def load_blocks():
    """Every timed block of every run: rows (step, dataset, threads, tag, load, n) and the passes of the chosen one."""
    blocks, passes = [], {}
    for sp in sorted((C.OUT / "raw").glob("*_summary_t*.json")):
        kind, rest = sp.name.split("_summary_t")
        threads, tag = int(rest[0]), rest[1:].replace(".json", "")
        d = json.loads(sp.read_text())
        P = pd.read_csv(C.OUT / "raw" / f"{kind}_passes_t{threads}{tag}.csv.gz")
        for r in d["rows"]:
            key = (r["step"], r["dataset"], threads)
            load = max(r["load_before"][0], r["load_after"][0])
            v = P[(P.step == r["step"]) & (P.dataset == r["dataset"])]["ms"].to_numpy()
            blocks.append({"step": r["step"], "dataset": r["dataset"], "threads": threads, "run": tag or "first",
                           "load_before": r["load_before"][0], "load_after": r["load_after"][0], "load_max": load,
                           "waited_s": r["waited_s"], "n": len(v), "median_ms": float(np.median(v)),
                           "iqr_ms": float(np.percentile(v, 75) - np.percentile(v, 25))})
            if key not in passes or load < passes[key][0]:
                passes[key] = (load, v, tag or "first")
    B = pd.DataFrame(blocks)
    B["chosen"] = [passes[(r.step, r.dataset, r.threads)][2] == r.run for r in B.itertuples()]
    return B.sort_values(["step", "dataset", "threads", "run"]), {k: v for k, (_, v, _) in passes.items()}


def main():
    (C.OUT / "tables").mkdir(exist_ok=True)
    B, samples = load_blocks()
    B.to_csv(C.OUT / "tables/block_choice.csv", index=False)
    threads = sorted({k[2] for k in samples})
    steps = pd.DataFrame([{"step": s, "dataset": d, "threads": t, "median_ms": np.median(v), "mean_ms": v.mean(),
                           "q25_ms": np.percentile(v, 25), "q75_ms": np.percentile(v, 75),
                           "iqr_ms": np.percentile(v, 75) - np.percentile(v, 25),
                           "p95_ms": np.percentile(v, 95), "n": len(v),
                           **{k: B[(B.step == s) & (B.dataset == d) & (B.threads == t) & B.chosen].iloc[0][k]
                              for k in ("run", "load_before", "load_after")}}
                          for (s, d, t), v in samples.items()])
    steps.sort_values(["step", "dataset", "threads"]).to_csv(C.OUT / "tables/steps.csv", index=False)

    def get(step, ds, t):
        for key in ((step, ds, t), (step, "both", t)):
            if key in samples:
                return samples[key]
        raise KeyError((step, ds, t))

    rates = {}
    for task, (_, _, _, _, _, fam, _) in TASK.items():
        sel = pd.read_csv(C.S3C / f"analysis/{task}/selected.csv")
        for cond in ("clean", "field"):
            r = sel[(sel.family == fam) & (sel.condition == cond)]
            if len(r) != 1:
                raise SystemExit(f"{task} {fam} {cond}: {len(r)} rows")
            r = r.iloc[0]
            rates[(task, cond)] = {"rate": float(r.rate), "rate_min": float(r.rate_min), "rate_max": float(r.rate_max),
                                   "design": f"{r.cheap_bb}+{r.cheap} -> {r.expensive}, {r.rule}, {r.budget}",
                                   "family": fam, "f1": float(r.f1)}
    rng = np.random.default_rng(0)
    out = []
    for t in threads:
        for task, (ds, L, cnn, hc, ha, fam, closest) in TASK.items():
            front, fb = get("mn_front_end", ds, t), get(f"fbank_{L}", ds, t)
            hcs, has = get(hc, ds, t), get(ha, ds, t)
            ast_path = {}
            for an in AST:
                ast = get(f"ast_{L}_{an}", ds, t)
                draw = rng.choice(fb, N_SIM) + rng.choice(ast, N_SIM) + rng.choice(has, N_SIM)
                mean = fb.mean() + ast.mean() + has.mean()
                ast_path[an] = (draw, mean)
                out.append({"threads": t, "task": task, "system": f"static short AST ({L} frames)", "cnn": "",
                            "ast": an, "condition": "any", "rate": 1.0, "mean_ms": mean, "median_ms": np.median(draw),
                            "p95_ms": np.percentile(draw, 95)})
            for cn, pat in CNN.items():
                eff = get(pat.format(cnn=cnn), ds, t)
                cheap_draw = rng.choice(front, N_SIM) + rng.choice(eff, N_SIM) + rng.choice(hcs, N_SIM)
                cheap_mean = front.mean() + eff.mean() + hcs.mean()
                out.append({"threads": t, "task": task, "system": f"static CNN ({cnn})", "cnn": cn, "ast": "",
                            "condition": "any", "rate": 0.0, "mean_ms": cheap_mean,
                            "median_ms": np.median(cheap_draw), "p95_ms": np.percentile(cheap_draw, 95)})
                for an in AST:
                    ast_draw, ast_mean = ast_path[an]
                    for cond in ("clean", "field"):
                        rr = rates[(task, cond)]
                        esc = rng.random(N_SIM) < rr["rate"]
                        lat = cheap_draw + esc * ast_draw
                        out.append({"threads": t, "task": task, "system": "cascade", "cnn": cn, "ast": an,
                                    "ast_closest_to_pick": an == closest, "condition": cond, "rate": rr["rate"],
                                    "rate_min": rr["rate_min"], "rate_max": rr["rate_max"], "design": rr["design"],
                                    "mean_ms": cheap_mean + rr["rate"] * ast_mean, "median_ms": np.median(lat),
                                    "p95_ms": np.percentile(lat, 95), "cheap_path_ms": cheap_mean,
                                    "escalated_clip_ms": cheap_mean + ast_mean,
                                    "static_ast_mean_ms": ast_mean,
                                    "faster_than_static_ast_mean": cheap_mean + rr["rate"] * ast_mean < ast_mean,
                                    "faster_than_static_ast_median": float(np.median(lat)) < float(np.median(ast_draw)),
                                    "faster_than_static_ast_p95": float(np.percentile(lat, 95)) < float(np.percentile(ast_draw, 95))})
    df = pd.DataFrame(out)
    df.to_csv(C.OUT / "tables/cascade_latency.csv", index=False)
    (C.OUT / "tables/rates.json").write_text(json.dumps({f"{k[0]}/{k[1]}": v for k, v in rates.items()}, indent=1))
    pd.set_option("display.width", 250)
    cols = ["threads", "task", "system", "cnn", "ast", "condition", "rate", "mean_ms", "median_ms", "p95_ms"]
    print(df[df.threads == 2][cols].round(3).to_string())


if __name__ == "__main__":
    main()
