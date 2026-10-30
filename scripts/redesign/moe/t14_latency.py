"""T14 notion 3 (ANALYSIS_PLAN_T14.md section 5): median, p95 and p95/median of the pair, the cheap expert alone and the AST alone from the stored latency run.
Reads tables/cascade_latency.csv (as stored) and re-simulates the same procedure from the raw passes with a bootstrap over passes (200 replicates, 50,000 clips each) and
the stored seed-fold range of the escalation rate. No timing is run. python t14_latency.py --out DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t13_common import LAT
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
D = pd.read_csv(LAT / "tables/cascade_latency.csv"); D["p95_over_median"] = D.p95_ms / D.median_ms
D.to_csv(a.out / "latency_stored_ratio.csv", index=False)
# match: the kernel closest to the pick (n4 for the 4-bit esc50 AST, int8 dynamic for w8g64 urban); runtime ort_int8 for the CNN
MATCH = {"esc50": ("ort_int8", "n4_kleidiai", 512, "esc50", "mn20", "head_cheap_mn20_esc50_4bit", "head_ast_esc50_fp32"),
         "urban": ("ort_int8", "int8_dynamic", 400, "urban8k", "mn10", "head_cheap_mn10_urban_fp32", "head_ast_urban_fp32")}
B = pd.read_csv(LAT / "tables/block_choice.csv"); B = B[B.chosen]
def passes(step, ds, t):
    for key in ((step, ds, t), (step, "both", t)):
        r = B[(B.step == key[0]) & (B.dataset == key[1]) & (B.threads == key[2])]
        if len(r):
            r = r.iloc[0]; tag = "" if r.run == "first" else r.run
            for kind in ("torch", "ort"):
                f = LAT / "raw" / f"{kind}_passes_t{t}{tag}.csv.gz"
                if f.exists():
                    P = pd.read_csv(f); v = P[(P.step == key[0]) & (P.dataset == key[1])].ms.to_numpy()
                    if len(v): return v
    raise KeyError((step, ds, t))
rates = json.load(open(LAT / "tables/rates.json")); rng = np.random.default_rng(20261025); N, R = 50000, 200
rows = []
for t in (1, 2, 4):
    for task, (cnnrt, kern, L, ds, cnn, hc, ha) in MATCH.items():
        front, fb, hcs, has = passes("mn_front_end", ds, t), passes(f"fbank_{L}", ds, t), passes(hc, ds, t), passes(ha, ds, t)
        eff, ast = passes(f"{cnn}_{cnnrt.replace('ort_int8','ort_int8')}", ds, t), passes(f"ast_{L}_{kern}", ds, t)
        astfp = passes(f"ast_{L}_fp32", ds, t)
        for cond in ("clean", "field"):
            rr = rates[f"{task}/{cond}"]
            for rate_name, rate in (("rate", rr["rate"]), ("rate_min", rr["rate_min"]), ("rate_max", rr["rate_max"])):
                med, p95 = [], []; amed, ap95 = [], []; cmed, cp95 = [], []
                reps = R if rate_name == "rate" else 20
                for _ in range(reps):
                    d = lambda v: rng.choice(v, N)
                    ch = d(front) + d(eff) + d(hcs); st = d(fb) + d(ast) + d(has)
                    lat = ch + (rng.random(N) < rate) * st
                    med.append(np.median(lat)); p95.append(np.percentile(lat, 95)); amed.append(np.median(st)); ap95.append(np.percentile(st, 95))
                    cmed.append(np.median(ch)); cp95.append(np.percentile(ch, 95))
                q = lambda x: (float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
                pr = np.array(p95) / np.array(med); ar = np.array(ap95) / np.array(amed); cr = np.array(cp95) / np.array(cmed)
                rows.append(dict(threads=t, task=task, condition=cond, rate_which=rate_name, rate=rate, cnn_runtime=cnnrt, ast_kernel=kern,
                                 pair_median=q(med)[0], pair_p95=q(p95)[0], pair_ratio=q(pr)[0], pair_ratio_lo=q(pr)[1], pair_ratio_hi=q(pr)[2],
                                 ast_median=q(amed)[0], ast_p95=q(ap95)[0], ast_ratio=q(ar)[0], ast_ratio_lo=q(ar)[1], ast_ratio_hi=q(ar)[2],
                                 cheap_median=q(cmed)[0], cheap_p95=q(cp95)[0], cheap_ratio=q(cr)[0], cheap_ratio_lo=q(cr)[1], cheap_ratio_hi=q(cr)[2],
                                 holds=bool(q(pr)[0] <= q(ar)[0])))
pd.DataFrame(rows).to_csv(a.out / "latency_bootstrap.csv", index=False); print(pd.DataFrame(rows)[["threads", "task", "condition", "rate_which", "pair_ratio", "ast_ratio", "holds"]].round(2).to_string())
