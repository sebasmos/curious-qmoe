"""Check 5, latency view: measured FP32 CPU wall clock (batch 1, 2 threads; check5_costs.json and
check5_frontend_time.json) per clip, three EfficientNet front-end cases: none (the claim's
convention), mel array only (hop-32 STFT, mel, dB) and as built (plus the PNG render, decode and
resize). AST: fbank + forward. Without low-bit kernels every AST configuration takes the FP32 time,
so the hull's upper end is the best AST system. Hull by linear interpolation of mean F1 at the
cascade's mean time.   python latency.py OUT"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(sys.argv[1]); R = OUT / "recompute"
wc = json.load(open(OUT / "check5_costs.json"))["wall_clock_fp32_cpu_2threads"]
fe = json.load(open(OUT / "check5_frontend_time.json"))
t_eff = float(np.median(wc["efficientnet_b3_300_s"])); t_ast = float(np.median(wc["ast_1024_s"])) + fe["ast_total_median_s"]
mel = float(np.median([e["stft_mel_db_s"] for e in fe["efficientnet_front_end"]]))
built = float(np.median([e["stft_mel_db_s"] + e["render_png_s"] + e["decode_resize_s"] for e in fe["efficientnet_front_end"]]))
picks = {"esc50": ("esc", "w4g128", "ast", "b30"), "esc5": ("4", "w4g128", "ast", "b30"), "urban": ("esc", "w4g128", "avg", "b30")}
rows = []
for t, p in picks.items():
    sp = pd.read_csv(R / t / "static_points.csv"); sp = sp[sp.view == "i"]
    f_eff = sp[sp.system == f"eff:{p[0]}"].f1.iloc[0]
    ab = sp[sp.system.str.startswith("ast:")].sort_values("f1").iloc[-1]
    d = pd.read_csv(R / t / "designs.csv", dtype={"cheap": str})
    r = d[(d.view == "i") & (d.set == "claim") & (d.cheap == p[0]) & (d.tier == p[1]) & (d.rule == p[2]) & (d.point == p[3])].iloc[0]
    for case, extra in [("no front end (claim convention)", 0.0), ("mel array, hop 32", mel), ("as built: mel + PNG render + resize", built)]:
        te = t_eff + extra
        mean = te + r.rate * t_ast
        if mean >= t_ast:
            hull, how = ab.f1, f"{ab.system} alone is faster than the cascade's mean"
        else:
            lam = (mean - te) / (t_ast - te); hull = f_eff + lam * (ab.f1 - f_eff); how = f"eff:{p[0]} + {lam:.2f} x {ab.system}"
        rows.append({"task": t, "case": case, "cheap_s": te, "ast_s": t_ast, "cascade_mean_s": mean, "escalated_clip_s": te + t_ast,
                     "cascade_f1": r.f1, "hull_f1": hull, "hull": how, "diff_pts": 100 * (r.f1 - hull), "ast_best_f1": ab.f1})
L = pd.DataFrame(rows); L.to_csv(OUT / "check5_latency.csv", index=False)
pd.set_option("display.width", 250); print(L.to_string())
