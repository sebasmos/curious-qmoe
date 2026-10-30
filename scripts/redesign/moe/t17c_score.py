"""T17(c) micro-pilot scoring (ANALYSIS_PLAN_T17.md section 8). Adapted from T12 t12_score.py. READS TEST LABELS (first (c) script that does).
    python t17c_score.py --runs DIR --steps N --seeds 0 1 2 --out FILE.json  (T12 frozen/twin/spec arrays are read from the T12 pilot folder for reference)"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1c_common import T1B, Q, load_bed, softmax, pmargin
ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=Path, required=True); ap.add_argument("--steps", type=int, required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--k", type=int, default=2); a = ap.parse_args()
T12 = Q / "2026-10-25_0312_moe-t12-feasibility_code-e469a35/pilot"
T1C = Q / "2026-10-25_0052_moe-t1c_code-7d7429e"; D = load_bed("urban"); y = D["y"]; assign = D["assign"]
CONDS = {"clean": "clean", "bg0": "bg__0db", "field": None}
def cell_logits(S, key, cn, te):
    if cn != "field": return S[f"{key}|{CONDS[cn]}"].astype(np.float64)
    cells = assign[te]; out = np.zeros_like(S[f"{key}|clean"], dtype=np.float64)
    for c in np.unique(cells): out[cells == c] = S[f"{key}|{c}"][cells == c]
    return out
corr = {cn: {} for cn in CONDS}; flag = {cn: [] for cn in CONDS}; overall = {cn: {} for cn in CONDS}
for s in a.seeds:
    S = np.load(T1C / f"urban/raw_strength/s{s}_f1.npz"); te = S["te"]; yt = y[te]; tau1 = float(S["tau1"])
    arms = {"frozen": np.load(T12 / f"frozen_k2_s{s}_n222.npz"), "twin": np.load(a.runs / f"twin_s{s}_n{a.steps}.npz"), "spec": np.load(a.runs / f"spec_s{s}_n{a.steps}.npz"), "t12_twin": np.load(T12 / f"twin_k2_s{s}_n222.npz"), "t12_spec": np.load(T12 / f"spec_k2_s{s}_n222.npz")}
    for cn in CONDS:
        z1 = cell_logits(S, "z1", cn, te); fl = pmargin(softmax(z1)) < tau1; flag[cn].append(fl)
        sets = {"E1": z1, "E2a_stored": cell_logits(S, "UA_E2a", cn, te), **{n: arms[n][f"test_{cn}"] for n in arms}}
        for n, z in sets.items():
            corr[cn].setdefault(n, []).append((z.argmax(1) == yt)); overall[cn].setdefault(n, []).append((z.argmax(1) == yt).mean() * 100)
rng = np.random.default_rng(12); res = {"seeds": a.seeds, "steps": a.steps}
# the flagged set depends on the seed (E1 is retrained per seed): accuracy on flagged clips is computed per seed, then averaged over seeds; the bootstrap resamples clips jointly for all seeds
for cn in CONDS:
    Fl = np.stack(flag[cn]).astype(float); Cm = {n: np.stack(v).astype(float) for n, v in corr[cn].items()}; N = Fl.shape[1]
    facc = lambda c, idx=slice(None): ((c[:, idx] * Fl[:, idx]).sum(1) / Fl[:, idx].sum(1)).mean() * 100
    r = {"n_rows": N, "n_flagged_per_seed": [int(f) for f in Fl.sum(1)], "overall_acc": {n: float(np.mean(v)) for n, v in overall[cn].items()}}
    r["flagged_acc"] = {n: float(facc(c)) for n, c in Cm.items()}
    r["flagged_acc_per_seed"] = {n: [float((c[i] * Fl[i]).sum() / Fl[i].sum() * 100) for i in range(len(a.seeds))] for n, c in Cm.items()}
    idx = rng.integers(0, N, (4000, N))
    for name, (u, v) in {"spec_minus_twin": ("spec", "twin"), "twin_minus_frozen": ("twin", "frozen"), "spec_minus_frozen": ("spec", "frozen"), "twin_minus_E1": ("twin", "E1"), "spec_minus_E1": ("spec", "E1")}.items():
        pt = facc(Cm[u]) - facc(Cm[v]); bs = np.sort([facc(Cm[u], i) - facc(Cm[v], i) for i in idx]); r[name] = dict(mean=float(pt), boot_lo=float(bs[99]), boot_hi=float(bs[3899]))
    r["spec_minus_twin_per_seed"] = [float(((Cm["spec"][i] - Cm["twin"][i]) * Fl[i]).sum() / Fl[i].sum() * 100) for i in range(len(a.seeds))]
    r["discordant_spec_right_twin_wrong_per_seed"] = [int(((Cm["spec"][i] == 1) & (Cm["twin"][i] == 0) & (Fl[i] == 1)).sum()) for i in range(len(a.seeds))]
    r["discordant_twin_right_spec_wrong_per_seed"] = [int(((Cm["spec"][i] == 0) & (Cm["twin"][i] == 1) & (Fl[i] == 1)).sum()) for i in range(len(a.seeds))]
    res[cn] = r
# pre-registered primary: mean of the spec-minus-twin contrasts over the two shifted conditions (bg0, field)
prim = (res["bg0"]["spec_minus_twin"]["mean"] + res["field"]["spec_minus_twin"]["mean"]) / 2
both_pos = res["bg0"]["spec_minus_twin"]["mean"] > 0 and res["field"]["spec_minus_twin"]["mean"] > 0
res["primary_contrast_points"] = prim; res["both_shifted_positive"] = bool(both_pos)
res["verdict"] = "PASS_MICROPILOT" if prim >= 1.0 else ("CLEARLY_NEGATIVE" if prim <= -1.0 else "INCONCLUSIVE")
res["V1_twin_minus_frozen_primary_pts"] = (res["bg0"]["twin_minus_frozen"]["mean"] + res["field"]["twin_minus_frozen"]["mean"]) / 2
a.out.write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k in ("primary_contrast_points", "both_shifted_positive", "verdict", "V1_twin_minus_frozen_primary_pts")}, indent=1))
for cn in CONDS: print(cn, res[cn]["n_flagged_per_seed"], {n: round(v, 2) for n, v in res[cn]["flagged_acc"].items()}, "spec-twin", round(res[cn]["spec_minus_twin"]["mean"], 2), [round(res[cn]["spec_minus_twin"]["boot_lo"], 2), round(res[cn]["spec_minus_twin"]["boot_hi"], 2)])
