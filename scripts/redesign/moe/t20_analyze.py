"""T20 analysis (ANALYSIS_PLAN_T20.md sections 2, 3.5 to 3.7). Reads test labels.
    python t20_analyze.py --run RESULTS --phase p1 --tasks urban esc50      (P1 only: ft-Q4 minus frozen w4g128, no cascade number)
    python t20_analyze.py --run RESULTS --phase full --tasks urban esc50 tau  (pick, cascade, hull, H1 to H3, S1; only after P1 passed)"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import t13_common as T13
from t13_common import A, NOISE, OPTB, S2, S3C
from t20_common import TASKS, SEEDS, cheap_file, inner_split, bed, test_rows
ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); ap.add_argument("--phase", choices=["p1", "full"], required=True)
ap.add_argument("--tasks", nargs="+", required=True); ap.add_argument("--family", type=int, default=17); a = ap.parse_args()
OUT = a.run / "tables"; OUT.mkdir(exist_ok=True)
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]; NBOOT = 10000
sm = A.softmax; marg = A.margin; f1f = A.f1_fast; thr_rate = A.thr_rate; hull_pair = A.hull_pair


# ---------------- loading ----------------
def tau_npz(p):
    z = np.load(p); iv = {int(k[7:]): z[k] for k in z.files if k.startswith("iv_idx_")}
    return {"z": z["test_logits"].astype(np.float64), "iv": {f: (iv[f].astype(np.int64), z[f"iv_logits_{f}"].astype(np.float64)) for f in sorted(iv)}}


def costs(task):
    if task == "tau":
        c = json.load(open(OPTB / "analysis/tau/costs.json"))
        return dict(cheap=c["cheap"]["cnn:mn20:w8a8:8"], mn10=c["cheap"]["cnn:mn10:w8a8:esc"], mn20=c["cheap"]["cnn:mn20:w8a8:esc"],
                    astfp32=c["ast"]["ast:fp32+esc"], astw4=c["ast"]["ast:w4g128+esc"], ftq4=c["ast"]["ast:w4g128+esc"], ftfp32=c["ast"]["ast:fp32+esc"])
    c = json.load(open(S3C / f"analysis/{task}/costs.json")); L = TASKS[task]["L"]; t = T13.cost_table(task)
    out = dict(mn10=t["mn10"], mn20=t["mn20"], astfp32=t["astfp32"], astw4=t["astw4"], astshort=t["astshort"], astshortfp32=c["ast"][f"fp32@{L}+esc"],
               ftq4=c["ast"][f"w4g128@{L}+esc"], ftfp32=c["ast"][f"fp32@{L}+esc"])
    out["cheap"] = t["mn10"] if task == "urban" else c["cheap"]["cnn:mn20:w8a8:4"]
    return out


def load_systems(task, seed, cond):
    """name -> {'z': [N,K] test logits (row's own test fold), 'iv': {fold: (rows, logits)}} for cond in clean, field or a noise cell (TAU: clean)."""
    S = {}
    if task == "tau":
        S["cheap"] = tau_npz(cheap_file("tau", seed))
        for e in ("mn10", "mn20", "astfp32", "astw4"): S[e] = tau_npz(T13.tau_file(e, seed))
    else:
        from analyze import ScoreDir
        if task == "urban": S["cheap"] = T13.load(task, "mn10", cond, seed)
        else: S["cheap"] = ScoreDir(S3C / f"scores_cnn/esc50/mn20_esc50_s{seed}/4").get("mn20_w8a8", cond)
        for e in ("mn10", "mn20", "astfp32", "astw4", "astshort"):
            if T13.available(task, e, cond, [seed]): S[e] = T13.load(task, e, cond, seed)
        d, _ = T13.sdir(task, "astshort", seed); sd = ScoreDir(d)
        if sd.has("fp32", cond): S["astshortfp32"] = sd.get("fp32", cond)
    return S


def load_ft(task, seed, cond, run):
    """ft-Q4 and ft-FP32 in the same layout; cond clean or field (TAU: clean, plus 'siv' rows)."""
    D = bed(task); N, K = D["N"], TASKS[task]["K"]; out = {}
    for cfg in ("q4", "fp32"):
        z = np.full((N, K), np.nan); iv = {}; siv = {}
        for f in range(1, TASKS[task]["nf"] + 1):
            p = run / "scores" / task / f"f{f}_s{seed}.npz"
            if not p.exists(): continue
            r = np.load(p); z[r["te"]] = r[f"{cfg}_{cond}_te"]; iv[f] = (r["iv"], r[f"{cfg}_{cond}_iv"].astype(np.float64))
            if task == "tau": siv[f] = (r["siv"], r[f"{cfg}_clean_siv"].astype(np.float64))
        out["ft" + cfg] = {"z": z, "iv": iv, "siv": siv}
    return out


def align_iv(sysd, f, rows):
    """logits of system on inner-validation rows `rows` of fold f (stored order may differ)."""
    r, z = sysd["iv"][f]; pos = {int(x): i for i, x in enumerate(r)}; return z[[pos[int(x)] for x in rows]]


# ---------------- stats ----------------
def nb(d, ratio, one_sided_lower=False):
    d = np.asarray(d, float); k = len(d); s2 = d.var(ddof=1); se = np.sqrt((1 / k + ratio) * s2)
    m = float(d.mean())
    if se == 0: return dict(mean=m, lo=m, hi=m, p=0.0 if m != 0 else 1.0, lb95=m)
    t = m / se; h = stats.t.ppf(.975, k - 1) * se
    return dict(mean=m, lo=m - h, hi=m + h, p=float(2 * stats.t.sf(abs(t), k - 1)), lb95=float(m - stats.t.ppf(.95, k - 1) * se))


def boot(d, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed); i = rng.integers(0, len(d), (NBOOT, len(d))); b = d[i].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def ratio(task, D):
    if task == "tau":
        n = [((D["fold"] == f) & ~D["is_a"]).sum() for f in range(1, 6)]; ntr = [((D["fold"] != f) & D["is_a"]).sum() for f in range(1, 6)]
    else:
        n = [(D["fold"] == f).sum() for f in range(1, TASKS[task]["nf"] + 1)]; ntr = [D["N"] - x for x in n]
    return float(np.mean(np.array(n) / np.array(ntr)))


def rows_of(task, D, f, which):
    te = test_rows(task, D, f)
    if task == "tau": return te[D["is_a"][te]] if which == "clean" else te[~D["is_a"][te]]
    return te


BEDS = {"urban": [("field", "field"), ("clean", "clean")], "esc50": [("field", "field"), ("clean", "clean")], "tau": [("shift", "clean"), ("clean", "clean")]}
PRIMARY = {"urban": "field", "esc50": "field", "tau": "shift"}
FROZEN = {"urban": "astshort", "esc50": "astshort", "tau": "astw4"}

res = {"phase": a.phase, "tasks": a.tasks}
# ---------------- P1 ----------------
p1rows = []
for task in a.tasks:
    D = bed(task); y = D["y"]; K = TASKS[task]["K"]; nf = TASKS[task]["nf"]; rt = ratio(task, D)
    for bname, cond in BEDS[task]:
        F = np.zeros((len(SEEDS), nf, 2))
        for i, s in enumerate(SEEDS):
            S = load_systems(task, s, cond); FT = load_ft(task, s, cond, a.run)
            for f in range(1, nf + 1):
                r = rows_of(task, D, f, bname)
                F[i, f - 1] = [100 * f1f(y[r], FT["ftq4"]["z"][r].argmax(1), K), 100 * f1f(y[r], S[FROZEN[task]]["z"][r].argmax(1), K)]
        fm = F.mean(0); d = fm[:, 0] - fm[:, 1]; t = nb(d, rt); lo, hi = boot(d)
        p1rows.append(dict(task=task, bed=bname, primary=bname == PRIMARY[task], ftq4_f1=fm[:, 0].mean(), frozen_w4g128_f1=fm[:, 1].mean(), diff=t["mean"], nb_lo=t["lo"], nb_hi=t["hi"], boot_lo=lo, boot_hi=hi,
                           p_nb=t["p"], folds_positive=int((d > 0).sum()), n_folds=nf, fold_diffs=np.round(d, 2).tolist(), passes_2pt=bool(t["mean"] >= 2.0)))
P1 = pd.DataFrame(p1rows); P1.to_csv(OUT / "p1.csv", index=False); print(P1.to_string())
res["p1"] = {r.task: dict(diff=r.diff, passes=r.passes_2pt) for r in P1[P1.primary].itertuples()}
if a.phase == "p1":
    (OUT / "p1_verdict.json").write_text(json.dumps(res, indent=1)); sys.exit(0)

# ---------------- full ----------------
def static_names(task, S, FT):
    base = [n for n in ("cheap", "mn10", "mn20", "astshort", "astshortfp32", "astw4", "astfp32") if n in S]
    if task == "urban": base.remove("mn10")         # the cheap tier is mn10 with the esc head on UrbanSound8K
    return base + ["ftq4", "ftfp32", "avg_ftq4", "avg_ftfp32", "avg_frozen"]


def probs(name, S, FT, rows=None, f=None, task=None, iv=False):
    """probabilities of a static on test rows (rows) or on fold f's inner-validation rows (iv=True, rows given)."""
    def one(n):
        src = S[n] if n in S else FT[n]
        return sm(align_iv(src, f, rows) if iv else src["z"][rows])
    if name.startswith("avg_"):
        other = {"avg_ftq4": "ftq4", "avg_ftfp32": "ftfp32", "avg_frozen": FROZEN[task]}[name]; return (one("cheap") + one(other)) / 2
    return one(name)


def cost_of(name, C, task):
    if name.startswith("avg_"): return C["cheap"] + C[{"avg_ftq4": "ftq4", "avg_ftfp32": "ftfp32", "avg_frozen": FROZEN[task]}[name]]
    return C[name]


rows_h1, rows_pick, rows_static, rows_h2, rows_h3, rows_sec = [], [], [], [], [], []
for task in a.tasks:
    D = bed(task); y = D["y"]; K = TASKS[task]["K"]; nf = TASKS[task]["nf"]; C = costs(task); rt = ratio(task, D); folds = range(1, nf + 1)
    clean = {s: (load_systems(task, s, "clean"), load_ft(task, s, "clean", a.run)) for s in SEEDS}
    names = static_names(task, *clean[0])
    # ---- pick on clean inner validation ----
    STRONG = ("ftq4", "ftfp32", FROZEN[task])
    ivF = {n: [] for n in names}; casc_iv = {(st, b): [] for st in STRONG for b in BUDGETS}; rate_iv = {b: [] for b in BUDGETS}; thr = {}
    for s in SEEDS:
        S, FT = clean[s]
        for f in folds:
            iv = inner_split(task, s, f); yv = y[iv]
            for n in names: ivF[n].append(f1f(yv, probs(n, S, FT, iv, f, task, iv=True).argmax(1), K))
            pc = probs("cheap", S, FT, iv, f, task, iv=True); m = marg(pc)
            for b in BUDGETS:
                th = thr_rate(m, b); thr[(s, f, b)] = th; e = m < th; rate_iv[b].append(e.mean())
                for st in STRONG:
                    ps = probs(st, S, FT, iv, f, task, iv=True); pred = np.where(e, ((pc + ps) / 2).argmax(1), pc.argmax(1)); casc_iv[(st, b)].append(f1f(yv, pred, K))
    spiv = pd.DataFrame([dict(system=n, f1=np.mean(ivF[n]), cost=cost_of(n, C, task)) for n in names])
    best = {}
    for st in STRONG:
        for b in BUDGETS:
            c = C["cheap"] + np.mean(rate_iv[b]) * C[st]; hv = hull_pair(spiv, c); g = np.mean(casc_iv[(st, b)]) - hv[0]
            rows_pick.append(dict(task=task, strong=st, budget=b, iv_rate=np.mean(rate_iv[b]), iv_cost=c, iv_f1=np.mean(casc_iv[(st, b)]), iv_hull=hv[0], hull_lo=hv[1], hull_hi=hv[2], gain_iv=g))
            if st not in best or g > best[st][1] + 1e-12: best[st] = (b, g)
    ptq = 100 * (np.mean(ivF["ftq4"]) - np.mean(ivF["ftfp32"]))
    # ---- test beds ----
    for bname, cond in BEDS[task]:
        SS = clean if cond == "clean" else {s: (load_systems(task, s, cond), load_ft(task, s, cond, a.run)) for s in SEEDS}
        Fs = {n: np.zeros((len(SEEDS), nf)) for n in names}
        variants = [("ftq4", "avg"), ("ftq4", "strong"), ("ftfp32", "avg"), (FROZEN[task], "avg")]
        Fc = {v: np.zeros((len(SEEDS), nf)) for v in variants}; Rr = {v: np.zeros((len(SEEDS), nf)) for v in variants}; Frg = np.zeros((len(SEEDS), nf))
        for i, s in enumerate(SEEDS):
            S, FT = SS[s]; rng = np.random.default_rng(100 + s)
            for f in folds:
                r = rows_of(task, D, f, bname); yt = y[r]
                P = {n: probs(n, S, FT, r, task=task) for n in names}
                for n in names: Fs[n][i, f - 1] = 100 * f1f(yt, P[n].argmax(1), K)
                m = marg(P["cheap"])
                for v in variants:
                    st, rule = v; e = m < thr[(s, f, best[st][0])]; comb = (P["cheap"] + P[st]) / 2 if rule == "avg" else P[st]
                    Fc[v][i, f - 1] = 100 * f1f(yt, np.where(e, comb.argmax(1), P["cheap"].argmax(1)), K); Rr[v][i, f - 1] = e.mean()
                    if v == ("ftq4", "avg"):
                        ne = int(e.sum()); acc = []
                        for _ in range(20):
                            er = np.zeros(len(r), bool); er[rng.choice(len(r), ne, replace=False)] = True
                            acc.append(100 * f1f(yt, np.where(er, ((P["cheap"] + P["ftq4"]) / 2).argmax(1), P["cheap"].argmax(1)), K))
                        Frg[i, f - 1] = np.mean(acc)
        for n in names: rows_static.append(dict(task=task, bed=bname, system=n, f1=Fs[n].mean(), cost=cost_of(n, C, task), fold_f1=np.round(Fs[n].mean(0), 2).tolist()))
        fmS = {n: Fs[n].mean(0) for n in names}
        sp = pd.DataFrame([dict(system=n, f1=fmS[n].mean(), cost=cost_of(n, C, task)) for n in names])
        old = [n for n in names if not (n.startswith("ft") or n in ("avg_ftq4", "avg_ftfp32"))]; sp_old = sp[sp.system.isin(old)]
        for v in variants:
            st, rule = v; fc = Fc[v].mean(0); rate = Rr[v].mean(); cost = C["cheap"] + rate * C[st]
            for hname, spx in (("rebuilt", sp), ("old_frozen", sp_old)):
                hv = hull_pair(spx, cost); lam = hv[3]; hf = fmS[hv[1]] if lam == 0 else (1 - lam) * fmS[hv[1]] + lam * fmS[hv[2]]
                d = fc - hf; t = nb(d, rt); lo, hi = boot(d)
                row = dict(task=task, bed=bname, strong=st, rule=rule, budget=best[st][0], gain_iv=best[st][1], rate=rate, cost=cost, cost_rel_strong=cost / C[st], cascade_f1=fc.mean(),
                           hull=hname, hull_f1=hf.mean(), hull_lo=hv[1], hull_hi=hv[2], hull_lam=lam, diff=t["mean"], nb_lo=t["lo"], nb_hi=t["hi"], boot_lo=lo, boot_hi=hi, p_nb=t["p"],
                           folds_positive=int((d > 0).sum()), n_folds=nf, fold_diffs=np.round(d, 2).tolist(), signflip_p=A.sign_flip_p(d), signflip_floor=2 / 2 ** nf,
                           random_gate_f1=Frg.mean() if v == ("ftq4", "avg") else np.nan, ptq_loss_iv_pts=ptq)
                (rows_h1 if (v == ("ftq4", "avg") and hname == "rebuilt") else rows_sec).append(row)
            if v == ("ftq4", "avg"):
                d2 = fc - fmS["ftq4"]; t2 = nb(d2, rt)
                rows_h2.append(dict(task=task, bed=bname, cascade_f1=fc.mean(), ftq4_f1=fmS["ftq4"].mean(), diff=t2["mean"], lb95_one_sided=t2["lb95"], cost=cost, ftq4_cost=C["ftq4"], cost_ratio=cost / C["ftq4"],
                                    passes=bool(t2["lb95"] > -1.0 and cost <= 0.5 * C["ftq4"])))
        dfz = Fc[("ftq4", "avg")].mean(0) - Fc[(FROZEN[task], "avg")].mean(0); tz = nb(dfz, rt)
        rows_sec.append(dict(task=task, bed=bname, strong="ftq4 minus frozen", rule="avg", hull="frozen_cascade", diff=tz["mean"], nb_lo=tz["lo"], nb_hi=tz["hi"], p_nb=tz["p"], folds_positive=int((dfz > 0).sum()), n_folds=nf, fold_diffs=np.round(dfz, 2).tolist()))
        # best single on inner validation of the condition family (reported)
        singles = [n for n in names if not n.startswith("avg_")]; fam = {n: [] for n in singles}
        for s in SEEDS:
            S, FT = SS[s]
            for f in folds:
                if task == "tau" and bname == "shift":
                    rr = FT["ftq4"]["siv"][f][0]
                    for n in singles: fam[n].append(f1f(y[rr], (FT[n]["siv"][f][1] if n.startswith("ft") else S[n]["z"][rr]).argmax(1), K))
                else:
                    iv = inner_split(task, s, f)
                    for n in singles: fam[n].append(f1f(y[iv], probs(n, S, FT, iv, f, task, iv=True).argmax(1), K))
        h1row = [r for r in rows_h1 if r["task"] == task and r["bed"] == bname][-1]
        bs = max(singles, key=lambda n: np.mean(fam[n])); bsc = max([n for n in singles if cost_of(n, C, task) <= h1row["cost"]] or ["cheap"], key=lambda n: np.mean(fam[n]))
        h1row.update(best_single=bs, best_single_f1=fmS[bs].mean(), best_single_cost=cost_of(bs, C, task), best_single_at_cost=bsc, best_single_at_cost_f1=fmS[bsc].mean())
    # ---- H3 humility (cheap tier only, pick budget of the primary strong tier) ----
    b = best["ftq4"][0]
    if task == "tau":
        S = {s: clean[s][0] for s in SEEDS}; rates = {}
        for dv in sorted(set(D["dev"])):
            v = []
            for s in SEEDS:
                for f in folds:
                    r = test_rows(task, D, f); r = r[D["dev"][r] == dv]; v.append((marg(sm(S[s]["cheap"]["z"][r])) < thr[(s, f, b)]).mean())
            rates[dv] = float(np.mean(v)); rows_h3.append(dict(task=task, kind="device", step=dv, rate=rates[dv]))
        exc = [dv for dv in rates if dv != "a" and rates[dv] <= rates["a"]]; rows_h3.append(dict(task=task, kind="verdict", step="all non-A above A", rate=float(len(exc) == 0), exceptions=" ".join(exc)))
    else:
        cells = ["clean"] + NOISE; rate = {}
        for cl in cells:
            v = []
            for s in SEEDS:
                Sc = load_systems(task, s, cl) if cl != "clean" else clean[s][0]
                for f in folds: r = test_rows(task, D, f); v.append((marg(sm(Sc["cheap"]["z"][r])) < thr[(s, f, b)]).mean())
            rate[cl] = float(np.mean(v)); rows_h3.append(dict(task=task, kind="cell", step=cl, rate=rate[cl]))
        exc = []
        for kind in ("bg", "pink", "white"):
            seq = ["clean"] + [f"{kind}__{x}db" for x in (20, 10, 5, 0)]
            exc += [f"{seq[j]}->{seq[j + 1]}" for j in range(4) if not rate[seq[j + 1]] > rate[seq[j]]]
        rows_h3.append(dict(task=task, kind="verdict", step="monotone in every noise kind", rate=float(len(exc) == 0), exceptions=" ".join(exc)))

H1 = pd.DataFrame(rows_h1); PR = H1[H1.apply(lambda r: r.bed == PRIMARY[r.task], axis=1)].copy()
PR["p_holm"] = holm(PR.p_nb.to_numpy()); PR["p_family17"] = np.minimum(1.0, a.family * PR.p_nb)
PR["passes"] = (PR["diff"] >= 1.0) & (PR.folds_positive > PR.n_folds / 2) & (PR.p_holm < 0.05)
need = 2 if len(a.tasks) == 3 else len(a.tasks)
CL = H1[H1.bed == "clean"].copy(); CL["p_holm"] = holm(CL.p_nb.to_numpy()); CL["passes"] = (CL["diff"] >= 1.0) & (CL.folds_positive > CL.n_folds / 2) & (CL.p_holm < 0.05)
pd.DataFrame(rows_pick).to_csv(OUT / "pick_iv.csv", index=False); pd.DataFrame(rows_static).to_csv(OUT / "statics.csv", index=False)
pd.concat([PR, CL]).to_csv(OUT / "h1.csv", index=False); pd.DataFrame(rows_sec).to_csv(OUT / "secondary_cascades.csv", index=False)
H2 = pd.DataFrame(rows_h2); H2.to_csv(OUT / "h2.csv", index=False); H3 = pd.DataFrame(rows_h3); H3.to_csv(OUT / "h3_escalation.csv", index=False)
res.update(H1=dict(passes=bool(PR.passes.sum() >= need), tasks_passing=PR.task[PR.passes].tolist(), need=need, per_task=PR[["task", "diff", "nb_lo", "nb_hi", "boot_lo", "boot_hi", "p_nb", "p_holm", "p_family17", "folds_positive", "n_folds", "passes"]].to_dict("records")),
           H2_primary=H2[H2.apply(lambda r: r.bed == PRIMARY[r.task], axis=1)][["task", "diff", "lb95_one_sided", "cost_ratio", "passes"]].to_dict("records"),
           H3=H3[H3.kind == "verdict"].to_dict("records"), S1_fires=bool((PR["diff"] < 0.3).all()), H2_clean=CL[["task", "diff", "p_holm", "passes"]].to_dict("records"))
(OUT / "verdict.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))
