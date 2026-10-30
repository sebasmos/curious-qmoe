"""T0: reproduce the UrbanSound8K baselines from stored per-clip logits: single-model front, the pre-registered cascade P3, the
always-average ensemble, the time-sharing hull (with and without the ensemble), the cheapest-correct oracle and the per-clip
correctness table. Compared to stage 3c files, the reviewer-extras file and the BACKLOG numbers (ANALYSIS_PLAN.md B10).
Reads only stored results. Uses two repo helpers: idea5b_common.mix_expected_f1 (exact expected macro F1 of random time-sharing) and
QWave-free numpy for everything else.
    python s0_t0.py --out DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "idea6/stage2"))
from common import *
from idea5b_common import mix_expected_f1, holm

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
K, SEEDS = 10, [0, 1, 2]
NOISE = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]
S3CA = S3C / "analysis/urban"

def field_assignment(n, seed):  # ceil(n/2) clean, the rest over the nine cells in turn (audio_shift.field_assignment)
    order = np.random.default_rng(seed).permutation(n)
    out = np.array(["clean"] * n, dtype=object)
    for r, i in enumerate(order[(n + 1) // 2:]):
        out[i] = NOISE[r % 9]
    return out
N = 8732
AT, AV = field_assignment(N, 5001), field_assignment(N, 5002)   # test rows, inner-validation rows

def path_of(system, seed):
    p = system.split(":")
    if p[0] == "eff":
        return S2 / f"scores/urban/effp_urban_s{seed}/{p[2]}", p[1]
    if p[0] == "cnn":
        m, bb, k = p[1], p[2], p[3]
        root = S3C if m in ("mn05", "mn20", "dymn04") else S3B
        return root / f"scores_cnn/urban/{m}_urban_s{seed}/{k}", f"{m}_{bb}"
    n, h = system[4:].split("+")
    if "@" in n:
        base, L = n.split("@")
        if L == "400": return S2 / f"scores_ast_short/urban/astshort_urban_s{seed}/{h}", base
        return S3B / f"scores_ast{L}/urban/ast{L}_urban_s{seed}/{h}", base
    return I5 / f"scores/urban/asturban_s{seed}/{h}", n

_c = {}
def raw(system, seed, cond):
    key = (system, seed, cond)
    if key not in _c:
        d, cfg = path_of(system, seed)
        z = np.load(d / f"{cfg}__{cond}.npz")
        _c[key] = dict(y=z["y"].astype(np.int64), fold=z["test_fold"].astype(int), z=z["test_logits"].astype(np.float64),
                       iv={int(k[7:]): (z[k], z[f"iv_logits_{k[7:]}"].astype(np.float64)) for k in z.files if k.startswith("iv_idx_")})
    return _c[key]

def logits(system, seed, cond, iv=False):
    """Test logits (or inner-validation logits per fold) of a condition; 'field' assembled as in analyze.ScoreDir.get."""
    if cond != "field":
        return raw(system, seed, cond)
    key = (system, seed, "field")
    if key not in _c:
        base = raw(system, seed, "clean")
        z = np.full_like(base["z"], np.nan); ivs = {f: (idx, np.full_like(v, np.nan)) for f, (idx, v) in base["iv"].items()}
        for c in ["clean"] + NOISE:
            r = raw(system, seed, c); m = AT == c
            z[m] = r["z"][m]
            for f, (idx, v) in ivs.items():
                sel = AV[idx] == c
                v[sel] = r["iv"][f][1][sel]
        assert not np.isnan(z).any() and not any(np.isnan(v).any() for _, v in ivs.values())
        _c[key] = dict(y=base["y"], fold=base["fold"], z=z, iv=ivs)
    return _c[key]

def cond_free(system, seed):  # drop the big per-condition caches once a system is done
    for k in [k for k in _c if k[0] == system and k[1] == seed and k[2] != "clean"]: del _c[k]

F1 = lambda y, p: f1_macro(y, p, K)
def tests(x, z, keys, ratio):
    """analyze.tests: seeds averaged inside folds, t over fold means, Nadeau-Bengio, diff in fraction."""
    d = np.asarray(x) - np.asarray(z); folds = np.array([k[1] for k in keys])
    fm = np.array([d[folds == f].mean() for f in np.unique(folds)]); k = len(fm)
    sd = fm.std(ddof=1)
    p_t = float(stats.ttest_1samp(fm, 0.0).pvalue); t_nb = fm.mean() / np.sqrt((1 / k + ratio) * sd ** 2)
    return dict(diff=float(d.mean()), fold_min=float(fm.min()), fold_max=float(fm.max()), p_t=p_t, p_nb=float(2 * stats.t.sf(abs(t_nb), k - 1)))

costs = json.load(open(S3CA / "costs.json"))
def cost_of(sysname):
    if sysname.startswith("ast:"): return costs["ast"][sysname[4:]]
    return costs["cheap"][sysname]
stored_static = pd.read_csv(S3CA / "static.csv")
stored_static = stored_static[stored_static.set.isin(["stage3b", "new"])]
systems = sorted(stored_static.system.unique())

# ---- 1. single-model front: F1 of every static in the hull set, clean and field --------------------------------------------
rows, preds = [], {}
for sy in systems:
    for s in SEEDS:
        for cond in ("clean", "field"):
            r = logits(sy, s, cond)
            pr = r["z"].argmax(1)
            preds[(sy, s, cond)] = pr.astype(np.int8)
            for f in range(1, 11):
                te = np.flatnonzero(r["fold"] == f)
                rows.append(dict(system=sy, condition=cond, seed=s, fold=f, f1=F1(r["y"][te], pr[te]), acc=float((pr[te] == r["y"][te]).mean()), cost=cost_of(sy)))
        cond_free(sy, s)
    print("static", sy, flush=True)
ST = pd.DataFrame(rows)
ST.to_csv(a.out / "static_recomputed.csv", index=False)
cmp = ST.merge(stored_static[["system", "condition", "seed", "fold", "f1", "cost"]], on=["system", "condition", "seed", "fold"], suffixes=("", "_stored"))
check = dict(rows_recomputed=len(ST), rows_matched=len(cmp), max_abs_f1_diff=float((cmp.f1 - cmp.f1_stored).abs().max()),
             max_rel_cost_diff=float(((cmp.cost - cmp.cost_stored).abs() / cmp.cost_stored).max()))
print(check)
sp = {c: ST[ST.condition == c].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index() for c in ("clean", "field")}
def front(df):
    f = []
    for i, r in df.iterrows():
        dom = ((df.cost <= r.cost) & (df.f1 >= r.f1) & ((df.cost < r.cost) | (df.f1 > r.f1))).any()
        f.append(not dom)
    return df[f].sort_values("cost")
for c in sp: front(sp[c]).to_csv(a.out / f"front_{c}.csv", index=False)

# ---- 2. P3 cascade, ensemble, hull, oracle -----------------------------------------------------------------------------------
CH, AS, A256 = "cnn:mn10:w8a8:esc", "ast:fp32+esc", "ast:w4g128@256+esc"
c_ch, c_as = cost_of(CH), cost_of(AS)
ref = raw(CH, 0, "clean")
n_te = np.array([np.sum(ref["fold"] == f) for f in range(1, 11)]); RATIO = float(np.mean(n_te / (N - n_te)))
def thr_rate(m_iv, rate):  # analyze.thr_rate
    u = np.unique(m_iv); c = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    rates = (m_iv[None, :] < c[:, None]).mean(1)
    return c[np.lexsort((rates, np.abs(rates - rate)))[0]]
def oracle_pred(y, pc, de, n):
    fix, harm = (pc != y) & (de == y), (pc == y) & (de != y)
    order = np.argsort(np.where(fix, 0, np.where(harm, 2, 1)), kind="stable")
    e = np.zeros(len(y), bool); e[order[:n]] = True
    return np.where(e, de, pc)

res, repro = {}, []
stored_sel = pd.read_csv(S3CA / "selected.csv"); stored_sel = stored_sel[(stored_sel.family == "P3")]
stored_ca = pd.read_csv(S3CA / "cascades.csv.gz")
extras = pd.read_csv(Q / "2026-10-24_1818_reviewer-extras_code-ac7e49d/A/a_ensemble_hull.csv")
extras = extras[extras.design == "urban P3 (pre-registered)"]
for cond in ("clean", "field"):
    keys, casc_f1, ens_f1, rate, ys, pc_, de_, pe_, esc_, ch_, as_ = [], [], [], [], {}, {}, {}, {}, {}, {}, {}
    for s in SEEDS:
        rc, ra = logits(CH, s, cond), logits(AS, s, cond)
        base_c, base_a = logits(CH, s, "clean"), logits(AS, s, "clean")
        pcs, pas = softmax(rc["z"]), softmax(ra["z"])
        for f in range(1, 11):
            te = np.flatnonzero(rc["fold"] == f)
            m_iv = pmargin(softmax(base_c["iv"][f][1]))          # clean inner validation, as in stage 3c
            thr = thr_rate(m_iv, 0.50)
            esc = pmargin(pcs[te]) < thr
            pc = pcs[te].argmax(1); de = ((pcs[te] + pas[te]) / 2).argmax(1); y = rc["y"][te]
            pred = np.where(esc, de, pc)
            kk = (s, f); keys.append(kk); ys[kk] = y; pc_[kk] = pc; de_[kk] = de; esc_[kk] = esc
            ch_[kk], as_[kk] = pc, pas[te].argmax(1)
            casc_f1.append(F1(y, pred)); ens_f1.append(F1(y, de)); rate.append(float(esc.mean()))
    casc_f1, ens_f1, rate = np.array(casc_f1), np.array(ens_f1), np.array(rate)
    cost = c_ch + rate.mean() * c_as
    res[cond] = dict(casc_f1=casc_f1.mean(), rate=rate.mean(), cost=cost, ens_f1=ens_f1.mean(), ens_cost=c_ch + c_as,
                     cheap_f1=np.mean([F1(ys[k], ch_[k]) for k in keys]), ast_f1=np.mean([F1(ys[k], as_[k]) for k in keys]),
                     same_pred_as_ens=all(((np.where(esc_[k], de_[k], pc_[k])) == de_[k]).all() for k in keys))
    # stored comparison, row by row
    g = stored_ca[(stored_ca.cheap_src == "mn10") & (stored_ca.cheap_bb == "w8a8") & (stored_ca.cheap.astype(str) == "esc") & (stored_ca.expensive == "fp32") &
                  (stored_ca.rule == "avg") & (stored_ca.budget == "b50") & (stored_ca.condition == cond)].sort_values(["seed", "fold"])
    res[cond]["max_abs_casc_f1_diff_rowwise"] = float(np.abs(g.f1.to_numpy() - casc_f1).max())
    res[cond]["max_abs_rate_diff_rowwise"] = float(np.abs(g.rate.to_numpy() - rate).max())
    # hull
    sp0 = sp[cond]
    sp1 = pd.concat([sp0, pd.DataFrame([{"system": "ENSEMBLE", "f1": float(ens_f1.mean()), "cost": c_ch + c_as}])], ignore_index=True)
    s_rec = {sy: dict(zip(zip(gg.seed, gg.fold), gg.f1)) for sy, gg in ST[ST.condition == cond].groupby("system")}
    def hull_pair(spx, c):
        lo_set, hi_set = spx[spx.cost <= c * (1 + 1e-12)], spx[spx.cost > c * (1 + 1e-12)]
        best = None
        for lo in lo_set.itertuples():
            for v in [(lo.f1, lo.system, lo.system, 0.0)] + [((1 - (c - lo.cost) / (hi.cost - lo.cost)) * lo.f1 + (c - lo.cost) / (hi.cost - lo.cost) * hi.f1, lo.system, hi.system, (c - lo.cost) / (hi.cost - lo.cost)) for hi in hi_set.itertuples()]:
                if best is None or v[0] > best[0] + 1e-15: best = v
        return best
    def pred_of(system, kk):
        if system == "ENSEMBLE": return de_[kk]
        r = logits(system, kk[0], cond); te = np.flatnonzero(r["fold"] == kk[1]); return r["z"][te].argmax(1).astype(np.int64)
    for name, spx in (("without", sp0), ("with", sp1)):
        v, lo, hi, lam = hull_pair(spx, cost)
        h = np.array([ (ens_f1[keys.index(kk)] if lo == "ENSEMBLE" else s_rec[lo][kk]) if lam == 0.0 else float(mix_expected_f1(ys[kk], pred_of(lo, kk), pred_of(hi, kk), lam, K)[0]) for kk in keys])
        t = tests(casc_f1, h, keys, RATIO)
        res[cond][name] = dict(lo=lo, hi=hi, lam=lam, hull_f1=h.mean(), **t)
    t_ec = tests(casc_f1, ens_f1, keys, RATIO)
    res[cond]["casc_minus_ens"] = t_ec
    # random time-sharing at the same rate, oracle at the same k, full oracle (stage 3c baselines)
    rnd = np.mean([float(mix_expected_f1(ys[k], pc_[k], de_[k], esc_[k].mean(), K)[0]) for k in keys])
    orc = np.mean([F1(ys[k], oracle_pred(ys[k], pc_[k], de_[k], int(esc_[k].sum()))) for k in keys])
    orc_full = np.mean([F1(ys[k], np.where((pc_[k] != ys[k]) & (de_[k] == ys[k]), de_[k], pc_[k])) for k in keys])
    res[cond].update(random_f1=rnd, oracle_k_f1=orc, oracle_full_f1=orc_full)
    st = stored_sel[stored_sel.condition == cond].iloc[0]
    res[cond]["stored_selected"] = dict(f1=st.f1, rate=st.rate, cost=st.cost, hull_f1=st.hull_f1, hull_diff=st.hull_diff, hull_p_t=st.hull_p_t_fold, hull_p_nb=st.hull_p_nb,
                                        random_f1=st.random_f1, oracle_k_f1=st.oracle_k_f1, oracle_full_f1=st.oracle_full_f1)
    cond_free(CH, 0)

# ---- 3. cheapest-correct oracle over three tiers and the per-clip correctness table ------------------------------------------
TIERS = [("mn10_w8a8", CH), ("ast256_w4g128", A256), ("ast_fp32_1024", AS)]
tc = np.array([cost_of(t[1]) for t in TIERS])
tab = pd.DataFrame({"clip": np.arange(N), "stem": load_table().stem, "fold": ref["fold"], "y": ref["y"]})
orc_rows = []
for s in SEEDS:
    P = np.stack([logits(t[1], s, "clean")["z"].argmax(1) for t in TIERS], 1)          # [N, 3]
    C = P == ref["y"][:, None]
    for j, t in enumerate(TIERS): tab[f"correct_{t[0]}_s{s}"] = C[:, j].astype(np.int8)
    first = np.where(C.any(1), C.argmax(1), 0)                                         # cheapest correct tier, else the cheapest
    tab[f"cheapest_correct_tier_s{s}"] = np.where(C.any(1), C.argmax(1), -1).astype(np.int8)
    pred = np.where(C.any(1), ref["y"], P[:, 0])
    orc_rows.append(dict(seed=s, oracle_acc=float(C.any(1).mean()), oracle_f1=float(np.mean([F1(ref["y"][ref["fold"] == f], pred[ref["fold"] == f]) for f in range(1, 11)])),
                         oracle_mean_cost=float(tc[first].mean()), **{f"acc_{t[0]}": float(C[:, j].mean()) for j, t in enumerate(TIERS)},
                         share_tier1_needed=float((C.any(1) & ~C[:, 0] & C[:, 1]).mean()),
                         share_tier2_needed=float((C.any(1) & ~C[:, 0] & ~C[:, 1] & C[:, 2]).mean()), share_none=float((~C.any(1)).mean())))
tab.to_csv(a.out / "correctness_table.csv.gz", index=False)
ORC = pd.DataFrame(orc_rows); ORC.to_csv(a.out / "oracle_cheapest_correct.csv", index=False)

# ---- 4. gate AUROC and b30 escalation of the mn10 Q8 tier ---------------------------------------------------------------------
au, esc30 = [], []
for s in SEEDS:
    r = logits(CH, s, "clean"); p = softmax(r["z"]); m = pmargin(p); wrong = p.argmax(1) != r["y"]
    au.append(roc_auc_score(wrong, 1 - m))
    rt = []
    for f in range(1, 11):
        te = r["fold"] == f; thr = thr_rate(pmargin(softmax(r["iv"][f][1])), 0.30); rt.append(float((m[te] < thr).mean()))
    esc30.append(np.mean(rt))
stored_au = pd.read_csv(S3CA / "auroc_wrong.csv"); stored_au = stored_au[(stored_au.cheap_src == "mn10") & (stored_au.cheap_bb == "w8a8") & (stored_au.cheap.astype(str) == "esc") & (stored_au.condition == "clean")].sort_values("seed")
stored_es = pd.read_csv(S3CA / "escalation.csv"); stored_es = stored_es[(stored_es.cheap_src == "mn10") & (stored_es.cheap_bb == "w8a8") & (stored_es.cheap.astype(str) == "esc") & (stored_es.budget == "b30") & (stored_es.condition == "clean")]


# ---- 4b. BACKLOG -57: gate of the exact-length mn10 Q8 (FP32-trained head on exact-length embeddings, "esc") on the ten conditions ----
UNP = Q / "2026-10-23_1918_redesign-idea6-stage3c-unpadded_code-cbf6be6"
CONDS57 = ["clean", "reverb_rt300ms", "reverb_rt600ms", "reverb_rt1200ms", "clip_p50", "clip_p25", "clip_p10", "lowpass_4000hz", "lowpass_2000hz", "lowpass_1000hz"]
def unp(seed, cond):
    d = (UNP / f"scores_unpr/urban/mn10unp_urban_s{seed}/esc") if cond == "clean" else (FINAL / f"scores/urban/mn10unp_urban_s{seed}/esc")
    z = np.load(d / f"mn10_w8a8__{cond}.npz")
    return z["y"].astype(np.int64), z["test_fold"].astype(int), z["test_logits"].astype(np.float64), {int(k[7:]): z[f"iv_logits_{k[7:]}"].astype(np.float64) for k in z.files if k.startswith("iv_idx_")}
au57, rt57 = {c: [] for c in CONDS57}, {c: [] for c in CONDS57}
for sd in SEEDS:
    y0, f0, z0, iv0 = unp(sd, "clean")
    thr = {f: thr_rate(pmargin(softmax(iv0[f])), 0.30) for f in range(1, 11)}
    for c in CONDS57:
        y, fo, z, _ = unp(sd, c); p = softmax(z); m = pmargin(p)
        au57[c].append(roc_auc_score(p.argmax(1) != y, 1 - m))
        rt57[c].append(np.mean([float((m[fo == f] < thr[f]).mean()) for f in range(1, 11)]))
fin_au = pd.read_csv(FINAL / "analysis/b3_gate_auroc.csv"); fin_au = fin_au[fin_au.task == "urban"].set_index("condition").auroc_wrong
fin_es = pd.read_csv(FINAL / "analysis/b_urban/escalation_summary.csv"); fin_es = fin_es[(fin_es.tier == "mn10") & (fin_es.kind == "esc") & (fin_es.budget == "b30")].iloc[0]

# ---- 5. comparison table ----------------------------------------------------------------------------------------------------------
out = []
def add(q, source, expected, got, unit_scale=1.0, nd=3):
    e, g = round(float(expected) * unit_scale, nd), round(float(got) * unit_scale, nd)
    out.append(dict(quantity=q, source=source, expected=e, recomputed=g, match3=bool(e == g)))
for cond in ("clean", "field"):
    r, ss = res[cond], res[cond]["stored_selected"]
    ex = extras[extras.condition == cond]
    exw, exn = ex[ex.hull_set == "with"].iloc[0], ex[ex.hull_set == "without"].iloc[0]
    add(f"{cond}: cascade F1", "stage 3c selected.csv", ss["f1"], r["casc_f1"])
    add(f"{cond}: cascade F1 per-row max abs diff to cascades.csv.gz (must be 0.000)", "stage 3c cascades.csv.gz", 0.0, r["max_abs_casc_f1_diff_rowwise"])
    add(f"{cond}: escalation rate", "stage 3c selected.csv", ss["rate"], r["rate"])
    add(f"{cond}: cost, T bit-ops", "stage 3c selected.csv", ss["cost"] / 1e12, r["cost"] / 1e12)
    add(f"{cond}: always-average F1", "reviewer extras A", exw.ens_f1, r["ens_f1"])
    add(f"{cond}: always-average cost, T", "reviewer extras A", exw.ens_cost / 1e12, r["ens_cost"] / 1e12)
    add(f"{cond}: hull F1 without ensemble", "stage 3c selected.csv", ss["hull_f1"], r["without"]["hull_f1"])
    add(f"{cond}: cascade minus hull without ensemble, points", "stage 3c selected.csv", ss["hull_diff"], r["without"]["diff"], 100)
    add(f"{cond}: fold t p, hull without ensemble", "stage 3c selected.csv", ss["hull_p_t"], r["without"]["p_t"])
    add(f"{cond}: NB p, hull without ensemble", "stage 3c selected.csv", ss["hull_p_nb"], r["without"]["p_nb"])
    add(f"{cond}: hull F1 with ensemble", "reviewer extras A", exw.hull_f1, r["with"]["hull_f1"])
    add(f"{cond}: cascade minus hull with ensemble, points", "reviewer extras A / BACKLOG -59", exw["diff"], r["with"]["diff"], 100, 2)
    add(f"{cond}: fold t p, hull with ensemble", "reviewer extras A", exw.p_t, r["with"]["p_t"])
    add(f"{cond}: NB p, hull with ensemble", "reviewer extras A", exw.p_nb, r["with"]["p_nb"])
    add(f"{cond}: cascade minus ensemble, points", "reviewer extras A", exw.cascade_minus_ens, r["casc_minus_ens"]["diff"], 100, 2)
    add(f"{cond}: random time-sharing F1 at same rate", "stage 3c selected.csv", ss["random_f1"], r["random_f1"])
    add(f"{cond}: oracle (same k) F1", "stage 3c selected.csv", ss["oracle_k_f1"], r["oracle_k_f1"])
    add(f"{cond}: oracle (all fixable) F1", "stage 3c selected.csv", ss["oracle_full_f1"], r["oracle_full_f1"])
    add(f"{cond}: cheap tier F1", "reviewer extras A", exw.cheap_f1, r["cheap_f1"])
    add(f"{cond}: AST tier F1", "reviewer extras A", exw.ast_f1, r["ast_f1"])
add("BACKLOG -59: urban clean cascade minus hull (with ensemble), points", "BACKLOG -59 text", 0.0049, res["clean"]["with"]["diff"], 100, 2)
add("BACKLOG -59: urban field cascade minus hull (with ensemble), points", "BACKLOG -59 text", 0.0048, res["field"]["with"]["diff"], 100, 2)
add("BACKLOG -59: urban field escalation rate (62 percent)", "BACKLOG -59 text", 0.62, res["field"]["rate"], 1, 2)
add("BACKLOG -59: urban field cascade cost, T (83.3)", "BACKLOG -59 text", 83.3, res["field"]["cost"] / 1e12, 1, 1)
add("BACKLOG -59: urban always-average cost, T (133.7)", "BACKLOG -59 text", 133.7, res["field"]["ens_cost"] / 1e12, 1, 1)
add("BACKLOG -59: UrbanSound8K field, cascade and always-average make the same predictions on every clip (1 = yes)", "BACKLOG -59 text", 1, int(res["field"]["same_pred_as_ens"]))
add("gate AUROC mn10 Q8 esc, clean, seed-mean", "stage 3c auroc_wrong.csv", stored_au.auroc_wrong.mean(), np.mean(au))
add("clean escalation at b30, seed-mean (padded mn10 Q8 esc, stage 3c)", "stage 3c escalation.csv", stored_es.rate.mean(), np.mean(esc30))
for c in CONDS57:
    add(f"final round, exact-length mn10 Q8 gate AUROC, {c}", "final b3_gate_auroc.csv (BACKLOG -57)", fin_au[c], np.mean(au57[c]))
    add(f"final round, exact-length mn10 Q8 escalation at b30 clean threshold, {c}", "final escalation_summary.csv (BACKLOG -57)", fin_es[c], np.mean(rt57[c]))
add("BACKLOG -57 text: UrbanSound8K clean escalation 0.301", "BACKLOG -57 text", 0.301, np.mean(rt57["clean"]))
add("BACKLOG -57 text: reverb 0.3 s escalation 0.376", "BACKLOG -57 text", 0.376, np.mean(rt57["reverb_rt300ms"]))
add("BACKLOG -57 text: reverb 0.6 s escalation 0.378", "BACKLOG -57 text", 0.378, np.mean(rt57["reverb_rt600ms"]))
add("BACKLOG -57 text: reverb 1.2 s escalation 0.377", "BACKLOG -57 text", 0.377, np.mean(rt57["reverb_rt1200ms"]))
add("BACKLOG -57 text: clipping 0.5 escalation 0.310", "BACKLOG -57 text", 0.310, np.mean(rt57["clip_p50"]))
add("BACKLOG -57 text: low-pass 1 kHz escalation 0.670", "BACKLOG -57 text", 0.670, np.mean(rt57["lowpass_1000hz"]))
add("BACKLOG -57 text: low-pass 1 kHz gate AUROC 0.808", "BACKLOG -57 text", 0.808, np.mean(au57["lowpass_1000hz"]))
add("BACKLOG -57 text: reverb gate AUROC 0.83 to 0.85, lowest (0.83)", "BACKLOG -57 text", 0.83, min(np.mean(au57[c]) for c in CONDS57[1:4]), 1, 2)
add("BACKLOG -57 text: reverb gate AUROC 0.83 to 0.85, highest (0.85)", "BACKLOG -57 text", 0.85, max(np.mean(au57[c]) for c in CONDS57[1:4]), 1, 2)
add("static rows: max abs F1 diff to stage 3c static.csv (must be 0.000)", "stage 3c static.csv", 0.0, check["max_abs_f1_diff"] if "max_abs_f1_diff" in check else check["max_abs_f1_diff"])
CMP = pd.DataFrame(out); CMP.to_csv(a.out / "t0_comparison.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120); print(CMP.to_string())
for c in ("clean", "field"):
    r = res[c]
    print(c, {k: (v if not isinstance(v, dict) else {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in r.items() if k != "stored_selected"})
print(ORC.to_string())
summary = dict(static_check=check, results={c: {k: v for k, v in res[c].items() if k != "stored_selected"} for c in res}, oracle=ORC.to_dict("records"),
               all_match=bool(CMP.match3.all()), n_compared=len(CMP), n_mismatch=int((~CMP.match3).sum()))
write_json(a.out / "t0_summary.json", summary)
print("T0 ALL MATCH" if CMP.match3.all() else f"T0 MISMATCHES: {(~CMP.match3).sum()}")
