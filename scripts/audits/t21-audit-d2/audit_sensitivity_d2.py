"""D2 sensitivity checks on the fixed design (refit/refit, b30 from clean inner validation). Linear-interpolation hull
inside every subset (statics and costs recomputed on the subset), so compare with the full-set linear row.
Usage: python -I audit_sensitivity_d2.py <T21 results> <audit out>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from audit_common import BUDGETS, FOLDS, SEEDS, Store, macro_f1, margin, nb_test, softmax, thr_for_rate, upper_hull_pair  # noqa: E402

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
pool = pd.read_csv(R / "csv/d2/pool.csv", dtype={"uploader": str})
pool["fname"] = pool.name.str.replace(".wav", "", regex=False)
y = pool.class_id.to_numpy()
fold = pool.folder.to_numpy()
cc = np.load(OUT / "cost_mn_per_clip.npy")
ca = np.load(OUT / "cost_ast_per_clip.npy")
C, S = Store(R, "mn10_w8a8"), Store(R, "ast_w4g128")
ratio = float(np.mean([(fold == f).sum() / (fold != f).sum() for f in FOLDS]))
log = open(OUT / "audit_sensitivity.log", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


thr = {}
for s in SEEDS:
    for f in FOLDS:
        pc = softmax(C.get(s, f"iv__{f}__clean__refit"))
        for b, r in BUDGETS.items():
            thr[(b, s, f)] = thr_for_rate(margin(pc), r)
pred = {}
for c in ("clean", "bg__10db", "bg__0db", "field"):
    for s in SEEDS:
        pc, pa = softmax(C.get(s, f"test__{c}__refit")), softmax(S.get(s, f"test__{c}__refit"))
        pred[(c, s)] = (pc.argmax(1), pa.argmax(1), ((pc + pa) / 2).argmax(1), margin(pc))
    C.drop("test__"), S.drop("test__")


def evaluate(c, keep=None, ast_cost=ca, b="b30", per_fold_hull=False, weights=None):
    keep = np.ones(len(y), bool) if keep is None else keep
    rows = []
    for s in SEEDS:
        mn, ast, avg, m = pred[(c, s)]
        for f in FOLDS:
            te = (fold == f) & keep
            esc = m[te] < thr[(b, s, f)]
            casc = np.where(esc, avg[te], mn[te])
            yt = y[te]
            rows.append({"seed": s, "fold": f, "mn": macro_f1(yt, mn[te]) * 100, "ast": macro_f1(yt, ast[te]) * 100, "avg": macro_f1(yt, avg[te]) * 100,
                         "cascade": macro_f1(yt, casc) * 100, "c_mn": cc[te].mean(), "c_ast": ast_cost[te].mean(), "c_avg": (cc[te] + ast_cost[te]).mean(),
                         "c_cas": np.mean(cc[te] + esc * ast_cost[te]), "rate": esc.mean(), "n": int(te.sum())})
    D = pd.DataFrame(rows)
    if per_fold_hull:
        hs = []
        for f in FOLDS:
            g = D[D.fold == f]
            pts = {nm: (g[f"c_{nm}"].mean(), g[nm].mean()) for nm in ("mn", "ast", "avg")}
            _, lo, hi, lam = upper_hull_pair(pts, g.c_cas.mean())
            hs.append(((1 - lam) * g[lo] + lam * g[hi]) if lam > 0 else g[lo])
        D["hull"] = pd.concat(hs).sort_index()
        hdesc = "per fold"
    else:
        pts = {nm: (D[f"c_{nm}"].mean(), D[nm].mean()) for nm in ("mn", "ast", "avg")}
        _, lo, hi, lam = upper_hull_pair(pts, D.c_cas.mean())
        D["hull"] = (1 - lam) * D[lo] + lam * D[hi] if lam > 0 else D[lo]
        hdesc = f"{lo}|{hi}|{lam:.4f}"
    fm = D.groupby("fold").mean(numeric_only=True)
    d = fm.cascade - fm.hull
    p, lo_, hi_ = nb_test(d.to_numpy(), ratio)
    out = {"cond": c, "budget": b, "hull": hdesc, "rate": D.rate.mean(), "n_test_clips": int(keep.sum()), "d_hull": d.mean(), "p_nb": p,
           "nb_ci": f"{lo_:.3f}..{hi_:.3f}", "folds_pos": int((d > 0).sum()), "ast_minus_mn": (fm.ast - fm.mn).mean(),
           "fold_d": json.dumps(np.round(d.to_numpy(), 3).tolist())}
    if weights is not None:
        w = np.array([weights[f] for f in d.index], float)
        out["d_hull_weighted_by_fold_size"] = float((d * w).sum() / w.sum())
    return out


res = []


def add(label, r):
    r = {"check": label, **r}
    res.append(r)
    P({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


fsize = {f: int((fold == f).sum()) for f in FOLDS}
for c in ("field", "clean"):
    add("reference (linear hull, full test)", evaluate(c, weights=fsize))
    add("hull per fold at each fold's cascade cost", evaluate(c, per_fold_hull=True))
    add("length-free AST cost (every clip at the mean AST cost)", evaluate(c, ast_cost=np.full(len(y), ca.mean())))
    add("Fire and Wood test clips dropped", evaluate(c, keep=~pool.category.isin(["Fire", "Wood"]).to_numpy()))
    for f_ in FOLDS:
        pass
# background drawn from a Freesound recording that is also an FSD50K pool clip
bp = json.loads((R / "emb/d2/mn_pool/bg_plan.json").read_text())
src = pool.fname.map(lambda f: bp[f]["bg_clip"].split("-")[0])
in_pool = src.isin(set(pool.fname)).to_numpy()
self_src = (src == pool.fname).to_numpy()
fid2fold = dict(zip(pool.fname, pool.folder))
src_fold = src.map(lambda x: fid2fold.get(x, 0)).to_numpy()
src_train = in_pool & (src_fold != fold)
P(f"pool clips whose background source recording is an FSD50K pool clip: {in_pool.sum()} ({in_pool.mean() * 100:.2f}%), "
  f"its own recording: {self_src.sum()}, a training clip of the same fold: {src_train.sum()}")
from_field = None
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    order = np.random.default_rng(5001).permutation(len(y))
    cells = [f"{k}__{s_}db" for k in ("bg", "pink", "white") for s_ in (10, 5, 0)]
    at = np.array(["clean"] * len(y), dtype=object)
    for r_, i in enumerate(order[(len(y) + 1) // 2:]):
        at[i] = cells[r_ % 9]
    from_field = at
except Exception as e:  # noqa: BLE001
    P("field assignment not rebuilt", e)
# my field assignment against the stored field logits: field rows equal the cell rows for clean, 10 dB and 0 dB cells
for T, nm in ((C, "mn10"), (S, "ast")):
    fz = T.get(0, "test__field__refit")
    ok, tot = 0, 0
    for c in ("clean", "bg__10db", "bg__0db", "pink__10db", "pink__0db", "white__10db", "white__0db"):
        z = T.get(0, f"test__{c}__refit")
        sel = from_field == c
        ok += int(np.all(fz[sel] == z[sel], axis=1).sum())
        tot += int(sel.sum())
    T.drop("test__")
    P(f"{nm}: field rows equal to the assigned cell's rows: {ok} of {tot} (5 dB cells not stored in scores: {int(np.isin(from_field, ['bg__5db', 'pink__5db', 'white__5db']).sum())})")
bg_field = np.isin(from_field, ["bg__10db", "bg__5db", "bg__0db"])
P(f"field test clips with bg noise from an FSD50K pool recording: {(bg_field & in_pool).sum()} of {bg_field.sum()} bg-cell clips")
for c in ("field", "bg__10db", "bg__0db"):
    add("background source in the FSD50K pool: those clips dropped", evaluate(c, keep=~in_pool))
    if c != "field":
        add("reference (linear hull, full test)", evaluate(c))
# near duplicates across folds (from audit_dups)
nn = OUT / "crossfold_nearest.csv.gz"
if nn.exists():
    nd = pd.read_csv(nn)
    assert (nd.name == pool.name).all()
    for t in (0.999, 0.995):
        for c in ("field", "clean"):
            add(f"test clips with a cross-fold MN10 FP32 partner above cosine {t} dropped", evaluate(c, keep=(nd.best_cross_cos <= t).to_numpy()))
hz = OUT / "pool_hashes.csv.gz"
if hz.exists():
    hh = pd.read_csv(hz)
    dup = hh.duplicated("pcm_md5", keep=False).to_numpy() & hh.pcm_md5.notna().to_numpy()
    for c in ("field", "clean"):
        add("test clips with an identical-PCM twin anywhere in the pool dropped", evaluate(c, keep=~dup))
# budget on another bed: field inner validation with the clean thresholds (descriptive)
brow = []
for b in BUDGETS:
    f1s, lin = [], []
    for s in SEEDS:
        for f in FOLDS:
            iv = C.get(s, f"iv_idx_{f}")
            pc, pa = softmax(C.get(s, f"iv__{f}__field__refit")), softmax(S.get(s, f"iv__{f}__field__refit"))
            ok = np.isfinite(pc).all(1) & np.isfinite(pa).all(1)
            yi, pc, pa = y[iv][ok], pc[ok], pa[ok]
            esc = margin(pc) < thr[(b, s, f)]
            avg = ((pc + pa) / 2).argmax(1)
            casc = np.where(esc, avg, pc.argmax(1))
            ci = cc[iv][ok]
            ai = ca[iv][ok]
            lam = (esc * ai).sum() / ai.sum()
            f1s.append(macro_f1(yi, casc))
            lin.append((1 - lam) * macro_f1(yi, pc.argmax(1)) + lam * macro_f1(yi, avg))
    brow.append({"budget": b, "bed": "field inner validation, clean thresholds", "gain_linear_mn_to_avg": (np.mean(f1s) - np.mean(lin)) * 100})
BF = pd.DataFrame(brow)
P(BF.round(4).to_string())
BF.to_csv(OUT / "audit_budget_on_field_iv.csv", index=False)
pd.DataFrame(res).to_csv(OUT / "audit_sensitivity.csv", index=False)
log.close()
