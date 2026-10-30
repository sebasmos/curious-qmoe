"""D1 (FSDKaggle2018) follow-up: test clips whose background slice comes from a Freesound recording that is itself a
D1 clip, and the field-mix cascade minus hull with those clips dropped. Hull as the experimenter defines it: endpoints
and lambda from the (subset) test statics at the cascade's mean cost, expected macro F1 of random time-sharing per
(seed, fold) (exact; my own computation). Fixed design: refit / refit, b20 from clean inner validation (recomputed).
Usage: python -I audit_d1_noise_overlap.py <T21 results> <audit out> <worktree>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from audit_common import exact_mix_f1_present, macro_f1, margin, nb_test, signflip, softmax, thr_for_rate, upper_hull_pair  # noqa: E402

R, OUT, WT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
K, SEEDS, FOLDS, RATE = 41, [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], 0.20
log = open(OUT / "audit_d1_noise_overlap.log", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


pool = pd.read_csv(R / "csv/d1/pool.csv", dtype={"freesound_id": str})
extra = pd.read_csv(R / "csv/d1/extra.csv", dtype={"freesound_id": str})
pool["stem"] = pool.name.str.replace(".wav", "", regex=False)
y, fold = pool.class_id.to_numpy(), pool.folder.to_numpy()
# ---- costs (same formulas as audit_costs_d2.py, 41 classes) ----
L = np.load(R / "emb/d1/mn_pool/lengths__clean.npy").astype(np.int64)
nfr = np.minimum(1 + (L - 400) // 160, 1024)
t = (nfr - 16) // 10 + 1
assert np.array_equal(t, np.load(R / "emb/d1/ast_pool/time_patches.npy"))
npatch, ntok, d, h, nl, g = 12 * t, 12 * t + 2, 768, 3072, 12, 128
lin = nl * ntok * (4 * d * d + 2 * d * h)
resc = nl * ntok * (4 * d * (d // g) + h * (d // g) + d * (h // g))
att = nl * ntok * ntok * d
fb = nfr * (400 + 512 * 9 * 2.5 + 2 * 257 + 257 * 128 + 128)
ca = lin * 32 + resc * 1024 + att * 64 + att * 128 + npatch * 256 * d * 1024 + fb * 1024 + (768 * 640 + 640 * 320 + 320 * K) * 1024
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(2)
sys.path.insert(0, str(WT))
from QWave.efficientat_mn import build_mn  # noqa: E402  model definition only

net = build_mn("mn10").eval()
n32 = np.ceil(2 * L).astype(np.int64)
mfr = 1 + n32 // 320
memo = {}
for T in np.unique(mfr):
    acc = [0.0, 0.0]

    def hk(mod, i, o, acc=acc):
        if isinstance(mod, nn.Conv2d):
            acc[0] += o.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        else:
            acc[0] += mod.in_features * mod.out_features * (o.numel() // mod.out_features)
        acc[1] += o.numel()
    hs = [m.register_forward_hook(hk) for m in net.features.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        net.features(torch.zeros(1, 1, 128, int(T)))
    for x in hs:
        x.remove()
    memo[int(T)] = acc
mn_core = np.array([memo[int(T)][0] for T in mfr])
mn_out = np.array([memo[int(T)][1] for T in mfr])
mn_fe = 25 * n32 + n32 + mfr * (800 + 1024 * 10 * 2.5 + 2 * 513 + 513 * 128 / 4 + 2 * 128)
cc = mn_core * 64 + mn_out * 1024 + mn_fe * 1024 + (960 * 640 + 640 * 320 + 320 * K) * 1024
# ---- background source overlap ----
bp_m = json.loads((R / "emb/d1/mn_pool/bg_plan.json").read_text())
bp_a = json.loads((R / "emb/d1/ast_pool/bg_plan.json").read_text())
P("bg_plan identical between tiers:", bp_m == bp_a, "| keys equal pool stems:", set(bp_m) == set(pool.stem))
src = pool.stem.map(lambda s: bp_m[s]["bg_clip"].split("-")[0])
ids_pool, ids_extra = set(pool.freesound_id), set(extra.freesound_id)
in_pool = src.isin(ids_pool).to_numpy()
in_any = src.isin(ids_pool | ids_extra).to_numpy()
id2fold = dict(zip(pool.freesound_id, pool.folder))
src_fold = src.map(lambda x: id2fold.get(x, 0)).to_numpy()
own = (src == pool.freesound_id).to_numpy()
P(f"background source recordings used: {src.nunique()}; equal to a D1 pool clip id: {len(set(src) & ids_pool)}; "
  f"to a non-verified official-train clip id (not in the pool): {len(set(src) & (ids_extra - ids_pool))}")
P(f"pool clips whose background comes from a D1 pool recording: {in_pool.sum()} of {len(pool)} ({in_pool.mean() * 100:.2f}%); "
  f"from any D1 recording (pool or non-verified train): {in_any.sum()}; their own recording: {own.sum()}; "
  f"a training clip of the same fold: {(in_pool & (src_fold != fold)).sum()}; a test clip of the same fold: {(in_pool & (src_fold == fold) & ~own).sum()}")
order = np.random.default_rng(5001).permutation(len(y))
cells = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]
at = np.array(["clean"] * len(y), dtype=object)
for r_, i in enumerate(order[(len(y) + 1) // 2:]):
    at[i] = cells[r_ % 9]
bgf = np.isin(at, ["bg__10db", "bg__5db", "bg__0db"])
P(f"field mix: clips in a bg cell {bgf.sum()}, of which background from a D1 pool recording {(bgf & in_pool).sum()}, from any D1 recording {(bgf & in_any).sum()}")
src_lab = src.map(dict(zip(pool.freesound_id, pool.category)))
P(f"of the {(bgf & in_pool).sum()} field bg clips, background recording's D1 label equals the clip's own label: {int((src_lab[bgf & in_pool] == pool.category[bgf & in_pool]).sum())}")
# ---- scores, thresholds, predictions ----
Z = {(tier, s): np.load(R / f"scores/d1/pool_{tier}_s{s}.npz") for tier in ("mn10_w8a8", "ast_w4g128") for s in SEEDS}
for (tier, s), z in Z.items():
    assert np.array_equal(z["y"], y) and np.array_equal(z["fold"], fold)
thr = {(s, f): thr_for_rate(margin(softmax(Z[("mn10_w8a8", s)][f"iv__{f}__clean__refit"])), RATE) for s in SEEDS for f in FOLDS}
pred = {}
for s in SEEDS:
    pc, pa = softmax(Z[("mn10_w8a8", s)]["test__field__refit"]), softmax(Z[("ast_w4g128", s)]["test__field__refit"])
    pred[s] = {"mn": pc.argmax(1), "ast": pa.argmax(1), "avg": ((pc + pa) / 2).argmax(1), "m": margin(pc)}
ratio = float(np.mean([(fold == f).sum() / (fold != f).sum() for f in FOLDS]))


def run(label, keep):
    rows, P_ = [], {}
    for s in SEEDS:
        p = pred[s]
        for f in FOLDS:
            te = (fold == f) & keep
            esc = p["m"][te] < thr[(s, f)]
            pr = {"mn": p["mn"][te], "ast": p["ast"][te], "avg": p["avg"][te]}
            pr["cascade"] = np.where(esc, pr["avg"], pr["mn"])
            P_[(s, f)] = (y[te], pr)
            rows.append({"s": s, "f": f, **{nm: macro_f1(y[te], q, K) * 100 for nm, q in pr.items()},
                         "c_mn": cc[te].mean(), "c_ast": ca[te].mean(), "c_avg": (cc[te] + ca[te]).mean(),
                         "c_cas": np.mean(cc[te] + esc * ca[te]), "rate": esc.mean(), "absent": K - len(np.unique(y[te]))})
    D = pd.DataFrame(rows)
    pts = {nm: (D[f"c_{nm}"].mean(), D[nm].mean()) for nm in ("mn", "ast", "avg")}
    _, lo, hi, lam = upper_hull_pair(pts, D.c_cas.mean())
    D["hull"] = [exact_mix_f1_present(*P_[(r.s, r.f)][:1], P_[(r.s, r.f)][1][lo], P_[(r.s, r.f)][1][hi], lam, K)[1] * 100 if lam > 0
                 else r[lo] for r in D.itertuples()]
    fm = D.groupby("f").mean(numeric_only=True)
    dd = (fm.cascade - fm.hull).to_numpy()
    p, l_, h_ = nb_test(dd, ratio)
    out = {"check": label, "n_test_clips": int(keep.sum()), "hull": f"{lo}|{hi}|{lam:.4f}", "rate": D.rate.mean(), "cascade_f1": D.cascade.mean(),
           "hull_f1": D.hull.mean(), "d_hull": dd.mean(), "nb_lo": l_, "nb_hi": h_, "p_nb": p, "p_family17": min(1, 17 * p),
           "folds_pos": int((dd > 0).sum()), "p_signflip": signflip(dd), "fold_d": json.dumps(np.round(dd, 3).tolist()),
           "ast_minus_mn": (fm.ast - fm.mn).mean(), "max_absent_classes": int(D.absent.max())}
    P({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    return out


res = [run("reference, full field test", np.ones(len(y), bool)),
       run("dropped: background from a D1 pool recording (any field cell)", ~in_pool),
       run("dropped: background from any D1 recording (any field cell)", ~in_any),
       run("dropped: only field clips in a bg cell with background from a D1 pool recording", ~(bgf & in_pool))]
pd.DataFrame(res).to_csv(OUT / "audit_d1_noise_overlap.csv", index=False)
log.close()
