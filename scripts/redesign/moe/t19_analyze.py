"""T19 steps 19.9 and 19.10 (ANALYSIS_PLAN_T19.md B7, B8): cascade, pick on clean inner validation, hull, comparators, H1 to H4, S1, S2.
CPU only; reads score files, writes <results>/analysis/. Every number in the summary carries the table it comes from.

    python t19_analyze.py --cheap n4a4 [--tasks esc50 urban tau esc5]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t19_common import LEGACY_SHORT, LEGACY_TAU, OPTB, S2, STATICS_NEW, TASKS, results_dir, system_cost  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("s2_analyze", Path(__file__).resolve().parents[1] / "idea6/stage2/analyze.py")
A = importlib.util.module_from_spec(_spec)   # idea 6 stage 2 helpers: ScoreDir (field assembly), thr_rate, hull_pair
_spec.loader.exec_module(A)                  # (by path: idea5 also has an analyze.py earlier on sys.path)
import t13_common as T13  # noqa: E402  stored CNN tiers
from idea5b_common import holm  # noqa: E402

warnings.filterwarnings("ignore")
R = results_dir()
BUDGETS = [0.05, 0.10, 0.20, 0.30]
RULES = ["B", "avg"]
NOISE = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (20, 10, 5, 0)]
CORR = ["reverb_rt300ms", "reverb_rt600ms", "reverb_rt1200ms", "clip_p50", "clip_p25", "clip_p10",
        "lowpass_4000hz", "lowpass_2000hz", "lowpass_1000hz"]
FAMILY = 17
rng0 = np.random.default_rng(0)


def softmax(z):
    z = z - np.nanmax(z, 1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def macro_f1(y, pred, K):
    """sklearn macro F1 over the classes present in y or pred (as benchmark.py and T13), in points."""
    return 100 * A.f1_fast(np.asarray(y, np.int64), np.asarray(pred, np.int64), K)


# ── loading ───────────────────────────────────────────────────────────────

class Sys:
    """One static system of a task: kind new/leg/legstored/cnn, config, head variant; get(seed, bed) -> dict."""

    def __init__(self, task, name, cost, loader):
        self.task, self.name, self.cost, self.loader = task, name, cost, loader
        self.cache = {}

    def get(self, seed, bed):
        k = (seed, bed)
        if k not in self.cache:
            self.cache[k] = self.loader(seed, bed)
        return self.cache[k]


def tau_my(d: Path, cfg):
    def load(seed, bed):
        f = d / f"s{seed}" / f"{cfg}__clean.npz"
        if not f.exists():
            return None
        z = np.load(f)
        meta = TAU_META
        iv = {int(k[7:]): (z[k], z[f"iv_logits_{k[7:]}"].astype(np.float64)) for k in z.files if k.startswith("iv_idx_")}
        ivs = {int(k[8:]): (z[k], z[f"ivs_logits_{k[8:]}"].astype(np.float64)) for k in z.files if k.startswith("ivs_idx_")}
        return {"y": z["y"].astype(int), "fold": z["test_fold"].astype(int), "device": meta, "z": z["test_logits"].astype(np.float64),
                "iv": iv, "ivs": ivs}
    return load


def tau_stored(cfg):
    def load(seed, bed):
        f = OPTB / f"scores/ast_tau_s{seed}/esc/{cfg}__all.npz"
        if not f.exists():
            return None
        z = np.load(f)
        iv = {int(k[7:]): (z[k], z[f"iv_logits_{k[7:]}"].astype(np.float64)) for k in z.files if k.startswith("iv_idx_")}
        return {"y": z["y"].astype(int), "fold": z["fold"].astype(int), "device": z["device"].astype(str), "z": z["test_logits"].astype(np.float64),
                "iv": iv, "ivs": None}
    return load


def tau_cnn(ex):
    def load(seed, bed):
        D = T13.tau_load(ex, seed)
        return {**D, "ivs": None}
    return load


def short_dir(d: Path, cfg):
    sds = {}

    def load(seed, bed):
        p = d / f"s{seed}" if d.name != "esc" else d
        p = Path(str(p).replace("{seed}", str(seed)))
        if p not in sds:
            if not p.exists():
                return None
            sds[p] = A.ScoreDir(p)
        sd = sds[p]
        try:
            if not sd.has(cfg, bed):
                return None
            return sd.get(cfg, bed)
        except (SystemExit, FileNotFoundError, KeyError):
            return None
    return load


def cnn_short(task, ex):
    def load(seed, bed):
        if not T13.available(task, ex, bed, [seed]):
            return None
        return T13.load(task, ex, bed, seed)
    return load


def systems(task, cheap):
    """Every static of B3 available for the task, with priced cost (and unpriced in .ucost)."""
    out = {}
    sc = R / "scores" / task
    for c in STATICS_NEW:
        for head, sub in (("own", c), ("w8head", f"{c}@w8a8")):
            if head == "w8head" and c == "w8a8":
                continue
            d = sc / sub
            if not d.exists():
                continue
            ld = tau_my(d, c) if task == "tau" else short_dir(d, c)
            out[f"{c}:{head}"] = Sys(task, f"{c}:{head}", system_cost(task, c), ld)
            out[f"{c}:{head}"].ucost = system_cost(task, c, False)
    for c in (LEGACY_TAU if task == "tau" else LEGACY_SHORT):
        d = sc / f"L_{c}"
        if d.exists():
            ld = tau_my(d, c) if task == "tau" else short_dir(d, c)
            out[f"L:{c}:own"] = Sys(task, f"L:{c}:own", system_cost(task, "L:" + c), ld)
            out[f"L:{c}:own"].ucost = system_cost(task, "L:" + c, False)
        if task == "tau":
            ld = tau_stored(c)
        else:
            ld = short_dir(Path(str(S2 / f"scores_ast_short/{task}/astshort_{task}_s") + "{seed}") / "esc", c)
        out[f"L:{c}:stored"] = Sys(task, f"L:{c}:stored", system_cost(task, "L:" + c), ld)
        out[f"L:{c}:stored"].ucost = system_cost(task, "L:" + c, False)
    ct = T13.cost_table(task)
    for ex in ("mn10", "mn20"):
        out[ex] = Sys(task, ex, ct[ex], tau_cnn(ex) if task == "tau" else cnn_short(task, ex))
        out[ex].ucost = ct[ex]
    return out


# ── beds and rows ─────────────────────────────────────────────────────────

def bed_src(task, bed):
    """(score condition, row selector) of a bed: TAU clean/shift/device:<d> come from the one clean file."""
    if task == "tau":
        return "clean", bed
    return bed, None


def test_rows(D, fold, sel):
    m = D["fold"] == fold
    if sel is None:
        m &= np.isfinite(D["z"]).all(1)
        return np.flatnonzero(m)
    dev = D["device"]
    if sel == "clean":
        return np.flatnonzero(m & (dev == "a"))
    if sel == "shift":
        return np.flatnonzero(m & (dev != "a"))
    return np.flatnonzero(m & (dev == sel.split(":")[1]))


def iv_of(D, fold, family):
    """(rows, logits) of fold's inner validation for the family: clean, field (assembled) or TAU shift (ivs)."""
    if family == "shift":
        return None if D.get("ivs") is None else D["ivs"][fold]
    return D["iv"][fold]


def folds_of(task):
    return list(range(1, TASKS[task]["nf"] + 1))


# ── fold-level helpers ────────────────────────────────────────────────────

def static_fold_f1(S, task, bed):
    """[nf] fold F1 (seeds averaged) of a static on a bed, or None."""
    K = TASKS[task]["K"]
    cond, sel = bed_src(task, bed)
    vals = []
    for f in folds_of(task):
        v = []
        for s in TASKS[task]["seeds"]:
            D = S.get(s, cond)
            if D is None:
                return None
            r = test_rows(D, f, sel)
            if len(r) == 0 or not np.isfinite(D["z"][r]).all():
                return None
            v.append(macro_f1(D["y"][r], D["z"][r].argmax(1), K))
        vals.append(np.mean(v))
    return np.array(vals)


def static_iv_f1(S, task, family):
    K = TASKS[task]["K"]
    cond = "clean" if family in ("clean", "shift") or task == "tau" else family
    v = []
    for s in TASKS[task]["seeds"]:
        D = S.get(s, cond)
        if D is None:
            return None
        for f in folds_of(task):
            iv = iv_of(D, f, family)
            if iv is None or not np.isfinite(iv[1]).all():
                return None
            v.append(macro_f1(D["y"][iv[0]], iv[1].argmax(1), K))
    return float(np.mean(v))


def nb(d, ratio):
    d = np.asarray(d, float)
    K = len(d)
    s2 = d.var(ddof=1)
    se = np.sqrt((1 / K + ratio) * s2)
    if se == 0:
        return dict(mean=float(d.mean()), lo=float(d.mean()), hi=float(d.mean()), p=1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / se
    h = stats.t.ppf(.975, K - 1) * se
    return dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), p=float(2 * stats.t.sf(abs(t), K - 1)))


def boot(d, n=10000):
    d = np.asarray(d, float)
    m = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def maj(K):
    return K // 2 + 1


def ratio_of(task, bed):
    """mean over folds of n_test / n_train (TAU: bed rows of the fold over device-A training rows)."""
    S = SYS[task][f"w8a8:own"]
    cond, sel = bed_src(task, bed)
    D = S.get(TASKS[task]["seeds"][0], cond)
    r = []
    for f in folds_of(task):
        nt = len(test_rows(D, f, sel))
        ntr = int(((D["fold"] != f) & ((D["device"] == "a") if task == "tau" else True)).sum())
        r.append(nt / ntr)
    return float(np.mean(r))


# ── the cascade ───────────────────────────────────────────────────────────

def cascade(task, cheap_sys, exp_sys, bed, budget, rule, iv_mode=False, thr_override=None, random_gate=0, oracle=False):
    """Per (seed, fold): F1, rate, n_escalated. Thresholds from the cheap tier's clean inner-validation margins of the fold.
    iv_mode: evaluate on the clean inner-validation rows instead of a test bed."""
    K = TASKS[task]["K"]
    cond, sel = bed_src(task, bed)
    rows = []
    for s in TASKS[task]["seeds"]:
        Cc, Ce = cheap_sys.get(s, "clean"), exp_sys.get(s, "clean")
        Dc, De = (Cc, Ce) if iv_mode else (cheap_sys.get(s, cond), exp_sys.get(s, cond))
        if Dc is None or De is None:
            return None
        for f in folds_of(task):
            ivr, ivz = Cc["iv"][f]
            assert np.array_equal(ivr, Ce["iv"][f][0]), "inner validation rows differ between tiers"
            thr = A.thr_rate(margin(softmax(ivz)), budget)
            if iv_mode:
                y, zc, ze = Cc["y"][ivr], ivz, Ce["iv"][f][1]
            else:
                r = test_rows(Dc, f, sel)
                assert not np.isin(r, ivr).any(), "threshold rows overlap test rows"
                y, zc, ze = Dc["y"][r], Dc["z"][r], De["z"][r]
            pc, pe = softmax(zc), softmax(ze)
            esc = margin(pc) < thr
            fused = pe if rule == "B" else (pc + pe) / 2
            pred = np.where(esc, fused.argmax(1), pc.argmax(1))
            row = dict(seed=s, fold=f, f1=macro_f1(y, pred, K), rate=float(esc.mean()), n=len(y), n_esc=int(esc.sum()))
            if random_gate:
                g = np.random.default_rng(1000 * s + f)
                v = []
                for _ in range(random_gate):
                    e2 = np.zeros(len(y), bool)
                    e2[g.choice(len(y), esc.sum(), replace=False)] = True
                    v.append(macro_f1(y, np.where(e2, fused.argmax(1), pc.argmax(1)), K))
                row["f1_random_gate"] = float(np.mean(v))
                v = []
                for _ in range(random_gate):
                    e2 = np.zeros(len(y), bool)
                    e2[g.choice(len(y), esc.sum(), replace=False)] = True
                    v.append(macro_f1(y, np.where(e2, pe.argmax(1), pc.argmax(1)), K))
                row["f1_random_ts"] = float(np.mean(v))
            if oracle:
                tgt = (pc.argmax(1) != y) & (fused.argmax(1) == y)
                order = np.lexsort((g.random(len(y)) if random_gate else np.arange(len(y)), ~tgt))
                e2 = np.zeros(len(y), bool)
                e2[order[:esc.sum()]] = True
                row["f1_oracle"] = macro_f1(y, np.where(e2, fused.argmax(1), pc.argmax(1)), K)
            rows.append(row)
    return pd.DataFrame(rows)


def fold_mean(df, col):
    return df.groupby("fold")[col].mean().sort_index().to_numpy()


# ── main ──────────────────────────────────────────────────────────────────

def hull_at(task, bed, cost, ucost_view=False, exclude=()):
    """T13 rule: hull pair on the bed's mean F1 at `cost`, fold values interpolated. Returns (fold array, description)."""
    rows, arr = [], {}
    for name, S in SYS[task].items():
        if name in exclude:
            continue
        a = static_fold_f1(S, task, bed)
        if a is None:
            continue
        arr[name] = a
        rows.append(dict(system=name, f1=a.mean(), cost=S.ucost if ucost_view else S.cost))
    for name, (a, c, uc) in AVG[task].get(bed, {}).items():
        arr[name] = a
        rows.append(dict(system=name, f1=a.mean(), cost=uc if ucost_view else c))
    sp = pd.DataFrame(rows)
    v, lo, hi, lam = A.hull_pair(sp, cost)
    h = arr[lo] if lam == 0 else (1 - lam) * arr[lo] + lam * arr[hi]
    return h, f"{lo} + {hi} lam {lam:.3f}" if lam else lo


def always_avg(task, cheap_sys, exp_sys, bed):
    K = TASKS[task]["K"]
    cond, sel = bed_src(task, bed)
    vals = []
    for f in folds_of(task):
        v = []
        for s in TASKS[task]["seeds"]:
            Dc, De = cheap_sys.get(s, cond), exp_sys.get(s, cond)
            if Dc is None or De is None:
                return None
            r = test_rows(Dc, f, sel)
            v.append(macro_f1(Dc["y"][r], (softmax(Dc["z"][r]) + softmax(De["z"][r])).argmax(1), K))
        vals.append(np.mean(v))
    return np.array(vals)


SYS, AVG = {}, {}
TAU_META = None


def main():
    global TAU_META
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--tasks", nargs="+", default=["esc50", "urban", "tau", "esc5"])
    ap.add_argument("--out", default="analysis", help="output folder inside the results folder (analysis_interim before TAU is in)")
    a = ap.parse_args()
    out = R / a.out
    out.mkdir(exist_ok=True)
    TAU_META = pd.read_csv(TASKS["tau"]["csv"], usecols=["device"])["device"].astype(str).to_numpy()
    cheap = a.cheap
    res = {"cheap": cheap, "tasks": {}}
    primary_bed = {"esc50": "field", "esc5": "field", "urban": "field", "tau": "shift"}
    all_static_rows, casc_rows = [], []
    for task in a.tasks:
        if not (R / "scores" / task / cheap).exists():
            print(f"{task}: no scores, skipped")
            continue
        SYS[task] = systems(task, cheap)
        cs, es = SYS[task][f"{cheap}:own"], SYS[task]["w8a8:own"]
        c_cheap, c_exp = cs.cost, es.cost
        uc_cheap, uc_exp = cs.ucost, es.ucost
        beds = ["clean", primary_bed[task]] if task != "tau" else ["clean", "shift"]
        AVG[task] = {}
        for bed in beds:
            av = always_avg(task, cs, es, bed)
            if av is not None:
                AVG[task][bed] = {"always-average": (av, c_cheap + c_exp, uc_cheap + uc_exp)}
        # statics table
        for bed in beds:
            for name, S in SYS[task].items():
                a_ = static_fold_f1(S, task, bed)
                if a_ is not None:
                    all_static_rows.append(dict(task=task, bed=bed, system=name, cost_T=S.cost / 1e12, ucost_T=S.ucost / 1e12,
                                                f1=a_.mean(), folds=json.dumps(np.round(a_, 4).tolist())))
            if bed in AVG[task]:
                av = AVG[task][bed]["always-average"][0]
                all_static_rows.append(dict(task=task, bed=bed, system="always-average", cost_T=(c_cheap + c_exp) / 1e12,
                                            ucost_T=(uc_cheap + uc_exp) / 1e12, f1=av.mean(), folds=json.dumps(np.round(av, 4).tolist())))
        # pick on clean inner validation
        ivst = []
        for name, S in SYS[task].items():
            try:
                v = static_iv_f1(S, task, "clean")
            except AssertionError:
                v = None
            if v is not None:
                ivst.append(dict(system=name, f1=v, cost=S.cost))
        ivsp = pd.DataFrame(ivst)
        picks = []
        for b in BUDGETS:
            for rule in RULES:
                df = cascade(task, cs, es, "clean", b, rule, iv_mode=True)
                cost = c_cheap + df.rate.mean() * c_exp
                hv = A.hull_pair(ivsp, cost)[0]
                picks.append(dict(budget=b, rule=rule, iv_f1=df.f1.mean(), iv_rate=df.rate.mean(), iv_cost_T=cost / 1e12,
                                  iv_hull=hv, adv=df.f1.mean() - hv))
        P = pd.DataFrame(picks)
        P.to_csv(out / f"{task}_pick_table.csv", index=False)
        best = P.sort_values(["adv", "iv_cost_T"], ascending=[False, True]).iloc[0]
        b, rule = float(best.budget), best.rule
        T = {"pick": {"budget": b, "rule": rule, "iv_adv": float(best.adv)}, "beds": {}}
        for bed in beds:
            df = cascade(task, cs, es, bed, b, rule, random_gate=100, oracle=True)
            if df is None:
                continue
            fc = fold_mean(df, "f1")
            cost = c_cheap + df.rate.mean() * c_exp
            ucost = uc_cheap + df.rate.mean() * uc_exp
            h, hdesc = hull_at(task, bed, cost)
            hu, hudesc = hull_at(task, bed, ucost, ucost_view=True)
            ratio = ratio_of(task, bed)
            d = fc - h
            K_ = len(d)
            r_ = dict(bed=bed, f1=float(fc.mean()), rate=float(df.rate.mean()), cost_T=cost / 1e12, hull=float(h.mean()), hull_pair=hdesc,
                      diff=nb(d, ratio), boot=boot(d), folds_ge_1=int((d >= 1.0).sum()), folds_pos=int((d > 0).sum()), n_folds=K_,
                      fold_diffs=np.round(d, 4).tolist(), ratio=ratio,
                      unpriced=dict(hull=float(hu.mean()), pair=hudesc, diff=nb(fc - hu, ratio)),
                      random_gate=nb(fc - fold_mean(df, "f1_random_gate"), ratio), random_ts=nb(fc - fold_mean(df, "f1_random_ts"), ratio),
                      oracle=float(fold_mean(df, "f1_oracle").mean()), f1_random_gate=float(fold_mean(df, "f1_random_gate").mean()),
                      f1_random_ts=float(fold_mean(df, "f1_random_ts").mean()))
            # best single at <= cost chosen on inner validation of the condition family
            fam = "shift" if bed == "shift" else bed
            cands = []
            for name, S in SYS[task].items():
                if S.cost <= cost * (1 + 1e-12):
                    v = static_iv_f1(S, task, fam)
                    if v is not None:
                        cands.append((v, name))
            if cands:
                bs = max(cands)[1]
                a_ = static_fold_f1(SYS[task][bs], task, bed)
                r_["best_single"] = dict(system=bs, f1=float(a_.mean()), diff=nb(fc - a_, ratio))
            # mixed-precision statics (S2)
            r_["mixed"] = {}
            for mp in ("mp6", "mp8"):
                S = SYS[task].get(f"{mp}:own")
                a_ = None if S is None else static_fold_f1(S, task, bed)
                if a_ is not None:
                    r_["mixed"][mp] = dict(f1=float(a_.mean()), cost_T=S.cost / 1e12, minus_cascade=float(a_.mean() - fc.mean()))
            # secondary head (parent head on the cheap view)
            pv = SYS[task].get(f"{cheap}:w8head")
            if pv is not None:
                dp = cascade(task, pv, es, bed, b, rule)
                if dp is not None:
                    fp = fold_mean(dp, "f1")
                    hp, _ = hull_at(task, bed, c_cheap + dp.rate.mean() * c_exp)
                    r_["parent_head"] = dict(f1=float(fp.mean()), rate=float(dp.rate.mean()), diff=nb(fp - hp, ratio))
            # every (budget, rule) on this bed, descriptive
            allp = []
            for bb in BUDGETS:
                for rr in RULES:
                    dd = cascade(task, cs, es, bed, bb, rr)
                    cc = c_cheap + dd.rate.mean() * c_exp
                    hh, _ = hull_at(task, bed, cc)
                    allp.append(dict(task=task, bed=bed, budget=bb, rule=rr, f1=fold_mean(dd, "f1").mean(), rate=dd.rate.mean(),
                                     cost_T=cc / 1e12, hull=hh.mean(), diff=fold_mean(dd, "f1").mean() - hh.mean()))
            casc_rows += allp
            T["beds"][bed] = r_
        # H3 humility
        h3 = []
        steps = []
        if task != "tau":
            for kind in ("bg", "pink", "white"):
                seq = ["clean"] + [f"{kind}__{s}db" for s in (20, 10, 5, 0)]
                steps += list(zip(seq[:-1], seq[1:]))
            seqc = ["clean"] + CORR
            for kind in ("reverb", "clip", "lowpass"):
                cs_ = [c for c in CORR if c.startswith(kind)]
                steps += [("clean", cs_[0])] + list(zip(cs_[:-1], cs_[1:]))
        else:
            steps = [("device:a", f"device:{d}") for d in ("b", "c", "s1", "s2", "s3", "s4", "s5", "s6")]
        rates = {}

        def rate_of(bedname):
            if bedname not in rates:
                if task == "tau":
                    dfr = cascade(task, cs, es, bedname, b, rule)
                else:
                    try:
                        ok = all(cs.get(s, bedname) is not None and np.isfinite(cs.get(s, bedname)["z"]).all() for s in TASKS[task]["seeds"])
                    except Exception:
                        ok = False
                    dfr = cascade(task, cs, cs, bedname, b, "B") if ok else None
                rates[bedname] = None if dfr is None else fold_mean(dfr, "rate")
            return rates[bedname]
        for s0, s1 in steps:
            r0, r1 = rate_of(s0), rate_of(s1)
            if r0 is None or r1 is None:
                continue
            d = r1 - r0
            h3.append(dict(task=task, step=f"{s0} -> {s1}", rate_from=r0.mean(), rate_to=r1.mean(), diff=d.mean(),
                           folds_pos=int((d > 0).sum()), n_folds=len(d), p_nb=nb(d, 0.0)["p"] if d.std() > 0 else 0.0,
                           rises=bool(d.mean() > 0 and (d > 0).sum() >= maj(len(d))),
                           additive=not any(c in s1 for c in ("reverb", "clip", "lowpass"))))
        T["h3"] = h3
        # H4 rounding disagreement
        bed = primary_bed[task]
        s1s, s2s = SYS[task].get(f"{cheap}sr1"), None
        d1, d2 = R / "scores" / task / f"{cheap}sr1@{cheap}", R / "scores" / task / f"{cheap}sr2@{cheap}"
        if d1.exists() and d2.exists():
            L1 = tau_my(d1, f"{cheap}sr1") if task == "tau" else short_dir(d1, f"{cheap}sr1")
            L2 = tau_my(d2, f"{cheap}sr2") if task == "tau" else short_dir(d2, f"{cheap}sr2")
            cond, sel = bed_src(task, bed)
            au = []
            for f in folds_of(task):
                am, aj = [], []
                for s in TASKS[task]["seeds"]:
                    Dc, De, Q1, Q2 = cs.get(s, cond), es.get(s, cond), L1(s, cond), L2(s, cond)
                    if Q1 is None or Q2 is None:
                        break
                    r = test_rows(Dc, f, sel)
                    y = Dc["y"][r]
                    pc, pe = softmax(Dc["z"][r]), softmax(De["z"][r])
                    tgt = (pc.argmax(1) != y) & (pe.argmax(1) == y)
                    p1, p2 = softmax(Q1["z"][r]), softmax(Q2["z"][r])
                    mm = (p1 + p2) / 2
                    js = 0.5 * (np.sum(p1 * np.log((p1 + 1e-12) / (mm + 1e-12)), 1) + np.sum(p2 * np.log((p2 + 1e-12) / (mm + 1e-12)), 1))
                    if 0 < tgt.sum() < len(tgt):
                        am.append(roc_auc_score(tgt, -margin(pc)))
                        aj.append(roc_auc_score(tgt, js))
                if am:
                    au.append((np.mean(am), np.mean(aj)))
            if au:
                au = np.array(au)
                dd = au[:, 1] - au[:, 0]
                T["h4"] = dict(auroc_margin=float(au[:, 0].mean()), auroc_js=float(au[:, 1].mean()), diff=nb(dd, ratio_of(task, bed)),
                               folds_pos=int((dd > 0).sum()), n_folds=len(dd),
                               passes=bool(dd.mean() >= 0.02 and (dd > 0).sum() >= maj(len(dd))))
        res["tasks"][task] = T
        print(task, json.dumps(T["pick"]), {k: (round(v["f1"], 2), round(v["hull"], 2), round(v["diff"]["mean"], 2)) for k, v in T["beds"].items()}, flush=True)
    pd.DataFrame(all_static_rows).to_csv(out / "statics.csv", index=False)
    pd.DataFrame(casc_rows).to_csv(out / "cascade_all_points.csv", index=False)
    # verdicts
    main3 = [t for t in ("esc50", "urban", "tau") if t in res["tasks"]]
    V = {"tasks_available": main3}
    for H, bedkey in (("H1", "primary"), ("H2", "clean")):
        ps, rows = [], {}
        for t in main3:
            bed = primary_bed[t] if bedkey == "primary" else "clean"
            r_ = res["tasks"][t]["beds"].get(bed)
            if r_ is None:
                continue
            rows[t] = r_
            ps.append(r_["diff"]["p"])
        hp = holm(ps) if ps else []
        passes = 0
        V[H] = {}
        for (t, r_), p_h in zip(rows.items(), hp):
            ok = r_["diff"]["mean"] >= 1.0 and r_["folds_ge_1"] >= maj(r_["n_folds"]) and p_h < 0.05
            passes += ok
            V[H][t] = dict(diff=r_["diff"]["mean"], lo=r_["diff"]["lo"], hi=r_["diff"]["hi"], boot=r_["boot"], p_nb=r_["diff"]["p"],
                           p_holm=float(p_h), p_family17=float(min(1.0, p_h * FAMILY)), folds_ge_1=r_["folds_ge_1"], n_folds=r_["n_folds"],
                           task_pass=bool(ok))
        V[H]["passes"] = bool(passes >= 2)
    h3all = [x for t in main3 for x in res["tasks"][t].get("h3", []) if x.get("additive", True)]
    V["H3"] = dict(holds=bool(h3all) and all(x["rises"] for x in h3all), exceptions=[f"{x['task']} {x['step']}" for x in h3all if not x["rises"]],
                   n_steps=len(h3all))
    h4 = {t: res["tasks"][t].get("h4", "pending: SR passes not scored yet") for t in main3}
    V["H4"] = dict(per_task=h4, passes=sum(bool(isinstance(v, dict) and v["passes"]) for v in h4.values()) >= 2)
    V["S1"] = dict(fires=all(res["tasks"][t]["beds"][primary_bed[t]]["diff"]["mean"] <= 0.3 for t in main3) if main3 else None)
    s2 = {}
    for t in main3:
        r_ = res["tasks"][t]["beds"][primary_bed[t]]
        cc = r_["cost_T"]
        s2[t] = any(m["cost_T"] <= cc and m["minus_cascade"] >= -0.3 for m in r_["mixed"].values())
    V["S2"] = dict(per_task=s2, fires=all(s2.values()) if s2 else None)
    res["verdict"] = V
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "verdict.json").write_text(json.dumps(V, indent=1, default=float))
    print(json.dumps(V, indent=1, default=float))


if __name__ == "__main__":
    main()
