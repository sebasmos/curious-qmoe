"""Gate quality by precision, from stored per-clip logits only (no extraction, no scoring, no training).

Question: is the margin gate's quality the same at every backbone precision and every head precision?

Reads the score files of idea 6 Stage 2, 3b, 3c, 3c-unpadded, option B and C and idea 5 Stage 2 (read only),
writes one per-fold table (seeds averaged within folds) and the paired differences against FP32 to OUT.

Conventions copied from Stage 2's analyze.py (same softmax, margin, 15-bin ECE, temperature fit, threshold rule):
  * temperature T and the escalation thresholds are fitted once per (seed, fold) on the clean inner-validation
    logits and applied unchanged to every condition;
  * threshold for budget b = the margin cut that escalates a share b of the clean inner-validation clips;
  * the field mixture = clean for half of the clips, the nine cells (bg, pink, white at 10, 5, 0 dB) for the rest,
    positions from audio_shift.field_assignment with the test seed 5001.

    OMP_NUM_THREADS=2 nice -n 10 python gate_by_precision.py --out OUT [--fold-table OUT/fold_metrics.csv.gz]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

RES = Path.home() / "Documents/qmoe-results"
S3B = RES / "2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db"
S3C = RES / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1"
S2 = RES / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
I5 = RES / "2026-10-20_2359_redesign-idea5-stage2_code-3c18c38"
UNP = RES / "2026-10-23_1918_redesign-idea6-stage3c-unpadded_code-cbf6be6"
OB = RES / "2026-10-23_2133_redesign-idea6-optionB_code-f320295"
OC = RES / "2026-10-23_2133_redesign-idea6-optionC_code-d6e7913"

FIELD_CELLS = tuple((k, s) for k in ("bg", "pink", "white") for s in (10, 5, 0))
FIELD_SEED = 5001  # test assignment, Stage 2's audio_shift.FIELD_SEEDS["test"]
ECE_BINS = 15
BUDGETS = {"b10": 0.10, "b20": 0.20, "b30": 0.30}
KINDS = ("bg", "pink", "white")
SNRS = (20, 10, 5, 0)
ESC_CONDS = ["clean"] + [f"{k}{s}" for k in KINDS for s in SNRS] + ["field"]
ALL_CONDS = ["clean"] + [f"{k}{s}" for k in KINDS for s in (10, 5, 0)]  # the ten conditions every system has
TAU_DEV = ["a", "b", "c", "s1", "s2", "s3", "s4", "s5", "s6"]
TAU_CONDS = ["clean"] + TAU_DEV[1:] + ["shift"]
GATES = ["margin", "entropy", "energy", "tsmax"]
SEEDS = {"esc50": range(5), "esc5": range(5), "urban": range(3), "tau": range(3)}
NCLS = {"esc50": 50, "esc5": 5, "urban": 10, "tau": 10}


FILE_COND = {c: ("clean" if c == "clean" else f"{c.rstrip('0123456789')}__{c[len(c.rstrip('0123456789')):]}db") for c in ESC_CONDS if c != "field"}


# ── metrics (Stage 2's analyze.py) ─────────────────────────────────────────

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def thr_rate(m_iv, rate):
    u = np.unique(m_iv)
    c = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    rates = (m_iv[None, :] < c[:, None]).mean(1)
    return c[np.lexsort((rates, np.abs(rates - rate)))[0]]


def f1_fast(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))


def ece_conf(conf, correct, bins=ECE_BINS):
    b = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    n = np.bincount(b, minlength=bins)
    sc = np.bincount(b, weights=conf, minlength=bins)
    sa = np.bincount(b, weights=correct.astype(float), minlength=bins)
    ok = n > 0
    return float(np.sum(np.abs(sa[ok] - sc[ok])) / len(conf))


def fit_temperature(z, y):
    z = z.astype(np.float64)

    def nll(lt):
        q = z / np.exp(lt)
        q = q - q.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(q).sum(1)) - q[np.arange(len(y)), y]))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def auroc(wrong, score):
    return float(roc_auc_score(wrong, score)) if 0 < wrong.sum() < len(wrong) else np.nan


def field_assignment(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    out = ["clean"] * n
    for r, i in enumerate(order[(n + 1) // 2:]):
        k, s = FIELD_CELLS[r % len(FIELD_CELLS)]
        out[i] = f"{k}{s}"
    return np.array(out)


def cell_metrics(z, y, K, ivz, ivy):
    """All metrics of one (seed, fold, condition) cell. z logits of the test rows, iv* the clean inner validation."""
    p = softmax(z)
    pred = p.argmax(1)
    wrong = pred != y
    T = fit_temperature(ivz, ivy)
    pt = softmax(z / T)
    m = margin(p)
    out = {"n": len(y), "err": float(wrong.mean()), "f1": f1_fast(y, pred, K), "T": T,
           "ece": ece_conf(p.max(1), ~wrong), "ece_T": ece_conf(pt.max(1), pt.argmax(1) == y)}
    ent = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    out["auroc_margin"] = auroc(wrong, 1 - m)
    out["auroc_entropy"] = auroc(wrong, ent)
    out["auroc_energy"] = auroc(wrong, -logsumexp(z, axis=1))
    out["auroc_tsmax"] = auroc(wrong, 1 - pt.max(1))
    order = np.argsort(m, kind="stable")
    for frac in (10, 20, 30):
        k = int(round(len(y) * frac / 100))
        out[f"caught{frac}"] = float(wrong[order[:k]].sum() / wrong.sum()) if wrong.sum() else np.nan
    pv = softmax(ivz)
    mv = margin(pv)
    for b, rate in BUDGETS.items():
        out[f"rate_{b}"] = float((m < thr_rate(mv, rate)).mean())
    return out


# ── system registry ────────────────────────────────────────────────────────

def spec(task, model, kind, backbone, head, length, root_fn):
    return dict(task=task, model=model, fam=kind, backbone=backbone, head=head, length=length, path=root_fn,
                sid=f"{task}|{model}|{backbone}|{head}")


def build_specs():
    out = []
    heads4 = ("esc", "8", "4", "bitnet")
    for task in ("esc50", "esc5", "urban"):
        for model, root in (("mn10", S3B), ("mn20", S3C)):
            for bb in ("fp32", "w8a8"):
                for h in heads4:
                    out.append(spec(task, model, "cnn", {"fp32": "FP32", "w8a8": "Q8sim"}[bb], h, "padded" if task == "urban" else "5s",
                                    ("cnn", root, f"scores_cnn/{task}/{model}_{task}_s{{s}}/{h}/{model}_{bb}__{{c}}.npz")))
        # real int8, way (b): heads retrained on the int8 embeddings (scaler refit with them)
        for model, hs in {"esc50": {"mn10": "4", "mn20": "4"}, "esc5": {"mn10": "esc", "mn20": "esc"}, "urban": {"mn10": "esc"}}[task].items():
            out.append(spec(task, model, "cnn", "INT8", hs, "padded" if task == "urban" else "5s",
                            ("cnn", S3C, f"scores_int8b/{task}/{model}r_{task}_s{{s}}/{hs}/{model}_int8__{{c}}.npz")))
    for bb in ("fp32", "w8a8"):  # urban at exact clip length, heads retrained on exact-length embeddings
        for h in heads4:
            out.append(spec("urban", "mn10exact", "cnn", {"fp32": "FP32", "w8a8": "Q8sim"}[bb], h, "exact",
                            ("cnn", UNP, f"scores_unpr/urban/mn10unp_urban_s{{s}}/{h}/mn10_{bb}__{{c}}.npz")))
    for model in ("mn10", "mn20"):
        for bb in ("fp32", "w8a8"):
            for h in heads4:
                out.append(spec("tau", model, "cnn", {"fp32": "FP32", "w8a8": "Q8sim"}[bb], h, "10s",
                                ("tau", OB, f"scores/{model}_tau_s{{s}}/{h}/{model}_{bb}__all.npz")))
    out.append(spec("tau", "mn20", "cnn", "INT8", "8", "10s", ("tau", OB, "scores_int8/mn20_tau_s{s}/8/mn20_int8__all.npz")))
    # AST
    q1024 = {"fp32": "FP32", "w8": "Q8", "w6": "Q6", "w5": "Q5", "w4g128": "Q4", "w4": "Q4pc", "w3g64": "Q3"}
    qreal = {"fp32": "FP32", "w8g64": "Q8", "w4g128": "Q4", "w4": "Q4pc", "w3g64": "Q3"}
    for task in ("esc50", "esc5", "urban"):
        real = 400 if task == "urban" else 512
        for t, q in q1024.items():
            for h in ("esc", "4"):
                out.append(spec(task, "ast1024", "ast", q, h, "1024",
                                ("ast", I5, f"scores/{task}/ast{task}_s{{s}}/{h}/{t}__{{c}}.npz")))
        for t, q in qreal.items():
            for h in ("esc", "4"):
                out.append(spec(task, "astReal", "ast", q, h, str(real),
                                ("ast", S2, f"scores_ast_short/{task}/astshort_{task}_s{{s}}/{h}/{t}__{{c}}.npz")))
        for h in ("esc", "4"):  # option C: heads trained on the Q3 embeddings themselves
            out.append(spec(task, "astReal", "ast", "Q3ownhead", h, str(real),
                            ("ast", OC, f"scores_own/{task}/own_w3g64_{task}_s{{s}}/{h}/w3g64__{{c}}.npz")))
    for t, q in {"fp32": "FP32", "w8g64": "Q8", "w4g128": "Q4", "n4": "Q4n"}.items():
        for h in ("esc", "4"):
            out.append(spec("tau", "astReal", "ast", q, h, "10s", ("tau", OB, f"scores/ast_tau_s{{s}}/{h}/{t}__all.npz")))
    return out


# ── loading ────────────────────────────────────────────────────────────────

def load_npz(path, tau):
    z = np.load(path)
    iv = {int(k[7:]): (z[k], z[f"iv_logits_{k[7:]}"].astype(np.float64)) for k in z.files if k.startswith("iv_idx_")}
    fold = z["fold" if tau else "test_fold"].astype(int)
    d = {"y": z["y"].astype(np.int64), "fold": fold, "z": z["test_logits"].astype(np.float64), "iv": iv}
    if tau:
        d["device"] = z["device"]
    return d


def seed_rows(sp, seed, ref_y):
    """Metric rows (one per condition and fold) for one seed of one system. Returns [] if files are missing."""
    kind, root, tmpl = sp["path"]
    tau = kind == "tau"
    task, K = sp["task"], NCLS[sp["task"]]
    rows = []
    if tau:
        p = root / tmpl.format(s=seed)
        if not p.exists():
            return None
        d = load_npz(p, True)
        check_ref(task, d, ref_y)
        conds = {"clean": d["device"] == "a", "shift": d["device"] != "a"}
        for dv in TAU_DEV[1:]:
            conds[dv] = d["device"] == dv
        for cname, rowmask in conds.items():
            for f in sorted(d["iv"]):
                m = rowmask & (d["fold"] == f)
                if m.sum() < 20:
                    continue
                ivi, ivz = d["iv"][f]
                r = cell_metrics(d["z"][m], d["y"][m], K, ivz, d["y"][ivi])
                rows.append(dict(cond=cname, seed=seed, fold=f, **r))
        return rows
    data = {}
    for c, fc in FILE_COND.items():
        p = root / tmpl.format(s=seed, c=fc)
        if p.exists():
            data[c] = load_npz(p, False)
    if "clean" not in data:
        return None
    base = data["clean"]
    check_ref(task, base, ref_y)
    n = len(base["y"])
    at = field_assignment(n, FIELD_SEED)
    if all(c in data for c in ["clean"] + [f"{k}{s}" for k, s in FIELD_CELLS]):
        zf = np.full_like(base["z"], np.nan)
        for c in ["clean"] + [f"{k}{s}" for k, s in FIELD_CELLS]:
            zf[at == c] = data[c]["z"][at == c]
        data["field"] = {**base, "z": zf}
    if not all(np.isfinite(v[1]).all() for v in base["iv"].values()) or not np.isfinite(base["z"]).all():
        raise SystemExit(f"{sp['sid']} seed {seed}: clean logits not finite")
    for c, d in data.items():
        if not np.isfinite(d["z"]).all():  # a file that holds only some clips (idea 5's field cells): no per-cell metric
            SKIPPED.add((sp["sid"], c))
            continue
        for f in sorted(base["iv"]):
            m = d["fold"] == f
            ivi, ivz = base["iv"][f]
            z = d["z"][m]
            r = cell_metrics(z, d["y"][m], K, ivz, d["y"][ivi])
            rows.append(dict(cond=c, seed=seed, fold=f, **r))
    return rows


_REF = {}
SKIPPED = set()


def check_ref(task, d, ref_y):
    """Every system of a task must hold the same clips in the same order with the same fold assignment."""
    key = (task, "y")
    if key not in _REF:
        _REF[key] = (d["y"].copy(), d["fold"].copy())
    y0, f0 = _REF[key]
    if not (np.array_equal(y0, d["y"]) and np.array_equal(f0, d["fold"])):
        raise SystemExit(f"{task}: labels or folds differ between score files")


# ── aggregation and statistics ─────────────────────────────────────────────

MET = ["f1", "ece", "ece_T", "auroc_margin", "auroc_entropy", "auroc_energy", "auroc_tsmax",
       "caught10", "caught20", "caught30", "rate_b10", "rate_b20", "rate_b30", "err", "T"]


def fold_table(specs, out_dir):
    allrows, status = [], []
    for i, sp in enumerate(specs):
        t0 = time.time()
        got = 0
        for s in SEEDS[sp["task"]]:
            r = seed_rows(sp, s, None)
            if r is None:
                continue
            got += 1
            for x in r:
                x.update(sid=sp["sid"])
            allrows += r
        status.append({"sid": sp["sid"], "seeds_found": got, "seeds_expected": len(SEEDS[sp["task"]]), "sec": round(time.time() - t0, 1)})
        print(f"[{i + 1}/{len(specs)}] {sp['sid']} seeds {got}/{len(SEEDS[sp['task']])} {time.time() - t0:.1f}s", flush=True)
    df = pd.DataFrame(allrows)
    fold = df.groupby(["sid", "cond", "fold"], as_index=False)[MET + ["n"]].mean()  # seeds averaged within folds
    fold.to_csv(out_dir / "fold_metrics.csv.gz", index=False)
    pd.DataFrame(status).to_csv(out_dir / "load_status.csv", index=False)
    pd.DataFrame(sorted(SKIPPED), columns=["sid", "cond"]).to_csv(out_dir / "skipped_incomplete_files.csv", index=False)
    return fold


def add_derived(fold, specs):
    """ALL pseudo-condition (mean over the ten shared conditions, or nine TAU devices; UrbanSound8K AST files hold only
    clean, bg 10 and bg 0 dB for the full set of clips, so its ALL is the mean of those three) and the escalation rises."""
    meta = pd.DataFrame(specs)[["sid", "task"]]
    f = fold.merge(meta, on="sid")
    parts = []
    for task, g in f.groupby("task"):
        cs = [("clean" if d == "a" else d) for d in TAU_DEV] if task == "tau" else ALL_CONDS
        sub = g[g.cond.isin(cs)]
        a = sub.groupby(["sid", "fold"], as_index=False)[MET + ["n"]].mean()  # mean over the conditions a system has
        cnt = sub.groupby(["sid", "fold"]).cond.nunique().reset_index(name="k_conds")
        a = a.merge(cnt, on=["sid", "fold"])
        a = a[a.k_conds >= 3]
        a["cond"] = "ALL"
        a["task"] = task
        parts.append(a)
    f = pd.concat([f] + parts, ignore_index=True)
    # rises of the escalation rate over clean
    rate = f.pivot_table(index=["sid", "fold"], columns="cond", values=[f"rate_{b}" for b in BUDGETS])
    rises = []
    for b in BUDGETS:
        r = rate[f"rate_{b}"]
        d = pd.DataFrame(index=r.index)
        for k in KINDS:
            if f"{k}0" in r:
                d[f"rise_{b}_{k}0"] = r[f"{k}0"] - r["clean"]
        if all(f"{k}0" in r for k in KINDS):
            d[f"rise_{b}_mean0"] = d[[f"rise_{b}_{k}0" for k in KINDS]].mean(1, skipna=False)
        if "field" in r:
            d[f"rise_{b}_field"] = r["field"] - r["clean"]
        if "shift" in r:
            d[f"rise_{b}_shift"] = r["shift"] - r["clean"]
        rises.append(d)
    rises = pd.concat(rises, axis=1).reset_index()
    return f, rises


def paired(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    n = len(d)
    if n < 3:
        return dict(n=n, diff=np.nan, lo=np.nan, hi=np.nan, p=np.nan, decided=np.nan)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    p = 2 * stats.t.sf(abs(m) / se, n - 1) if se > 0 else (1.0 if m == 0 else 0.0)
    return dict(n=n, diff=m, lo=m - t * se, hi=m + t * se, p=p, decided=bool(m - t * se > 0 or m + t * se < 0))


def comparisons(specs):
    """(system, reference, panel) triples: backbone against FP32 at the same head, head against FP32 head at the same backbone."""
    by = {(s["task"], s["model"], s["backbone"], s["head"]): s["sid"] for s in specs}
    out = []
    for s in specs:
        k = (s["task"], s["model"], s["backbone"], s["head"])
        if s["backbone"] != "FP32":
            ref = by.get((s["task"], s["model"], "FP32", s["head"]))
            if ref:
                out.append((s["sid"], ref, "backbone"))
        if s["head"] != "esc":
            ref = by.get((s["task"], s["model"], s["backbone"], "esc"))
            if ref:
                out.append((s["sid"], ref, "head"))
    return out


def stats_table(f, rises, specs):
    wide = {sid: g for sid, g in f.groupby("sid")}
    rw = {sid: g.set_index("fold") for sid, g in rises.groupby("sid")}
    rows = []
    for sid, ref, panel in comparisons(specs):
        for cond, g in wide[sid].groupby("cond"):
            gr = wide[ref][wide[ref].cond == cond]
            if gr.empty:
                continue
            a, b = g.set_index("fold"), gr.set_index("fold")
            idx = a.index.intersection(b.index)
            for m in MET:
                if m == "T":
                    continue
                rows.append(dict(panel=panel, sid=sid, ref=ref, cond=cond, metric=m, mean=a.loc[idx, m].mean(),
                                 ref_mean=b.loc[idx, m].mean(), **paired(a.loc[idx, m].to_numpy(), b.loc[idx, m].to_numpy())))
        ra, rb = rw.get(sid), rw.get(ref)
        if ra is not None and rb is not None:
            idx = ra.index.intersection(rb.index)
            for m in [c for c in ra.columns if c.startswith("rise_") and c in rb.columns]:
                rows.append(dict(panel=panel, sid=sid, ref=ref, cond="rise", metric=m, mean=ra.loc[idx, m].mean(),
                                 ref_mean=rb.loc[idx, m].mean(), **paired(ra.loc[idx, m].to_numpy(), rb.loc[idx, m].to_numpy())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reuse", action="store_true", help="read OUT/fold_metrics.csv.gz instead of recomputing it")
    ap.add_argument("--only-task", nargs="*")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    specs = build_specs()
    if args.only_task:
        specs = [s for s in specs if s["task"] in args.only_task]
    pd.DataFrame(specs).drop(columns="path").to_csv(args.out / "systems.csv", index=False)
    fold = pd.read_csv(args.out / "fold_metrics.csv.gz") if args.reuse else fold_table(specs, args.out)
    f, rises = add_derived(fold, specs)
    f.to_csv(args.out / "fold_metrics_with_all.csv.gz", index=False)
    rises.to_csv(args.out / "fold_rises.csv", index=False)
    st = stats_table(f, rises, specs)
    st.to_csv(args.out / "paired_vs_fp32.csv", index=False)
    print("done", len(f), len(st))


if __name__ == "__main__":
    main()
