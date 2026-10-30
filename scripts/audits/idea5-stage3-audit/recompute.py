"""Stage 3 audit, check 1 and 5: independent recomputation of the idea 5 Stage 2
static systems, primary cascades, costs and paired tests from the saved head
scores (scores/<dataset>/<tag>/<head>/<config>__<condition>.npz). Shares no
code with scripts/redesign/idea5/analyze_stage2.py or front.py.

Writes, per dataset, to OUT/<dataset>/:
  static_pairs.csv    every (config, head, condition, seed, fold) macro F1 (sklearn)
  cascade_pairs.csv   every primary cascade point (cheap -> its W8 parent, FP32 head,
                      margin gate, b05 b10 b20 b30 match) per (seed, fold)
  points.csv          per point: mean F1, mean bit-ops, comparator, paired tests

Costs are computed here from the AST architecture (12 layers, D 768, FFN 3072,
1214 tokens, 1212 patches), not read from the extraction sidecars.

    python recompute.py --stage2 STAGE2_DIR --dataset esc50 --out OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

# ---- costs from first principles -------------------------------------------
L, D, FF, TOK, PATCH = 12, 768, 3072, 1214, 1212
LIN_W = L * (D * 3 * D + D * D + D * FF + FF * D)            # 84,934,656 Linear weights
LIN_MACS = LIN_W * TOK
ATT_MACS_HALF = TOK * TOK * D * L                               # Q K^T, and the same for P V
PATCH_MACS = D * 1 * 16 * 16 * PATCH


def backbone_bitops(cfg: str, group_overhead_bits: float = 0.0) -> float:
    """MACs x weight bits x activation bits. A8 for every config but fp32; Q K^T at
    8x8, P V at 16x8 (16-bit probabilities); patch conv at 32x32.
    group_overhead_bits: bit-ops charged per extra group rescale MAC (one per token,
    output row and group beyond the first) for group-wise configs; 0 = Stage 2's rule."""
    if cfg == "fp32":
        return LIN_MACS * 32 * 32 + 2 * ATT_MACS_HALF * 32 * 32 + PATCH_MACS * 32 * 32
    wb = int(cfg[1])
    g = int(cfg.split("g")[1]) if "g" in cfg else None
    b = LIN_MACS * wb * 8 + ATT_MACS_HALF * 8 * 8 + ATT_MACS_HALF * 16 * 8 + PATCH_MACS * 32 * 32
    if g and group_overhead_bits:
        extra_macs = TOK * sum(o * (i // g - 1) for o, i in [(3 * D, D), (D, D), (FF, D), (D, FF)]) * L
        b += extra_macs * group_overhead_bits
    return float(b)


def head_bitops(head: str, n_cls: int) -> float:
    macs = D * 640 + 640 * 320 + 320 * n_cls
    return float(macs * (32 * 32 if head == "esc" else 4 * 8))


def parent(cfg: str) -> str:
    return "w8" + (cfg[cfg.index("g"):] if "g" in cfg else "")


# ---- field mixture (re-implemented from the README's description) ---------
CELLS = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]


def field_assign(n: int, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(n)
    a = np.array(["clean"] * n, dtype=object)
    for r, i in enumerate(perm[(n + 1) // 2:]):
        a[i] = CELLS[r % 9]
    return a


# ---- scores ----------------------------------------------------------------
def softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Run:
    def __init__(self, d: Path):
        self.d, self.c = d, {}

    def exists(self, cfg, cond):
        if cond == "field":
            return all((self.d / f"{cfg}__{c}.npz").exists() for c in ["clean"] + CELLS)
        return (self.d / f"{cfg}__{cond}.npz").exists()

    def raw(self, cfg, cond):
        k = (cfg, cond)
        if k not in self.c:
            z = np.load(self.d / f"{cfg}__{cond}.npz")
            folds = sorted(int(f[7:]) for f in z.files if f.startswith("iv_idx_"))
            self.c[k] = dict(y=z["y"].astype(int), tf=z["test_fold"].astype(int), p=softmax(z["test_logits"]),
                             iv={f: (z[f"iv_idx_{f}"].astype(int), softmax(z[f"iv_logits_{f}"])) for f in folds})
        return self.c[k]

    def get(self, cfg, cond):
        if cond != "field":
            return self.raw(cfg, cond)
        k = (cfg, "field")
        if k not in self.c:
            base = self.raw(cfg, "clean")
            n = len(base["y"])
            at, av = field_assign(n, 5001), field_assign(n, 5002)
            p = np.full_like(base["p"], np.nan)
            iv = {f: (idx, np.full_like(pr, np.nan)) for f, (idx, pr) in base["iv"].items()}
            for c in ["clean"] + CELLS:
                r = self.raw(cfg, c)
                p[at == c] = r["p"][at == c]
                for f, (idx, pr) in iv.items():
                    s = av[idx] == c
                    pr[s] = r["iv"][f][1][s]
            assert np.isfinite(p).all() and all(np.isfinite(pr).all() for _, pr in iv.values())
            self.c[k] = dict(y=base["y"], tf=base["tf"], p=p, iv=iv)
        return self.c[k]

    def complete(self, cfg, cond):
        r = self.get(cfg, cond)
        return np.isfinite(r["p"]).all() and all(np.isfinite(pr).all() for _, pr in r["iv"].values())


def f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def f1_fast(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    present = den > 0
    return float(np.mean(2 * tp[present] / den[present]))


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def thr_rate(m_iv, rate):
    """Escalate the clips with margin below thr. Choose among the escalation counts
    reachable without splitting ties the one whose rate is closest to `rate`
    (ties: fewer escalations). Returns the threshold."""
    u = np.unique(m_iv)
    cands = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    rates = np.array([(m_iv < t).mean() for t in cands])
    order = np.lexsort((rates, np.abs(rates - rate)))
    return cands[order[0]]


def thr_match(m_iv, y, pc, pe, target, k):
    u = np.unique(m_iv)
    cands = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    rows = []
    for t in cands:
        e = m_iv < t
        rows.append((e.mean(), f1_fast(y, np.where(e, pe, pc), k), t))
    for r, f, t in sorted(rows, key=lambda r: (r[0], -r[1])):
        if f >= target - 1e-12:
            return t
    return max(rows, key=lambda r: (r[1], -r[0]))[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group-overhead-bits", type=float, default=0.0)
    a = ap.parse_args()
    sc = a.stage2 / "scores" / a.dataset
    tags = sorted(p.name for p in sc.iterdir() if p.is_dir() and "headw4" not in p.name)
    heads = ["esc", "4"]
    runs = {(int(t.split("_s")[1]), h): Run(sc / t / h) for t in tags for h in heads}
    r0 = next(iter(runs.values()))
    files = sorted(p.stem for p in r0.d.glob("*.npz"))
    configs = sorted({f.split("__")[0] for f in files})
    conds = sorted({f.split("__", 1)[1] for f in files})
    cheap = ["w4", "n4", "w4g128", "n4g128", "w3g64", "n3g64"]
    if all(r0.exists(c, "field") for c in cheap + ["w8", "w8g128", "w8g64"]):
        conds.append("field")
    # a condition counts only if every row is scored (UrbanSound8K field cells are partial)
    conds = [c for c in conds if r0.exists("n3g64", c) and r0.complete("n3g64", c)]
    print(a.dataset, "tags", tags, "conditions", conds, flush=True)
    n_cls = r0.raw("fp32", "clean")["p"].shape[1]
    ob = a.group_overhead_bits

    srows, crows = [], []
    for (seed, h), run in runs.items():
        hb = head_bitops(h, n_cls)
        for cond in conds:
            for cfg in configs:
                if not run.exists(cfg, cond) or not run.complete(cfg, cond):
                    continue
                r = run.get(cfg, cond)
                for fold in sorted(r["iv"]):
                    m = r["tf"] == fold
                    srows.append(dict(cond=cond, config=cfg, head=h, seed=seed, fold=fold,
                                      f1=f1(r["y"][m], r["p"][m].argmax(1)),
                                      bitops=backbone_bitops(cfg, ob) + hb))
            if h != "esc":
                continue
            for c in cheap:
                par = parent(c)
                if not (run.exists(c, cond) and run.exists(par, cond)):
                    continue
                rc, rp = run.get(c, cond), run.get(par, cond)
                assert (rc["y"] == rp["y"]).all() and (rc["tf"] == rp["tf"]).all()
                for fold in sorted(rc["iv"]):
                    idx, pci = rc["iv"][fold]
                    idx2, ppi = rp["iv"][fold]
                    assert (idx == idx2).all()
                    te = np.flatnonzero(rc["tf"] == fold)
                    assert len(np.intersect1d(te, idx)) == 0, "inner validation overlaps test"
                    yi, y = rc["y"][idx], rc["y"][te]
                    mi, mt = margin(pci), margin(rc["p"][te])
                    ths = {f"b{int(100 * v):02d}": thr_rate(mi, v) for v in (0.05, 0.10, 0.20, 0.30)}
                    ths["match"] = thr_match(mi, yi, pci.argmax(1), ppi.argmax(1),
                                             f1_fast(yi, ppi.argmax(1), n_cls), n_cls)
                    for pt, t in ths.items():
                        e = mt < t
                        pred = np.where(e, rp["p"][te].argmax(1), rc["p"][te].argmax(1))
                        rate = e.mean()
                        crows.append(dict(cond=cond, cheap=c, point=pt, seed=seed, fold=fold, f1=f1(y, pred),
                                          rate=rate, iv_rate=(mi < t).mean(), thr=t,
                                          bitops=backbone_bitops(c, ob) + hb + rate * (backbone_bitops(par, ob) + hb)))
        print(f"  seed {seed} head {h} done", flush=True)

    S, C = pd.DataFrame(srows), pd.DataFrame(crows)
    out = a.out / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "static_pairs.csv", index=False)
    C.to_csv(out / "cascade_pairs.csv", index=False)

    # points and comparisons
    rows = []
    for cond in conds:
        s = S[S.cond == cond]
        sp = s.groupby(["config", "head"]).agg(f1=("f1", "mean"), bitops=("bitops", "mean"), n=("f1", "size")).reset_index()
        for (c, pt), g in C[C.cond == cond].groupby(["cheap", "point"]):
            cost = g.bitops.mean()
            ok = sp[sp.bitops <= cost * (1 + 1e-9)].sort_values(["f1", "bitops"], ascending=[False, True])
            comp = ok.iloc[0]
            gs = s[(s.config == comp.config) & (s["head"] == comp["head"])].set_index(["seed", "fold"]).f1
            gc = g.set_index(["seed", "fold"]).f1
            keys = sorted(set(gs.index) & set(gc.index))
            x, z = gc.loc[keys].to_numpy(), gs.loc[keys].to_numpy()
            d = x - z
            pt_ = stats.ttest_rel(x, z).pvalue
            pw = stats.wilcoxon(x, z).pvalue
            # fold-level: average the seeds within each fold first (test clips are shared across seeds)
            df = pd.Series(d, index=pd.MultiIndex.from_tuples(keys)).groupby(level=1).mean()
            pt_fold = stats.ttest_1samp(df.to_numpy(), 0).pvalue
            pw_fold = stats.wilcoxon(df.to_numpy()).pvalue
            rows.append(dict(dataset=a.dataset, cond=cond, cheap=c, point=pt, f1=g.f1.mean(), bitops=cost,
                             rate=g.rate.mean(), iv_rate=g.iv_rate.mean(), comparator=f"{comp.config}+{comp['head']}",
                             comp_f1=comp.f1, comp_bitops=comp.bitops, n_pairs=len(keys), diff_pts=100 * d.mean(),
                             n_pos=int((d > 0).sum()), p_t=pt_, p_w=pw, n_folds=len(df), p_t_fold=pt_fold,
                             p_w_fold=pw_fold))
    P = pd.DataFrame(rows)
    P.to_csv(out / "points.csv", index=False)
    json.dump({"tags": tags, "conds": conds, "configs": configs, "n_cls": n_cls, "group_overhead_bits": ob,
               "lin_weights": LIN_W, "bitops": {c: backbone_bitops(c, ob) for c in configs}},
              open(out / "meta.json", "w"), indent=1)
    print(P[P.point == "b05"][["cond", "cheap", "f1", "bitops", "rate", "comparator", "comp_f1", "diff_pts", "p_t",
                               "p_w"]].to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
