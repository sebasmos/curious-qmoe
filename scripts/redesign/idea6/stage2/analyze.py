"""Idea 6 Stage 2 analysis of one task, as fixed in ANALYSIS_PLAN.md (commit
14c8504, amendment 1 at ed8b742): selection on clean inner validation, every
design on every condition, the hull, the random and audio-statistic gates,
oracle, humility. Amendment 1 adds the short-input AST (<config>@512 on
ESC-50, @400 on UrbanSound8K, own heads) as static systems and expensive
tiers, and reports the fold-level t-test with the Nadeau-Bengio corrected t
and the exact sign-flip p (no pair-level p).

Inputs (per-clip logits, test folds and inner validation, same rows):
  cheap tier  --eff-scores/<effp_task_s seed>/<kind>/<fp32|w8a8>__<cond>.npz (score_eff.py)
  AST tier    --ast-scores/<ast tag>/<esc|4>/<config>__<cond>.npz (idea 5 Stage 2)
  short AST   --ast-short-scores/<astshort_task_s seed>/<esc|4>/<config>__<cond>.npz (score_eff.py --dim 768)
  costs       --eff-emb and --ast-emb sidecars, heads from scripts/benchmark.py
  audio stats --audio-stats (idea 7 stage 1, ten statistics per clip)
Rows are checked equal across every file (labels, test folds, every fold's
inner validation indices) and the names checksum against the AST sidecars.

Field mixture: test rows take the test assignment's condition (seed 5001),
inner-validation rows the validation one (seed 5002), in both tiers.

Outputs in --out: static.csv (every static system, condition, pair, F1, cost
per view), selection.csv (every design's clean inner-validation gain over
the hull, per view), cascades.csv.gz (every design, condition, pair: F1,
rate, cost per view, ECE), picks.json (the designs of H1, H3, H4, E3),
selected.csv (the picked designs on every condition against the hull, the
random gate, the audio gate, the oracle and the best static system, raw p),
e2.csv (all designs on the field mixture against the hull), escalation.csv
(cheap systems at b10, b20, b30 on every cheap condition), auroc.csv,
calibration.csv.

    python scripts/redesign/idea6/stage2/analyze.py --task esc50 --eff-scores R/scores/esc50 \
        --ast-scores S2/scores/esc50 --eff-emb R/emb/esc50 --ast-emb S2/emb/esc50 \
        --audio-stats I7/audio_stats/esc50 --out R/analysis/esc50
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from audio_shift import FIELD_CELLS, FIELD_SEEDS, cond_name, field_assignment  # noqa: E402
from idea5b_common import BLOCKS, CORE, D as AST_D, L as AST_L, bitops as ast_bitops, mix_expected_f1, parse  # noqa: E402
from QWave.effnet_quant import ast_fbank_frontend_macs  # noqa: E402

TASKS = {
    "esc50": dict(eff="effp_esc50_s{}", ast="astesc50_s{}", short="astshort_esc50_s{}", seeds=[0, 1, 2, 3, 4], K=50),
    "esc5": dict(eff="effp_esc5_s{}", ast="astesc5_s{}", short="astshort_esc5_s{}", seeds=[0, 1, 2, 3, 4], K=5),
    "urban": dict(eff="effp_urban_s{}", ast="asturban_s{}", short="astshort_urban_s{}", seeds=[0, 1, 2], K=10),
    # amendment 3: ESC-50 with the official five categories, every head retrained here
    "esc5off": dict(eff="effp_esc5off_s{}", ast="astoff1024_esc5off_s{}", short="astshort_esc5off_s{}",
                    seeds=[0, 1, 2, 3, 4], K=5),
}
BACKBONES = ["fp32", "w8a8"]
KINDS = ["esc", "8", "4", "bitnet"]
AST_HEADS = ["esc", "4"]
EXPENSIVE_FULL = ["w4g128", "n4", "w4", "w5", "w6", "w8g64", "fp32"]
SHORT = ["fp32", "w8g64", "w4g128", "n4", "w3g64", "w4"]  # w4: amendment 2
SHORT_LEN = {"esc50": 512, "esc5": 512, "urban": 400, "esc5off": 512}
RULES = ["ast", "avg"]
BUDGETS = {"b05": 0.05, "b10": 0.10, "b20": 0.20, "b30": 0.30, "b50": 0.50}
VIEWS = {"priced": 1024.0, "unpriced": 0.0}
FAMILIES = {"H1": ("priced", None), "H3": ("priced", "fp32"), "H4": ("priced", "w8a8"), "E3": ("unpriced", None)}
FIELD = [cond_name(k, s) for k, s in FIELD_CELLS]
CHEAP_CONDS = ["clean"] + [cond_name(k, s) for k in ("bg", "pink", "white") for s in (20, 10, 5, 0)]
ECE_BINS = 15
DRAWS, RNG_IV = 200, 12346


# ── metrics (Stage 1's analyze_idea6.py) ───────────────────────────────────

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def f1_fast(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))


def f1_batch(y, P, k):
    D = P.shape[0]
    idx = (np.arange(D)[:, None] * (k * k) + y[None, :] * k + P).ravel()
    cm = np.bincount(idx, minlength=D * k * k).reshape(D, k, k)
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float)
    den = cm.sum(1) + cm.sum(2)
    r = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return r.sum(1) / (den > 0).sum(1)


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def thr_rate(m_iv, rate):
    u = np.unique(m_iv)
    c = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    rates = (m_iv[None, :] < c[:, None]).mean(1)
    return c[np.lexsort((rates, np.abs(rates - rate)))[0]]


def oracle_pred(y, pc, de, n):
    fix, harm = (pc != y) & (de == y), (pc == y) & (de != y)
    order = np.argsort(np.where(fix, 0, np.where(harm, 2, 1)), kind="stable")
    e = np.zeros(len(y), bool)
    e[order[:n]] = True
    return np.where(e, de, pc)


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


def _p(fn, *a):
    try:
        v = fn(*a).pvalue
        return float(v) if np.isfinite(v) else 0.0
    except ValueError:
        return 1.0


NB_RATIO = {}  # task -> mean over folds of n_test / n_train, set in main


def sign_flip_p(fm):
    """Exact two-sided sign-flip p of the mean of fold differences (2^k sign vectors)."""
    k = len(fm)
    signs = 1 - 2 * ((np.arange(2 ** k)[:, None] >> np.arange(k)[None, :]) & 1)
    stat = np.abs((signs * fm[None, :]).mean(1))
    return float(np.mean(stat >= abs(fm.mean()) - 1e-12))


def tests(x, z, keys, ratio=None):
    """x - z over (seed, fold) keys, seeds averaged within each fold: two-sided one-sample t-test over the
    fold means (the decision), the Nadeau-Bengio corrected t (variance x (1/k + n_test/n_train)) and the
    exact sign-flip p with its floor 2 / 2^k. No pair-level p."""
    ratio = NB_RATIO.get("task") if ratio is None else ratio
    d = np.asarray(x, float) - np.asarray(z, float)
    folds = np.array([k[1] for k in keys])
    fm = np.array([d[folds == f].mean() for f in np.unique(folds)])
    k = len(fm)
    out = {"n_pairs": len(d), "n_folds": k, "diff": float(d.mean()), "diff_fold_min": float(fm.min()),
           "diff_fold_max": float(fm.max()), "fold_diffs": json.dumps(np.round(fm, 6).tolist()),
           "signflip_floor": 2.0 / 2 ** k}
    if np.allclose(fm, 0):
        out.update(p_t_fold=1.0, p_nb=1.0, p_signflip=1.0)
        return out
    sd = fm.std(ddof=1)
    if sd == 0:
        out.update(p_t_fold=0.0, p_nb=0.0, p_signflip=sign_flip_p(fm))
        return out
    t_nb = fm.mean() / np.sqrt((1.0 / k + ratio) * sd ** 2)
    out.update(p_t_fold=_p(stats.ttest_1samp, fm, 0.0), p_nb=float(2 * stats.t.sf(abs(t_nb), k - 1)),
               p_signflip=sign_flip_p(fm))
    return out


def hull_pair(sp, c):
    """Best static system or two-system time-sharing at expected cost c, on mean F1 (Stage 1)."""
    lo_set = sp[sp.cost <= c * (1 + 1e-12)]
    hi_set = sp[sp.cost > c * (1 + 1e-12)]
    best = None
    for lo in lo_set.itertuples():
        cands = [(lo.f1, lo.system, lo.system, 0.0)]
        for hi in hi_set.itertuples():
            lam = (c - lo.cost) / (hi.cost - lo.cost)
            cands.append(((1 - lam) * lo.f1 + lam * hi.f1, lo.system, hi.system, lam))
        for v in cands:
            if best is None or v[0] > best[0] + 1e-15:
                best = v
    return best


# ── scores ─────────────────────────────────────────────────────────────────

class ScoreDir:
    """Logits of one run (seed, head) on every (config, condition); field rows assembled."""

    def __init__(self, d: Path):
        self.d, self.cache = d, {}
        self.files = {p.stem for p in d.glob("*.npz")}

    def raw(self, cfg, cond):
        key = (cfg, cond)
        if key not in self.cache:
            z = np.load(self.d / f"{cfg}__{cond}.npz")
            iv = {int(k[7:]): z[k] for k in z.files if k.startswith("iv_idx_")}
            self.cache[key] = {"y": z["y"].astype(np.int64), "fold": z["test_fold"].astype(int),
                               "z": z["test_logits"].astype(np.float64),
                               "iv": {f: (iv[f], z[f"iv_logits_{f}"].astype(np.float64)) for f in sorted(iv)},
                               "crc": int(z["names_crc32"]) if "names_crc32" in z.files else None}
        return self.cache[key]

    def complete(self, cfg, cond):
        if f"{cfg}__{cond}" not in self.files:
            return False
        r = self.raw(cfg, cond)
        return bool(np.isfinite(r["z"]).all() and all(np.isfinite(v).all() for _, v in r["iv"].values()))

    def has(self, cfg, cond):
        if cond == "field":
            return all(f"{cfg}__{c}" in self.files for c in ["clean"] + FIELD)
        return self.complete(cfg, cond)

    def get(self, cfg, cond):
        if cond != "field":
            return self.raw(cfg, cond)
        key = (cfg, "field")
        if key not in self.cache:
            base = self.raw(cfg, "clean")
            at, av = field_arrays(len(base["y"]))
            z = np.full_like(base["z"], np.nan)
            ivs = {f: (idx, np.full_like(v, np.nan)) for f, (idx, v) in base["iv"].items()}
            for c in ["clean"] + FIELD:
                r = self.raw(cfg, c)
                m = at == c
                z[m] = r["z"][m]
                for f, (idx, v) in ivs.items():
                    sel = av[idx] == c
                    v[sel] = r["iv"][f][1][sel]
            if np.isnan(z).any() or any(np.isnan(v).any() for _, v in ivs.values()):
                raise SystemExit(f"{self.d}: field rows missing for {cfg}")
            self.cache[key] = {"y": base["y"], "fold": base["fold"], "z": z, "iv": ivs, "crc": base["crc"]}
        return self.cache[key]


_FIELD = {}


def field_arrays(n):
    if n not in _FIELD:
        _FIELD[n] = tuple(np.array(field_assignment(n, FIELD_SEEDS[k])) for k in ("test", "val"))
    return _FIELD[n]


class AudioStats:
    """Idea 7's ten statistics per (condition, clip); field rows from the assignments."""

    def __init__(self, d: Path, stems):
        self.d, self.cache = d, {}
        clean = np.load(d / "clean.npz", allow_pickle=True)
        if [str(s) for s in clean["names"]] != list(stems):
            raise SystemExit(f"{d}: audio statistics rows differ from the embedding rows")
        self.n = len(stems)

    def available(self, cond):
        """Field: every cell file exists (rows checked when read); otherwise a complete file."""
        if cond == "field":
            return all((self.d / f"{c}.npz").exists() for c in ["clean"] + FIELD)
        return (self.d / f"{cond}.npz").exists() and bool(np.isfinite(self.cond(cond)).all())

    def cond(self, c):
        if c not in self.cache:
            self.cache[c] = np.load(self.d / f"{c}.npz", allow_pickle=True)["stats"]
        return self.cache[c]

    def rows(self, cond, idx, which):
        if cond != "field":
            out = self.cond(cond)[idx]
        else:
            a = field_arrays(self.n)[0 if which == "test" else 1]
            out = np.full((len(idx), 10), np.nan)
            for c in ["clean"] + FIELD:
                sel = a[idx] == c
                out[sel] = self.cond(c)[idx[sel]]
        if np.isnan(out).any():
            raise KeyError(cond)
        return out


def fit_lr(X, t):
    """Idea 7's noise_lr: logistic regression, C = 1, standardized inputs."""
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), t)
    return lambda Z: lr.predict_proba(sc.transform(Z))[:, 1]


def top_k(score, k):
    m = np.zeros(len(score), bool)
    if k > 0:
        m[np.argsort(-score, kind="stable")[:k]] = True
    return m


# ── costs ──────────────────────────────────────────────────────────────────

def ast_bitops_tok(cfg: str, price: float, tokens: int, patches: int) -> float:
    """idea 5b's bitops(cfg, price) with the token and patch counts as parameters (1214 and 1212 there)."""
    lin_macs = AST_L * sum(o * i for o, i in BLOCKS) * tokens
    att_half = tokens * tokens * AST_D * AST_L
    patch_macs = AST_D * 16 * 16 * patches
    if cfg == "fp32":
        return float((lin_macs + 2 * att_half + patch_macs) * 32 * 32)
    wb, g, pbits, _ = parse(cfg)
    resc = tokens * AST_L * sum(o * (i // g if g else 1) for o, i in BLOCKS)
    return float(lin_macs * wb * 8 + att_half * 8 * 8 + att_half * pbits * 8 + patch_macs * 32 * 32 + resc * price)


def short_tokens(max_length: int) -> tuple[int, int]:
    nt = (max_length - 16) // 10 + 1
    return 12 * nt, 12 * nt + 2


def costs(task, eff_emb: Path, ast_emb: Path, ast_configs, short_emb: Path, short_configs):
    from costs_idea6 import head_costs  # Stage 1's head costs (scripts/redesign/idea6)

    with contextlib.redirect_stdout(io.StringIO()):
        hk = "esc5" if task == "esc5off" else task  # same network, five classes
        eh, ah = head_costs(1536, KINDS)[hk], head_costs(768, AST_HEADS)[hk]
    side = {bb: json.loads((eff_emb / f"{bb}__clean.json").read_text())["cost_per_clip"] for bb in BACKBONES}
    lengths = np.load(eff_emb / "lengths__clean.npy")
    fbank = float(np.mean([ast_fbank_frontend_macs(int(n)) for n in lengths])) * 1024
    cheap, ast = {v: {} for v in VIEWS}, {v: {} for v in VIEWS}
    for v, price in VIEWS.items():
        for bb in BACKBONES:
            bbo = side[bb]["backbone_bitops_rescale1024"] if price else side[bb]["backbone_bitops"]
            for k in KINDS:
                cheap[v][(bb, k)] = side[bb]["frontend_bitops_mean"] + bbo + eh[k]["bitops"]
        for cfg in ast_configs:
            for h in AST_HEADS:
                ast[v][(cfg, h)] = fbank + ast_bitops(cfg, price) + ah[h]["bitops"]
        L_ = SHORT_LEN[task]
        P_, T_ = short_tokens(L_)
        for cfg in short_configs:
            for h in AST_HEADS:
                ast[v][(f"{cfg}@{L_}", h)] = fbank + ast_bitops_tok(cfg, price, T_, P_) + ah[h]["bitops"]
    # sidecar bit-ops must equal idea 5b's count (unpriced), at 1024 frames and at the short length
    for cfg in ast_configs:
        sc = json.loads((ast_emb / f"{cfg}__clean.json").read_text())["backbone_cost_per_clip"]["bitops"]
        if abs(sc - ast_bitops(cfg, 0.0)) > 1e-6 * sc or abs(ast_bitops_tok(cfg, 1024, 1214, 1212)
                                                              - ast_bitops(cfg, 1024)) > 1:
            raise SystemExit(f"{cfg}: sidecar bit-ops {sc} != {ast_bitops(cfg, 0.0)}")
    P_, T_ = short_tokens(SHORT_LEN[task])
    for cfg in short_configs:
        side_s = json.loads((short_emb / f"{cfg}__clean.json").read_text())
        sc = side_s["backbone_cost_per_clip"]["bitops"]
        if side_s["max_length"] != SHORT_LEN[task] or abs(sc - ast_bitops_tok(cfg, 0.0, T_, P_)) > 1e-6 * sc:
            raise SystemExit(f"{cfg}@{SHORT_LEN[task]}: sidecar bit-ops {sc} != {ast_bitops_tok(cfg, 0.0, T_, P_)}")
    detail = {"eff_backbone": side, "fbank_bitops_mean": fbank, "eff_heads": eh, "ast_heads": ah,
              "ast_backbone": {v: {**{c: ast_bitops(c, p) for c in ast_configs},
                                   **{f"{c}@{SHORT_LEN[task]}": ast_bitops_tok(c, p, *short_tokens(SHORT_LEN[task])[::-1])
                                      for c in short_configs}} for v, p in VIEWS.items()},
              "cheap": {v: {f"{bb}:{k}": c for (bb, k), c in d.items()} for v, d in cheap.items()},
              "ast": {v: {f"{c}+{h}": x for (c, h), x in d.items()} for v, d in ast.items()}}
    return cheap, ast, detail


# ── main ───────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--eff-scores", required=True, type=Path)
    ap.add_argument("--ast-scores", required=True, type=Path)
    ap.add_argument("--eff-emb", required=True, type=Path)
    ap.add_argument("--ast-emb", required=True, type=Path)
    ap.add_argument("--ast-short-scores", required=True, type=Path)
    ap.add_argument("--ast-short-emb", required=True, type=Path)
    ap.add_argument("--audio-stats", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6"))
    t0 = time.time()
    T = TASKS[a.task]
    global EXPENSIVE
    EXPENSIVE = EXPENSIVE_FULL + [f"{c}@{SHORT_LEN[a.task]}" for c in SHORT]
    K, seeds = T["K"], T["seeds"]
    a.out.mkdir(parents=True, exist_ok=True)

    # conditions: cheap tier everywhere it was extracted; cascades where every AST file is complete
    eff = {(s, k): ScoreDir(a.eff_scores / T["eff"].format(s) / k) for s in seeds for k in KINDS}
    full = {(s, h): ScoreDir(a.ast_scores / T["ast"].format(s) / h) for s in seeds for h in AST_HEADS}
    short = {(s, h): ScoreDir(a.ast_short_scores / T["short"].format(s) / h) for s in seeds for h in AST_HEADS}
    SL = SHORT_LEN[a.task]
    # AST systems by name: 1024-frame configurations as named in Stage 2, short ones as <config>@<frames>
    src = {}
    for cfg in sorted({f.split("__")[0] for f in full[(seeds[0], "esc")].files}):
        src[cfg] = (full, cfg)
    for cfg in SHORT:
        src[f"{cfg}@{SL}"] = (short, cfg)

    def ast_has(name, cond):
        dirs, base = src[name]
        return all(d.has(base, cond) for d in dirs.values())

    def ast_get(name, seed, h, cond):
        dirs, base = src[name]
        return dirs[(seed, h)].get(base, cond)

    cheap_conds = [c for c in CHEAP_CONDS if all(d.has(bb, c) for d in eff.values() for bb in BACKBONES)]
    cheap_conds.append("field")
    clean_cfgs = [c for c in src if "@" not in c and ast_has(c, "clean")]
    short_cfgs = [c for c in SHORT if ast_has(f"{c}@{SL}", "clean")]
    casc_conds = [c for c in cheap_conds if all(ast_has(cfg, c) for cfg in CORE)]
    cfgs_of = {c: [n for n in src if ast_has(n, c)] for c in casc_conds}
    tiers_of = {c: [e for e in EXPENSIVE if e in cfgs_of[c]] for c in casc_conds}
    print(f"{a.task}: cheap conditions {cheap_conds}\n  cascade conditions {casc_conds}\n  "
          f"expensive tiers missing: { {c: [e for e in EXPENSIVE if e not in t] for c, t in tiers_of.items()} }",
          flush=True)
    cheap_cost, ast_cost, cost_detail = costs(a.task, a.eff_emb, a.ast_emb, clean_cfgs, a.ast_short_emb, short_cfgs)
    (a.out / "costs.json").write_text(json.dumps(cost_detail, indent=1))

    stems = [str(s) for s in np.load(a.audio_stats / "clean.npz", allow_pickle=True)["names"]]
    audio = AudioStats(a.audio_stats, stems)
    crc_ast = {json.loads((a.ast_emb / f"{c}__clean.json").read_text())["names_crc32"] for c in clean_cfgs}
    crc_ast |= {json.loads((a.ast_short_emb / f"{c}__clean.json").read_text())["names_crc32"] for c in short_cfgs}

    static, sel_rows, casc, esc_rows, calib = [], [], [], [], []
    preds = {}  # (system, cond, seed, fold) -> test predictions
    ys = {}  # (seed, fold) -> test labels
    acs = {}  # (bb, k, cond, seed, fold) -> cheap test predictions
    escs = {}  # (bb, k, budget, cond, seed, fold) -> escalation mask on the test rows
    decs = {}  # (bb, k, e, rule, cond, seed, fold) -> decision on escalated clips
    au = {}  # (bb, k, e, cond, seed) -> per-fold arrays for the gate AUROC
    noise = {}  # (seed, fold) -> noise_lr scorer
    iv_static = {}  # (system, seed, fold) -> (f1_iv, preds_iv), clean inner validation
    yiv = {}
    designs = [(bb, k, e, r, b) for bb in BACKBONES for k in KINDS for e in EXPENSIVE for r in RULES for b in BUDGETS]

    for seed in seeds:
        ref = eff[(seed, "esc")].get("fp32", "clean")
        if ref["crc"] not in crc_ast or len(crc_ast) != 1:
            raise SystemExit(f"names checksum {ref['crc']} != AST sidecars {crc_ast}")
        folds = sorted(ref["iv"])
        checks = [(eff[(seed, k)], bb, c) for k in KINDS for bb in BACKBONES for c in casc_conds]
        checks += [(src[n][0][(seed, h)], src[n][1], c) for c in casc_conds for n in cfgs_of[c] for h in AST_HEADS]
        for d, cfg, c in checks:
            r = d.get(cfg, c)
            if not ((r["y"] == ref["y"]).all() and (r["fold"] == ref["fold"]).all()
                    and sorted(r["iv"]) == folds
                    and all(np.array_equal(r["iv"][f][0], ref["iv"][f][0]) for f in folds)):
                raise SystemExit(f"{d.d}/{cfg}__{c}: rows, folds or inner splits differ")
        if seed == seeds[0]:
            n_te = np.array([np.sum(ref["fold"] == f) for f in folds])
            NB_RATIO["task"] = float(np.mean(n_te / (len(ref["y"]) - n_te)))
        y_all = ref["y"]
        for fold in folds:
            te = np.flatnonzero(ref["fold"] == fold)
            iv = ref["iv"][fold][0]
            y, yv = y_all[te], y_all[iv]
            ys[(seed, fold)], yiv[(seed, fold)] = y, yv
            key = (seed, fold)
            # idea 7's noise detector, fit on this fold's inner validation under the field validation assignment
            av = field_arrays(len(y_all))[1]
            noise[key] = fit_lr(audio.rows("field", iv, "val"), (av[iv] != "clean").astype(int))

            # clean inner validation: static systems, thresholds, design selection rows
            civ = {(bb, k): softmax(eff[(seed, k)].get(bb, "clean")["iv"][fold][1]) for bb in BACKBONES for k in KINDS}
            aiv = {(n, h): softmax(ast_get(n, seed, h, "clean")["iv"][fold][1]) for n in cfgs_of["clean"]
                   for h in AST_HEADS}
            for (bb, k), p in civ.items():
                pr = p.argmax(1)
                iv_static[(f"eff:{bb}:{k}", seed, fold)] = (f1_fast(yv, pr, K), pr)
            for (cfg, h), p in aiv.items():
                pr = p.argmax(1)
                iv_static[(f"ast:{cfg}+{h}", seed, fold)] = (f1_fast(yv, pr, K), pr)
            thr = {}
            for (bb, k), p in civ.items():
                m = margin(p)
                ac = p.argmax(1)
                for b, rate in BUDGETS.items():
                    t = thr_rate(m, rate)
                    thr[(bb, k, b)] = t
                    escm = m < t
                    for e in tiers_of["clean"]:
                        pe = aiv[(e, "esc")]
                        for r in RULES:
                            de = pe.argmax(1) if r == "ast" else ((p + pe) / 2).argmax(1)
                            sel_rows.append({"cheap_bb": bb, "cheap": k, "expensive": e, "rule": r, "budget": b,
                                             "seed": seed, "fold": fold, "threshold": float(t),
                                             "iv_rate": float(escm.mean()),
                                             "f1_iv": f1_fast(yv, np.where(escm, de, ac), K)})
            # temperatures of the cheap systems, clean inner validation
            temps = {(bb, k): fit_temperature(eff[(seed, k)].get(bb, "clean")["iv"][fold][1], yv)
                     for bb in BACKBONES for k in KINDS}

            # test rows, every cheap condition
            for c in cheap_conds:
                cp = {}
                for bb in BACKBONES:
                    for k in KINDS:
                        z = eff[(seed, k)].get(bb, c)["z"][te]
                        p = softmax(z)
                        pr = p.argmax(1)
                        cp[(bb, k)] = p
                        name = f"eff:{bb}:{k}"
                        preds[(name, c, seed, fold)] = pr.astype(np.int16)
                        static.append({"system": name, "condition": c, "seed": seed, "fold": fold,
                                       "f1": f1_fast(y, pr, K), "acc": float((pr == y).mean()),
                                       **{f"cost_{v}": cheap_cost[v][(bb, k)] for v in VIEWS}})
                        pt = softmax(z / temps[(bb, k)])
                        calib.append({"system": name, "condition": c, "seed": seed, "fold": fold,
                                      "T": temps[(bb, k)], "ece": ece_conf(p.max(1), pr == y),
                                      "conf_gap": float(p.max(1).mean() - (pr == y).mean()),
                                      "ece_T": ece_conf(pt.max(1), pt.argmax(1) == y),
                                      "conf_gap_T": float(pt.max(1).mean() - (pt.argmax(1) == y).mean())})
                        m = margin(p)
                        for b in ("b10", "b20", "b30"):
                            esc_rows.append({"cheap_bb": bb, "cheap": k, "budget": b, "condition": c, "seed": seed,
                                             "fold": fold, "rate": float((m < thr[(bb, k, b)]).mean())})
                if c not in casc_conds:
                    continue
                ap_ = {}
                for cfg in cfgs_of[c]:
                    for h in AST_HEADS:
                        p = softmax(ast_get(cfg, seed, h, c)["z"][te])
                        pr = p.argmax(1)
                        name = f"ast:{cfg}+{h}"
                        preds[(name, c, seed, fold)] = pr.astype(np.int16)
                        static.append({"system": name, "condition": c, "seed": seed, "fold": fold,
                                       "f1": f1_fast(y, pr, K), "acc": float((pr == y).mean()),
                                       **{f"cost_{v}": ast_cost[v][(cfg, h)] for v in VIEWS}})
                        if h == "esc":
                            ap_[cfg] = p
                for (bb, k), p in cp.items():
                    m, ac = margin(p), p.argmax(1)
                    acs[(bb, k, c, seed, fold)] = ac.astype(np.int16)
                    qs = {}
                    for e in tiers_of[c]:
                        pe = ap_[e]
                        for r in RULES:
                            q = pe if r == "ast" else (p + pe) / 2
                            qs[(e, r)] = (q.argmax(1), q.max(1))
                            decs[(bb, k, e, r, c, seed, fold)] = qs[(e, r)][0].astype(np.int16)
                        a_ = au.setdefault((bb, k, e, c, seed), {"score": [], "wrong": [], "fixable": []})
                        a_["score"].append(1 - m)
                        a_["wrong"].append(ac != y)
                        a_["fixable"].append((ac != y) & (pe.argmax(1) == y))
                    for b in BUDGETS:
                        escm = m < thr[(bb, k, b)]
                        escs[(bb, k, b, c, seed, fold)] = escm
                        rate = float(escm.mean())
                        for (e, r), (de, qmax) in qs.items():
                            pred = np.where(escm, de, ac)
                            conf = np.where(escm, qmax, p.max(1))
                            casc.append({"cheap_bb": bb, "cheap": k, "expensive": e, "rule": r, "budget": b,
                                         "condition": c, "seed": seed, "fold": fold,
                                         "f1": f1_fast(y, pred, K), "acc": float((pred == y).mean()),
                                         "rate": rate, "ece": ece_conf(conf, pred == y),
                                         **{f"cost_{v}": cheap_cost[v][(bb, k)] + rate * ast_cost[v][(e, "esc")]
                                            for v in VIEWS}})
            print(f"{a.task} seed {seed} fold {fold}: {time.time() - t0:.0f}s", flush=True)
        for d in list(eff.values()) + list(full.values()) + list(short.values()):
            d.cache.clear()

    S = pd.DataFrame(static)
    S.to_csv(a.out / "static.csv", index=False)
    CA = pd.DataFrame(casc)
    CA.to_csv(a.out / "cascades.csv.gz", index=False)
    pd.DataFrame(esc_rows).to_csv(a.out / "escalation.csv", index=False)
    pd.DataFrame(calib).to_csv(a.out / "calibration.csv", index=False)

    # gate AUROC per seed pooled over folds (margin), cheap-wrong for every cheap condition below
    rows = []
    for (bb, k, e, c, seed), a_ in au.items():
        sc, w, fx = (np.concatenate(a_[x]) for x in ("score", "wrong", "fixable"))
        rows.append({"cheap_bb": bb, "cheap": k, "expensive": e, "condition": c, "seed": seed,
                     "auroc_wrong": roc_auc_score(w, sc) if 0 < w.sum() < len(w) else np.nan,
                     "auroc_fixable": roc_auc_score(fx, sc) if 0 < fx.sum() < len(fx) else np.nan,
                     "n_fixable": int(fx.sum())})
    pd.DataFrame(rows).to_csv(a.out / "auroc_cascade_conditions.csv", index=False)

    # ── selection on clean inner validation (per view and family) ─────────
    SR = pd.DataFrame(sel_rows)
    rng = np.random.default_rng(RNG_IV)
    sel_out = []
    for view in VIEWS:
        st = []
        for (name, seed, fold), (f1, _) in iv_static.items():
            kind, sysn = name.split(":", 1)
            if kind == "eff":
                bb, k = sysn.split(":")
                cst = cheap_cost[view][(bb, k)]
            else:
                cfg, h = sysn.split("+")
                cst = ast_cost[view][(cfg, h)]
            st.append({"system": name, "seed": seed, "fold": fold, "f1": f1, "cost": cst})
        st = pd.DataFrame(st)
        sp = st.groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
        for (bb, k, e, r, b), g in SR.groupby(["cheap_bb", "cheap", "expensive", "rule", "budget"], sort=False):
            g = g.sort_values(["seed", "fold"])
            c_iv = float(np.mean(cheap_cost[view][(bb, k)] + g.iv_rate.to_numpy() * ast_cost[view][(e, "esc")]))
            v_interp, lo, hi, lam = hull_pair(sp, c_iv)
            hull = []
            for seed, fold in zip(g.seed, g.fold):
                plo = iv_static[(lo, seed, fold)][1]
                if lam == 0.0:
                    hull.append(iv_static[(lo, seed, fold)][0])
                    continue
                phi = iv_static[(hi, seed, fold)][1]
                draws = rng.random((DRAWS, len(plo))) < lam
                hull.append(float(f1_batch(yiv[(seed, fold)], np.where(draws, phi[None, :], plo[None, :]), K).mean()))
            d = g.f1_iv.to_numpy() - np.array(hull)
            sel_out.append({"view": view, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r, "budget": b,
                            "f1_iv": g.f1_iv.mean(), "iv_rate": g.iv_rate.mean(), "cost_iv": c_iv, "hull_lo": lo,
                            "hull_hi": hi, "lam": lam, "hull_f1_iv": float(np.mean(hull)), "gain_iv": float(d.mean())})
    SO = pd.DataFrame(sel_out)
    SO.to_csv(a.out / "selection.csv", index=False)
    picks = {}
    for fam, (view, bb) in FAMILIES.items():
        g = SO[SO.view == view]
        if bb is not None:
            g = g[g.cheap_bb == bb]
        g = g.sort_values(["gain_iv", "cost_iv"], ascending=[False, True])
        top = g.iloc[0]
        picks[fam] = {"view": view, "design": [top.cheap_bb, top.cheap, top.expensive, top.rule, top.budget],
                      "gain_iv": float(top.gain_iv), "runner_up_gain_iv": float(g.iloc[1].gain_iv),
                      "cost_iv": float(top.cost_iv), "f1_iv": float(top.f1_iv), "iv_rate": float(top.iv_rate)}
    (a.out / "picks.json").write_text(json.dumps(picks, indent=1))
    print(f"{a.task}: picks {json.dumps({f: p['design'] for f, p in picks.items()})}", flush=True)

    # ── test comparisons ───────────────────────────────────────────────────
    # the audio gate needs each fold's test row indices
    test_idx = {}
    ref = ScoreDir(a.eff_scores / T["eff"].format(seeds[0]) / "esc").get("fp32", "clean")
    for f in sorted(ref["iv"]):
        test_idx[f] = np.flatnonzero(ref["fold"] == f)

    s_rec = {(n, c): dict(zip(zip(g.seed, g.fold), g.f1)) for (n, c), g in S.groupby(["system", "condition"])}
    CAg = {key: g.sort_values(["seed", "fold"]) for key, g in
           CA.groupby(["cheap_bb", "cheap", "expensive", "rule", "budget", "condition"])}
    sp_cache = {}

    def hull_for(view, c, cost):
        if (view, c) not in sp_cache:
            sp_cache[(view, c)] = (S[S.condition == c].groupby("system")
                                   .agg(f1=("f1", "mean"), cost=(f"cost_{view}", "mean")).reset_index())
        sp = sp_cache[(view, c)]
        v_interp, lo, hi, lam = hull_pair(sp, cost)
        ok = sp[sp.cost <= cost * (1 + 1e-12)].sort_values(["f1", "cost"], ascending=[False, True]).iloc[0]
        return v_interp, lo, hi, lam, ok.system

    def hull_f1(c, lo, hi, lam, key):
        if lam == 0.0:
            return s_rec[(lo, c)][key]
        return float(mix_expected_f1(ys[key], preds[(lo, c, *key)].astype(np.int64),
                                     preds[(hi, c, *key)].astype(np.int64), lam, K)[0])

    def compare(design, view, c, with_baselines):
        bb, k, e, r, b = design
        g = CAg[(bb, k, e, r, b, c)]
        keys = list(zip(g.seed, g.fold))
        x = g.f1.to_numpy()
        cost = float(g[f"cost_{view}"].mean())
        v_interp, lo, hi, lam, best_static = hull_for(view, c, cost)
        hull = np.array([hull_f1(c, lo, hi, lam, kk) for kk in keys])
        row = {"view": view, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r, "budget": b, "condition": c,
               "f1": float(x.mean()), "cost": cost, "rate": float(g.rate.mean()), "rate_min": float(g.rate.min()),
               "rate_max": float(g.rate.max()), "ece": float(g.ece.mean()), "hull_lo": lo, "hull_hi": hi,
               "lam": lam, "hull_f1_interp": v_interp, "hull_f1": float(hull.mean())}
        row.update({f"hull_{kk}": vv for kk, vv in tests(x, hull, keys).items()})
        if not with_baselines:
            return row
        rnd, orc, orc_full, aud = [], [], [], []
        for kk in keys:
            pc = acs[(bb, k, c, *kk)].astype(np.int64)
            de = decs[(bb, k, e, r, c, *kk)].astype(np.int64)
            escm = escs[(bb, k, b, c, *kk)]
            yk = ys[kk]
            rnd.append(float(mix_expected_f1(yk, pc, de, escm.mean(), K)[0]))
            orc.append(f1_fast(yk, oracle_pred(yk, pc, de, int(escm.sum())), K))
            fix = (pc != yk) & (de == yk)
            orc_full.append(f1_fast(yk, np.where(fix, de, pc), K))
            if audio.available(c):
                sc = noise[kk](audio.rows(c, test_idx[kk[1]], "test"))
                aud.append(f1_fast(yk, np.where(top_k(sc, int(escm.sum())), de, pc), K))
        row["random_f1"] = float(np.mean(rnd))
        row.update({f"random_{kk}": vv for kk, vv in tests(x, rnd, keys).items()})
        row["oracle_k_f1"], row["oracle_full_f1"] = float(np.mean(orc)), float(np.mean(orc_full))
        row["best_static"] = best_static
        z = np.array([s_rec[(best_static, c)][kk] for kk in keys])
        row["best_static_f1"] = float(z.mean())
        row.update({f"static_{kk}": vv for kk, vv in tests(x, z, keys).items()})
        if aud:
            row["audio_f1"] = float(np.mean(aud))
            row.update({f"audio_{kk}": vv for kk, vv in tests(x, aud, keys).items()})
        return row

    sel_rows_out = []
    for fam, p in picks.items():
        for c in casc_conds:
            if tuple(p["design"]) + (c,) not in CAg:
                continue  # its expensive tier was not extracted in this condition
            row = compare(tuple(p["design"]), p["view"], c, True)
            row["family"] = fam
            sel_rows_out.append(row)
    pd.DataFrame(sel_rows_out).to_csv(a.out / "selected.csv", index=False)
    print(f"{a.task}: selected designs compared, {time.time() - t0:.0f}s", flush=True)

    e2 = [compare(d, "priced", "field", False) for d in designs if d + ("field",) in CAg]
    pd.DataFrame(e2).to_csv(a.out / "e2.csv", index=False)
    print(f"{a.task}: E2 done, {time.time() - t0:.0f}s", flush=True)

    # cheap-wrong AUROC on every cheap condition and the noise detector's AUROC where its statistics exist
    rows = []
    for seed in seeds:
        for bb in BACKBONES:
            for k in KINDS:
                d = eff[(seed, k)]
                for c in cheap_conds:
                    r = d.get(bb, c)
                    m = r["fold"] >= 0
                    p = softmax(r["z"][m])
                    wrong = p.argmax(1) != r["y"][m]
                    row = {"cheap_bb": bb, "cheap": k, "condition": c, "seed": seed,
                           "auroc_wrong_margin": roc_auc_score(wrong, 1 - margin(p)), "error_rate": float(wrong.mean())}
                    if audio.available(c):
                        sc = np.empty(len(r["y"]))
                        for f, idx in test_idx.items():
                            sc[idx] = noise[(seed, f)](audio.rows(c, idx, "test"))
                        row["auroc_wrong_noise_lr"] = roc_auc_score(wrong, sc[m])
                    rows.append(row)
                d.cache.clear()
    pd.DataFrame(rows).to_csv(a.out / "auroc.csv", index=False)
    meta = {"task": a.task, "K": K, "seeds": seeds, "cheap_conditions": cheap_conds, "cascade_conditions": casc_conds,
            "static_ast_configs": cfgs_of, "tiers_of": tiers_of, "nb_ratio": NB_RATIO["task"],
            "draws_iv": DRAWS, "rng_iv": RNG_IV, "n_designs": len(designs),
            "picks": picks, "seconds": time.time() - t0}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{a.task}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
