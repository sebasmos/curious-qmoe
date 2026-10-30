"""Idea 7 stage 1: a quantized on-device AST that defers clips to the FP32 AST
in the cloud, scored from Stage 2's head outputs (no new forward pass).

PRE-SPECIFIED ANALYSIS (fixed in this docstring and committed before this
script was first run on any test fold)

Systems. Device: one quantized AST configuration with one Stage 2 head, run on
every clip: n3g64 (nested 3-bit), w3g64, n4, w4g128, w8g64, each with the FP32
head ("esc") and the 4-bit head ("4"). Cloud: the FP32 AST with the FP32 head
of the same seed. A deferred clip takes the cloud's prediction under the same
condition. Device memory is the weight storage of the device configuration
plus its head (Stage 2 accounting).

Gate. Device softmax margin; the score is s = 1 - (p1 - p2) and a clip is
deferred when s > t. Thresholds come from the fold's CLEAN inner-validation
split only (protocol 3, grouped by recording) and are applied unchanged to
every condition (the deployable rule):
  pre   the smallest inner-validation deferral rate at which the
        inner-validation macro F1 of device + deferral reaches the cloud's
        inner-validation macro F1 minus 0.005 (0.5 points); it is always
        reached at rate 1
  b05, b10, b20   inner-validation deferral rate closest to 5, 10, 20%
                  (ties: fewer deferrals), for the curves

Baselines, per (seed, fold, condition) at the gate's realized test deferral
count k:
  random     expected macro F1 over 500 random subsets of k test clips
  noise_lr   PRIMARY noise detector: logistic regression (C = 1, standardized
             inputs) on the ten audio statistics of audio_stats.py, fit on the
             fold's inner-validation clips under the field-mixture validation
             assignment to predict "clip is corrupted"; it never sees any
             model output; defers its k most-likely-noisy clips
  snr_est    the blind SNR estimate alone (lowest first), exploratory
  flatness   median spectral flatness alone (flattest first), exploratory
  err_lr     the same logistic regression fit to predict "device wrong" on
             the same inner-validation clips (it sees the device's errors but
             not its outputs at test time), exploratory
  oracle     defers exactly the test clips the device gets wrong and the
             cloud gets right (its own rate), and oracle_k, at most k of
             those clips (expected over 200 random choices when there are
             more than k)
  device alone, the cloud alone, and the next device up alone (n3g64 -> n4,
  w3g64 -> w4g128, n4 and w4g128 -> w8g64) are scored per condition.

Primary analysis, one per task (ESC-50 50 classes, ESC-50 5 classes,
UrbanSound8K): device n3g64 with the FP32 head, rule pre, condition field.
  Claim A  F1(device + deferral) - F1(cloud) > -0.005, one-sided
           (non-inferiority by 0.5 points)
  Claim B  F1(device + deferral) > F1(random) and > F1(noise_lr), each
           one-sided
Each test is a one-sample t-test on paired differences, computed twice: over
folds (seeds averaged within each fold) and over (seed, fold) pairs. Holm over
the 9 hypotheses (3 tasks x {A, B random, B noise_lr}) separately for each of
the two pairings. A hypothesis holds when its Holm-adjusted p < 0.05 under
both pairings. GO when all 9 hold. Everything else is exploratory (stats.py
states the family size).

Humility outputs: realized deferral rate per condition (SNR and noise type);
share of each condition's recoverable errors (device wrong, cloud right)
captured by the deferred clips; ECE (15 bins, top label) of the device alone
before and after temperature scaling fit by NLL on the fold's clean
inner-validation logits; AUROC of the margin and of each audio score for
"device wrong" and for "device wrong, cloud right".

Costs: device weight memory, cloud calls per 1000 clips, and bits sent per
deferred clip: the clip's audio as 16-bit PCM at 16 kHz mono (the cloud runs
the FP32 AST on the audio, so the device's embedding cannot stand in for it).

Conditions: every condition whose score files are complete for every device,
the cloud, every seed and both heads, plus the field mixture (test clips take
the test assignment, inner-validation clips the validation assignment).

    python scripts/redesign/idea7/defer.py --stage2 R --task esc50 --audio-stats OUT/audio_stats/esc50 \
        --out OUT/outputs/esc50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea5"))

from audio_shift import FIELD_CELLS, FIELD_SEEDS, cond_name, field_assignment  # noqa: E402

TASKS = {
    "esc50": {"tags": [f"astesc50_s{s}" for s in range(5)], "emb": "esc50", "audio": "esc50"},
    "esc5": {"tags": [f"astesc5_s{s}" for s in range(5)], "emb": "esc50", "audio": "esc50"},
    "urban": {"tags": [f"asturban_s{s}" for s in range(3)], "emb": "urban8k", "audio": "urban"},
}
DEVICES = ["n3g64", "w3g64", "n4", "w4g128", "w8g64"]
NEXT_UP = {"n3g64": "n4", "w3g64": "w4g128", "n4": "w8g64", "w4g128": "w8g64"}
HEADS = ["esc", "4"]
CLOUD = "fp32"
NI_MARGIN = 0.005
BUDGETS = {"b05": 0.05, "b10": 0.10, "b20": 0.20}
N_RANDOM, N_ORACLE = 500, 200
ECE_BINS = 15
FIELD = [cond_name(k, s) for k, s in FIELD_CELLS]


# ── metrics (Stage 2's definitions) ────────────────────────────────────────

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macro_f1(y, p):
    """sklearn's macro F1 (zero_division 0) over the labels in y or p."""
    labels, inv = np.unique(np.concatenate([y, p]), return_inverse=True)
    k = len(labels)
    cm = np.bincount(inv[:len(y)] * k + inv[len(y):], minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    return float(np.mean(2 * tp / (cm.sum(0) + cm.sum(1))))


def macro_f1_batch(y, P, C):
    """macro_f1 for every row of P [R, n] at once (same label rule)."""
    R = P.shape[0]
    off = (np.arange(R) * C)[:, None]
    flat = (off + P).ravel()
    pred_c = np.bincount(flat, minlength=R * C).reshape(R, C)
    tp = np.bincount(flat[(P == y[None, :]).ravel()], minlength=R * C).reshape(R, C)
    den = pred_c + np.bincount(y, minlength=C)[None, :]
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan)
    return np.nanmean(f1, 1)


def ece(prob, y, bins=ECE_BINS):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(tot)


def margin_score(prob):
    s = np.sort(prob, 1)
    return 1 - (s[:, -1] - s[:, -2])


def candidates(scores):
    u = np.unique(scores)
    return np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])


def threshold_for_rate(scores, rate):
    best = None
    for t in candidates(scores):
        r = float(np.mean(scores > t))
        key = (abs(r - rate), r)
        if best is None or key < best[0]:
            best = (key, t)
    return best[1]


def threshold_pre(scores, y, pd_, pc_, target):
    """Smallest inner-validation deferral rate whose F1 reaches target."""
    rows = []
    for t in candidates(scores):
        d = scores > t
        rows.append((float(d.mean()), macro_f1(y, np.where(d, pc_, pd_)), t))
    rows.sort(key=lambda r: (r[0], -r[1]))
    for r, f, t in rows:
        if f >= target - 1e-12:
            return t
    raise AssertionError("deferring every clip must reach the target")


def fit_temperature(z, y):
    def nll(logt):
        q = z / np.exp(logt)
        q = q - q.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(q).sum(1)) - q[np.arange(len(y)), y]))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def auroc(target, score):
    return float(roc_auc_score(target, score)) if 0 < target.sum() < len(target) else np.nan


# ── inputs ─────────────────────────────────────────────────────────────────

class Run:
    """Logits of one (seed, head) run on every (config, condition)."""

    def __init__(self, d: Path):
        self.d, self.cache = d, {}
        self.files = {p.stem for p in d.glob("*.npz")}

    def raw(self, cfg, cond):
        key = (cfg, cond)
        if key not in self.cache:
            z = np.load(self.d / f"{cfg}__{cond}.npz")
            iv = {int(k[7:]): z[k] for k in z.files if k.startswith("iv_idx_")}
            self.cache[key] = {"y": z["y"].astype(int), "fold": z["test_fold"].astype(int),
                               "z": z["test_logits"].astype(np.float64),
                               "iv": {f: (iv[f], z[f"iv_logits_{f}"].astype(np.float64)) for f in iv}}
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
            n = len(base["y"])
            at, av = field_arrays(n)
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
            self.cache[key] = {"y": base["y"], "fold": base["fold"], "z": z, "iv": ivs}
        return self.cache[key]


_FIELD = {}


def field_arrays(n):
    if n not in _FIELD:
        _FIELD[n] = tuple(np.array(field_assignment(n, FIELD_SEEDS[k])) for k in ("test", "val"))
    return _FIELD[n]


class Audio:
    """Audio statistics per (condition, clip); field rows from the assignments."""

    def __init__(self, d: Path):
        self.d, self.cache = d, {}
        clean = np.load(d / "clean.npz")
        self.features = clean["features"].tolist()
        self.n_samples = clean["n_samples"]

    def cond(self, c):
        if c not in self.cache:
            self.cache[c] = np.load(self.d / f"{c}.npz")["stats"]
        return self.cache[c]

    def rows(self, cond, idx, which):
        """Statistics of rows idx under cond; field: the test or val assignment."""
        if cond != "field":
            out = self.cond(cond)[idx]
        else:
            a = field_arrays(len(self.n_samples))[0 if which == "test" else 1]
            out = np.full((len(idx), len(self.features)), np.nan)
            for c in ["clean"] + FIELD:
                sel = a[idx] == c
                out[sel] = self.cond(c)[idx[sel]]
        if np.isnan(out).any():
            raise SystemExit(f"audio statistics missing for {cond}")
        return out


def fit_lr(X, t):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), t)
    return lambda Z: lr.predict_proba(sc.transform(Z))[:, 1]


def top_k(score, k):
    """Boolean mask of the k highest scores (stable: earlier rows win ties)."""
    m = np.zeros(len(score), bool)
    if k > 0:
        m[np.argsort(-score, kind="stable")[:k]] = True
    return m


def rng_for(*key):
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


def random_f1(y, pd_, pc_, k, C, rng):
    n = len(y)
    if k == 0:
        return macro_f1(y, pd_), 0.0
    if k == n:
        return macro_f1(y, pc_), 1.0
    U = rng.random((N_RANDOM, n))
    thr = np.partition(U, k - 1, axis=1)[:, k - 1]
    P = np.where(U <= thr[:, None], pc_[None, :], pd_[None, :])
    f = macro_f1_batch(y, P, C)
    return float(f.mean()), float(f.std(ddof=1) / np.sqrt(N_RANDOM))


def oracle_k_f1(y, pd_, pc_, rec, k, C, rng):
    ri = np.flatnonzero(rec)
    if len(ri) <= k:
        return macro_f1(y, np.where(rec, pc_, pd_))
    U = rng.random((N_ORACLE, len(ri)))
    thr = np.partition(U, k - 1, axis=1)[:, k - 1] if k > 0 else np.full(N_ORACLE, -1.0)
    P = np.tile(pd_, (N_ORACLE, 1))
    pick = U <= thr[:, None]
    rows, cols = np.nonzero(pick)
    P[rows, ri[cols]] = pc_[ri[cols]]
    return float(macro_f1_batch(y, P, C).mean())


# ── main ───────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage2", type=Path, required=True)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--audio-stats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    T = TASKS[args.task]
    scores = args.stage2 / "scores" / args.task
    audio = Audio(args.audio_stats)
    S = audio.features.index("snr_est_db"), audio.features.index("flatness_db")

    runs = {(int(tag.rsplit("_s", 1)[1]), h): Run(scores / tag / h) for tag in T["tags"] for h in HEADS}
    cfgs = DEVICES + [CLOUD]
    all_conds = sorted({f.split("__", 1)[1] for r in runs.values() for f in r.files})
    conds, pending = [], []
    for c in all_conds:
        ok = all(r.has(cfg, c) for r in runs.values() for cfg in cfgs)
        (conds if ok else pending).append(c)
    if all(r.has(cfg, "field") for r in runs.values() for cfg in cfgs):
        conds.append("field")
    pending = [c for c in pending if c not in FIELD or args.task != "urban"]
    if args.task != "urban":
        pending = sorted(set(pending) | {c for c in ("bg__20db", "pink__20db", "white__20db") if c not in conds})
    print(f"{args.task}: conditions {conds}; pending or partial {pending}", flush=True)

    # costs
    emb = args.stage2 / "emb" / T["emb"]
    side = {c: json.loads((emb / f"{c}__clean.json").read_text())["backbone_cost_per_clip"] for c in cfgs}
    st = pd.read_csv(args.stage2 / "analysis" / args.task / "static.csv")
    st = st[st["condition"] == "clean"].groupby(["config", "head"])["storage_bytes"].first()
    costs = {"backbone": {c: {"storage_MB": side[c]["storage_bits"] / 8e6, "bitops_T": side[c]["bitops"] / 1e12,
                              "weight_bits": side[c]["weight_bits"], "group": side[c]["group"],
                              "nested": side[c]["nested"]} for c in cfgs},
             "head_MB": {h: float(st[(CLOUD, h)] - side[CLOUD]["storage_bits"] / 8) / 1e6 for h in HEADS},
             "bits_per_sample": 16, "sample_rate": 16000,
             "mean_clip_bits": float(16 * audio.n_samples.mean()),
             "conditions": conds, "pending": pending}
    (args.out / "costs.json").write_text(json.dumps(costs, indent=1))
    clip_bits = 16 * audio.n_samples.astype(np.float64)

    pairs, devrows = [], []
    noise_models = {}
    for (seed, head), run in sorted(runs.items()):
        cloud = runs[(seed, "esc")]
        y_all = cloud.raw(CLOUD, "clean")["y"]
        C = int(y_all.max()) + 1
        folds = sorted(cloud.raw(CLOUD, "clean")["iv"])
        for dev in DEVICES:
            for fold in folds:
                idx, zi = run.get(dev, "clean")["iv"][fold]
                idx_c, zci = cloud.get(CLOUD, "clean")["iv"][fold]
                assert (idx == idx_c).all()
                yi = y_all[idx]
                pdi, pci = softmax(zi), softmax(zci)
                si = margin_score(pdi)
                f1_cloud_iv = macro_f1(yi, pci.argmax(1))
                ths = {"pre": threshold_pre(si, yi, pdi.argmax(1), pci.argmax(1), f1_cloud_iv - NI_MARGIN)}
                ths.update({k: threshold_for_rate(si, v) for k, v in BUDGETS.items()})
                iv_rate = {k: float(np.mean(si > t)) for k, t in ths.items()}
                temp = fit_temperature(zi, yi)

                # model-free noise detector: per (seed, fold), fit on inner validation under the field
                # validation assignment, target "clip corrupted"
                if (seed, fold) not in noise_models:
                    av = field_arrays(len(y_all))[1]
                    Xi = audio.rows("field", idx, "val")
                    noise_models[(seed, fold)] = fit_lr(Xi, (av[idx] != "clean").astype(int))
                noise_lr = noise_models[(seed, fold)]
                zfi = run.get(dev, "field")["iv"][fold][1]
                err_iv = (zfi.argmax(1) != yi).astype(int)
                if 0 < err_iv.sum() < len(err_iv):
                    err_lr = fit_lr(audio.rows("field", idx, "val"), err_iv)
                else:
                    err_lr = lambda Z: np.zeros(len(Z))  # noqa: E731

                for cond in conds:
                    rd, rc = run.get(dev, cond), cloud.get(CLOUD, cond)
                    m = rd["fold"] == fold
                    ti = np.flatnonzero(m)
                    y = rd["y"][m]
                    zd = rd["z"][m]
                    pdt, pct = softmax(zd), softmax(rc["z"][m])
                    pd_, pc_ = pdt.argmax(1), pct.argmax(1)
                    st_ = margin_score(pdt)
                    X = audio.rows(cond, ti, "test")
                    sc_noise, sc_err = noise_lr(X), err_lr(X)
                    sc_snr, sc_flat = -X[:, S[0]], X[:, S[1]]
                    wrong = pd_ != y
                    rec = wrong & (pc_ == y)
                    f1_dev, f1_cloud = macro_f1(y, pd_), macro_f1(y, pc_)
                    f1_oracle = macro_f1(y, np.where(rec, pc_, pd_))
                    nxt = NEXT_UP.get(dev)
                    f1_next = macro_f1(y, softmax(run.get(nxt, cond)["z"][m]).argmax(1)) if nxt else np.nan
                    ptemp = softmax(zd / temp)
                    devrows.append({
                        "task": args.task, "device": dev, "head": head, "seed": seed, "fold": fold,
                        "condition": cond, "n": len(y), "f1_device": f1_dev, "f1_cloud": f1_cloud,
                        "f1_next": f1_next, "next": nxt, "f1_oracle": f1_oracle, "oracle_rate": float(rec.mean()),
                        "err_rate": float(wrong.mean()), "rec_rate": float(rec.mean()),
                        "ece": ece(pdt, y), "ece_temp": ece(ptemp, y), "temperature": temp,
                        "conf_gap": float(pdt.max(1).mean() - (pd_ == y).mean()),
                        "conf_gap_temp": float(ptemp.max(1).mean() - (pd_ == y).mean()),
                        **{f"auc_wrong_{k}": auroc(wrong, v) for k, v in
                           [("margin", st_), ("noise_lr", sc_noise), ("snr_est", sc_snr), ("flatness", sc_flat),
                            ("err_lr", sc_err)]},
                        **{f"auc_rec_{k}": auroc(rec, v) for k, v in
                           [("margin", st_), ("noise_lr", sc_noise), ("snr_est", sc_snr), ("flatness", sc_flat),
                            ("err_lr", sc_err)]},
                    })
                    n_rec = int(rec.sum())
                    for rule, t in ths.items():
                        d = st_ > t
                        k = int(d.sum())
                        rng = rng_for(args.task, dev, head, seed, fold, cond, rule)
                        f1_rand, se_rand = random_f1(y, pd_, pc_, k, C, rng)
                        row = {"task": args.task, "device": dev, "head": head, "rule": rule, "seed": seed,
                               "fold": fold, "condition": cond, "threshold": t, "iv_rate": iv_rate[rule],
                               "n": len(y), "k": k, "rate": k / len(y), "f1_sys": macro_f1(y, np.where(d, pc_, pd_)),
                               "f1_random": f1_rand, "se_random": se_rand,
                               "f1_oracle_k": oracle_k_f1(y, pd_, pc_, rec, k, C, rng),
                               "n_rec": n_rec, "cap_sys": (d & rec).sum() / n_rec if n_rec else np.nan,
                               "cap_random": k / len(y) if n_rec else np.nan,
                               "bits_sent_per_1000": 1000 * clip_bits[ti][d].sum() / len(y),
                               "fixed": int((d & rec).sum()), "broken": int((d & ~wrong & (pc_ != y)).sum())}
                        for name, sc in [("noise_lr", sc_noise), ("snr_est", sc_snr), ("flatness", sc_flat),
                                         ("err_lr", sc_err)]:
                            dd = top_k(sc, k)
                            row[f"f1_{name}"] = macro_f1(y, np.where(dd, pc_, pd_))
                            row[f"cap_{name}"] = (dd & rec).sum() / n_rec if n_rec else np.nan
                        pairs.append(row)
            print(f"  seed {seed} head {head} {dev}: {time.time() - t_start:.0f}s", flush=True)

    pd.DataFrame(pairs).to_csv(args.out / "pairs.csv.gz", index=False)
    pd.DataFrame(devrows).to_csv(args.out / "device.csv.gz", index=False)
    print(f"wrote {args.out} in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
