"""Static front, uncertainty-gated precision cascades, the humility curve and
calibration for idea 5 Stage 2, from score_stage2.py outputs.

System = (backbone configuration, head). Heads: "esc" (FP32 head) and "4"
(4-bit head), trained on clean FP32-backbone embeddings, one run per seed.
Every number is per (seed, fold): a fold's test clips scored by the head of
that fold, macro F1 over the fold.

Conditions: every <config>__<condition> file, plus "field": each test clip
takes its condition from the test field assignment and each inner-validation
clip from the validation assignment (audio_shift.field_assignment).

Cascade (cheap, expensive, head, signal): every clip runs the cheap
configuration and the head; a clip escalates when its score, computed from
the cheap pass's softmax, is above a threshold, and then re-runs the
expensive configuration and the same head, whose prediction replaces the
cheap one. Signals: margin (1 - (p1 - p2)) and entropy. Thresholds come from
the fold's inner validation split only, scored through the same pipeline
under the same condition (field: the validation assignment):
  b05, b10, b20, b30   escalation rate on inner validation closest to 5, 10,
                       20, 30% (ties: fewer escalations)
  match                the lowest inner-validation escalation rate whose
                       cascade F1 reaches the inner-validation F1 of the cheap
                       configuration's W8 parent (w8, or w8g<G> for group
                       configurations), whichever the expensive target; when
                       none does, the rate with the highest
                       inner-validation F1 (match_reached false)
  clean_b10, clean_b20 the threshold of b10 / b20 set on the clean inner
                       validation split and applied to every condition (the
                       deployed rule: calibrated once, on clean audio)
Cost per clip: backbone bit-ops of the cheap pass plus its head, plus the
test escalation rate times the expensive pass and its head
(QWave.backbone_quant.cascade_cost, which also gives bits read and storage;
a nested child escalating to its own 8-bit parent stores the parent only).

The rules above were fixed before any test fold was scored (no max-F1
point: it escalated 35 to 53% of UrbanSound8K clips in Stage 1).

Records for scripts/redesign/front.py, one dataset key per (dataset,
condition), written as JSON lines in three families so the Bonferroni
correction runs over a declared set:
  primary    FP32 head, margin, cheap in --cheap, expensive = its W8
             parent, points b05..b30 and match
  secondary  the same points with FP32 as the expensive target, with the
             entropy gate, and with the 4-bit head
  deployed   clean-calibrated thresholds (clean_b10, clean_b20), FP32 head,
             margin, W8 parent
  primaryread  the primary points with weight bits read per clip in place of
             bit-ops as the cost (the memory-bound view; the records' bitops
             field then holds bits read)
Every family file carries every static system (both heads) as singles.

Also written: auroc.csv (AUROC of each signal for the test clips the cheap
pass gets wrong and its W8 parent gets right, FP32 head, per seed),
static.csv (every system, condition and pair), cascades.csv
(every cascade point and pair, with test escalation rate, bits read,
storage and ECE), humility.csv (escalation rate per condition under the
clean-calibrated thresholds) and calibration.csv (top-label ECE, 15 bins, and
mean confidence minus accuracy).

    python scripts/redesign/idea5/analyze_stage2.py --scores SCORES/esc5 --emb-dir EMB/esc50 \
        --runs-root OUT/outputs --tags astesc5_s0 astesc5_s1 astesc5_s2 --dataset esc5 \
        --cheap w4 n4 w3g128 n3g128 --out ANALYSIS/esc5
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from QWave.backbone_quant import PARENT_BITS, cascade_cost, parse_config  # noqa: E402
from audio_shift import FIELD_CELLS, FIELD_SEEDS, cond_name, field_assignment  # noqa: E402

BUDGETS = {"b05": 0.05, "b10": 0.10, "b20": 0.20, "b30": 0.30}
CLEAN_POINTS = {"clean_b10": 0.10, "clean_b20": 0.20}
SIGNALS = ("margin", "entropy")
ECE_BINS = 15


# ── metrics ────────────────────────────────────────────────────────────────

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macro_f1(y, p):
    """sklearn's f1_score(average='macro', zero_division=0): the mean over the
    labels in y or p of 2TP / (2TP + FP + FN)."""
    labels, inv = np.unique(np.concatenate([y, p]), return_inverse=True)
    k = len(labels)
    cm = np.bincount(inv[:len(y)] * k + inv[len(y):], minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    return float(np.mean(2 * tp / (cm.sum(0) + cm.sum(1))))


def ece(prob, y, bins=ECE_BINS):
    """Top-label expected calibration error, equal-width confidence bins."""
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(tot)


def conf_gap(prob, y):
    """Mean top-label confidence minus accuracy (negative: underconfident)."""
    return float(prob.max(1).mean() - (prob.argmax(1) == y).mean())


def signal(prob, kind):
    if kind == "margin":
        s = np.sort(prob, 1)
        return 1 - (s[:, -1] - s[:, -2])
    if kind == "entropy":
        return -(prob * np.log(np.clip(prob, 1e-12, 1))).sum(1)
    raise ValueError(kind)


def candidates(scores):
    """Escalate iff score > t: -inf escalates all, +inf none, one threshold
    between each pair of distinct scores."""
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


def threshold_match(scores, y, pc, pe, target):
    """Lowest escalation rate whose cascade F1 reaches target; else the best
    F1 (ties: fewer escalations). Returns (threshold, reached)."""
    rows = []
    for t in candidates(scores):
        esc = scores > t
        rows.append((float(esc.mean()), macro_f1(y, np.where(esc, pe, pc)), t))
    rows.sort(key=lambda r: (r[0], -r[1]))
    for r, f, t in rows:
        if f >= target - 1e-12:
            return t, True
    return max(rows, key=lambda r: (r[1], -r[0]))[2], False


# ── scores ─────────────────────────────────────────────────────────────────

class Scores:
    """Softmax outputs of one run (seed, head) on every (config, condition)."""

    def __init__(self, d: Path):
        self.d, self.cache = d, {}
        self.files = {p.stem for p in d.glob("*.npz")}

    def has(self, config, cond):
        if cond == "field":
            return all(f"{config}__{c}" in self.files for c in ["clean"] + field_cells())
        return f"{config}__{cond}" in self.files and self.complete(config, cond)

    def complete(self, config, cond):
        """Every test and inner-validation row scored (a field cell extracted
        for some clips only is not a condition of its own)."""
        r = self.raw(config, cond)
        return bool(np.isfinite(r["p"]).all() and all(np.isfinite(p).all() for _, p in r["iv"].values()))

    def raw(self, config, cond):
        key = (config, cond)
        if key not in self.cache:
            z = np.load(self.d / f"{config}__{cond}.npz")
            iv = {int(k[7:]): z[k] for k in z.files if k.startswith("iv_idx_")}
            self.cache[key] = {"y": z["y"], "fold": z["test_fold"].astype(int),
                               "p": softmax(z["test_logits"].astype(np.float64)),
                               "iv": {f: (iv[f], softmax(z[f"iv_logits_{f}"].astype(np.float64))) for f in iv}}
        return self.cache[key]

    def get(self, config, cond):
        if cond != "field":
            return self.raw(config, cond)
        key = (config, "field")
        if key not in self.cache:
            base = self.raw(config, "clean")
            n = len(base["y"])
            at = np.array(field_assignment(n, FIELD_SEEDS["test"]))
            av = np.array(field_assignment(n, FIELD_SEEDS["val"]))
            p = np.full_like(base["p"], np.nan)
            ivs = {f: (idx, np.full_like(pr, np.nan)) for f, (idx, pr) in base["iv"].items()}
            for c in ["clean"] + list(field_cells()):
                r = self.raw(config, c)
                m = at == c
                p[m] = r["p"][m]
                for f, (idx, pr) in ivs.items():
                    sel = av[idx] == c
                    pr[sel] = r["iv"][f][1][sel]
            for f, (idx, pr) in ivs.items():
                if np.isnan(pr).any():
                    raise SystemExit(f"{self.d}: field validation rows missing for {config}")
            if np.isnan(p).any():
                raise SystemExit(f"{self.d}: field test rows missing for {config}")
            self.cache[key] = {"y": base["y"], "fold": base["fold"], "p": p, "iv": ivs}
        return self.cache[key]


def field_cells():
    return [cond_name(k, s) for k, s in FIELD_CELLS]


def parent_of(config):
    spec, _ = parse_config(config)
    return f"w{PARENT_BITS}" + (f"g{spec.group}" if spec.group else "")


# ── costs ──────────────────────────────────────────────────────────────────

def backbone_costs(emb_dir: Path) -> dict:
    out = {}
    for j in emb_dir.glob("*__clean.json"):
        side = json.loads(j.read_text())
        out[side["config"]] = side["backbone_cost_per_clip"]
    return out


def head_costs(runs_root: Path, tag: str, kinds) -> dict:
    """{kind: (bit-ops, bytes)} of one fold's head (same size every fold)."""
    from omegaconf import OmegaConf

    from QWave.costs import cost, weight_bytes
    from scripts.benchmark import build_model

    out = {}
    for k in kinds:
        cfg = OmegaConf.load(runs_root / tag / k / "config.yaml")
        df = pd.read_csv(cfg.experiment.datasets.esc.csv, usecols=[cfg.experiment.datasets.esc.get("label_col",
                                                                                                    "class_id")])
        with contextlib.redirect_stdout(io.StringIO()):
            m = build_model(k, 768, int(df.iloc[:, 0].nunique()), cfg)
        out[k] = (float(cost(m)[1]), float(weight_bytes(m)))
    return out


# ── main ───────────────────────────────────────────────────────────────────

def seed_of(tag):
    return int(re.search(r"_s(\d+)$", tag).group(1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--heads", nargs="+", default=["esc", "4"])
    ap.add_argument("--cheap", nargs="+", required=True)
    ap.add_argument("--configs", nargs="*", default=None, help="static configurations (default: every one found)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    bcost = backbone_costs(args.emb_dir)
    hcost = head_costs(args.runs_root, args.tags[0], args.heads)
    runs = {(seed_of(t), h): Scores(args.scores / t / h) for t in args.tags for h in args.heads}
    any_run = next(iter(runs.values()))
    configs = args.configs or sorted({f.split("__")[0] for f in any_run.files})
    conds = sorted({f.split("__", 1)[1] for f in any_run.files})
    if all(any_run.has(c, "field") for c in args.cheap):
        conds.append("field")
    print(f"{args.dataset}: {len(configs)} configs, conditions {conds}", flush=True)

    static, casc = [], []
    for (seed, head), sc in runs.items():
        hb, hbytes = hcost[head]
        for cond in conds:
            for cfg in configs:
                if not sc.has(cfg, cond):
                    continue
                r = sc.get(cfg, cond)
                c = bcost[cfg]
                for fold in sorted(r["iv"]):
                    m = r["fold"] == fold
                    y, p = r["y"][m], r["p"][m]
                    static.append({"dataset": args.dataset, "condition": cond, "config": cfg, "head": head,
                                   "seed": seed, "fold": fold, "f1": macro_f1(y, p.argmax(1)), "ece": ece(p, y),
                                   "conf_gap": conf_gap(p, y),
                                   "bitops": c["bitops"] + hb, "storage_bytes": c["storage_bits"] / 8 + hbytes,
                                   "bits_read": c["bits_read"]})
            # cascades
            for cheap in args.cheap:
                for exp in (parent_of(cheap), "fp32"):
                    if not (sc.has(cheap, cond) and sc.has(exp, cond) and sc.has(parent_of(cheap), cond)):
                        continue
                    rc, re_, rp = sc.get(cheap, cond), sc.get(exp, cond), sc.get(parent_of(cheap), cond)
                    clean_ok = sc.has(cheap, "clean") and sc.has(exp, "clean")
                    cost_args = (bcost[cheap], bcost[exp])
                    for fold in sorted(rc["iv"]):
                        idx, pci = rc["iv"][fold]
                        pei = re_["iv"][fold][1]
                        yi = rc["y"][idx]
                        m = rc["fold"] == fold
                        y, pct, pet = rc["y"][m], rc["p"][m], re_["p"][m]
                        f1_parent_iv = macro_f1(yi, rp["iv"][fold][1].argmax(1))
                        for sig in SIGNALS:
                            si, st = signal(pci, sig), signal(pct, sig)
                            ths = {k: (threshold_for_rate(si, v), None) for k, v in BUDGETS.items()}
                            ths["match"] = threshold_match(si, yi, pci.argmax(1), pei.argmax(1), f1_parent_iv)
                            if clean_ok:
                                cl = sc.get(cheap, "clean")["iv"][fold][1]
                                scl = signal(cl, sig)
                                for k, v in CLEAN_POINTS.items():
                                    ths[k] = (threshold_for_rate(scl, v), None)
                            for point, (t, reached) in ths.items():
                                esc = st > t
                                pred = np.where(esc, pet.argmax(1), pct.argmax(1))
                                prob = np.where(esc[:, None], pet, pct)
                                rate = float(esc.mean())
                                cc = cascade_cost(*cost_args, rate, hb, hb)
                                casc.append({"dataset": args.dataset, "condition": cond, "cheap": cheap,
                                             "expensive": exp, "head": head, "signal": sig, "point": point,
                                             "seed": seed, "fold": fold, "f1": macro_f1(y, pred), "ece": ece(prob, y),
                                             "conf_gap": conf_gap(prob, y),
                                             "rate": rate, "iv_rate": float(np.mean(si > t)),
                                             "match_reached": reached, "bitops": cc["bitops"],
                                             "storage_bytes": cc["storage_bits"] / 8 + hbytes,
                                             "bits_read": cc["bits_read"], "nested": cc["nested"]})
        print(f"  seed {seed} head {head}: done", flush=True)

    # gate quality: AUROC of each signal for the test clips the cheap pass gets wrong and its W8 parent gets
    # right (pooled over the folds of a seed, FP32 head)
    from sklearn.metrics import roc_auc_score
    auc = []
    for (seed, head), sc in runs.items():
        if head != "esc":
            continue
        for cond in conds:
            for cheap in args.cheap:
                par = parent_of(cheap)
                if not (sc.has(cheap, cond) and sc.has(par, cond)):
                    continue
                rc, rp = sc.get(cheap, cond), sc.get(par, cond)
                m = rc["fold"] >= 0
                y, pc, pp = rc["y"][m], rc["p"][m], rp["p"][m]
                target = (pc.argmax(1) != y) & (pp.argmax(1) == y)
                row = {"condition": cond, "cheap": cheap, "seed": seed, "n_target": int(target.sum()),
                       "gap_pts": 100 * (np.mean(pp.argmax(1) == y) - np.mean(pc.argmax(1) == y))}
                for sig in SIGNALS:
                    row[sig] = float(roc_auc_score(target, signal(pc, sig))) if 0 < target.sum() < len(y) else np.nan
                auc.append(row)
    if auc:
        pd.DataFrame(auc).to_csv(args.out / "auroc.csv", index=False)

    S, C = pd.DataFrame(static), pd.DataFrame(casc)
    S.to_csv(args.out / "static.csv", index=False)
    C.to_csv(args.out / "cascades.csv", index=False)

    # records for front.py, per (dataset, condition)
    def single_records(cond, cost="bitops"):
        g = S[S["condition"] == cond]
        return [{"method": f"{r.config}+{r.head}", "dataset": f"{args.dataset}|{cond}", "seed": int(r.seed),
                 "fold": int(r.fold), "f1_macro": r.f1, "bitops_per_clip": getattr(r, cost),
                 "weight_bytes": r.storage_bytes, "operating_point": "static", "is_single": True}
                for r in g.itertuples()]

    def cascade_records(g, cost="bitops"):
        return [{"method": f"{r.cheap}>{r.expensive}+{r.head}|{r.signal}", "dataset": f"{args.dataset}|{r.condition}",
                 "seed": int(r.seed), "fold": int(r.fold), "f1_macro": r.f1, "bitops_per_clip": getattr(r, cost),
                 "weight_bytes": r.storage_bytes, "operating_point": r.point, "is_single": False}
                for r in g.itertuples()]

    fam_dir = args.out / "records"
    fam_dir.mkdir(exist_ok=True)
    primary_points = list(BUDGETS) + ["match"]
    if len(C):
        parent = C["expensive"] == C["cheap"].map(parent_of)
        main = (C["head"] == "esc") & (C["signal"] == "margin") & parent
        families = {
            "primary": C[main & C["point"].isin(primary_points)],
            "secondary": C[~main & C["point"].isin(primary_points)],
            "deployed": C[main & C["point"].isin(list(CLEAN_POINTS))],
            "primaryread": C[main & C["point"].isin(primary_points)],
        }
        for fam, g in families.items():
            cost = "bits_read" if fam == "primaryread" else "bitops"
            for cond in conds:
                recs = single_records(cond, cost) + cascade_records(g[g["condition"] == cond], cost)
                with open(fam_dir / f"{fam}__{cond}.jsonl", "w") as fh:
                    fh.writelines(json.dumps(r) + "\n" for r in recs)

    # humility: test escalation rate under clean-calibrated thresholds, per condition
    if len(C):
        hum = (C[C["point"].isin(list(CLEAN_POINTS))]
               .groupby(["cheap", "expensive", "head", "signal", "point", "condition"])
               .agg(rate=("rate", "mean"), f1=("f1", "mean"), n=("rate", "size")).reset_index())
        hum.to_csv(args.out / "humility.csv", index=False)
    cal = S.groupby(["condition", "config", "head"]).agg(ece=("ece", "mean"), conf_gap=("conf_gap", "mean"),
                                                         f1=("f1", "mean")).reset_index()
    cal.to_csv(args.out / "calibration.csv", index=False)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
