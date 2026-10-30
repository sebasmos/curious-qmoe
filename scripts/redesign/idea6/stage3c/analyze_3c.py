"""Idea 6 Stage 3c analysis of one task, as fixed in scripts/redesign/idea6/stage3c/ANALYSIS_PLAN.md (commit
47b0cb7): Stage 3b's analysis (analyze_3b.py, whose helpers are imported) with the EfficientAT siblings mn05, mn20
and dymn04 as statics and cheap tiers (P3), the real-int8 MN scored two ways as a static and as a substitute cheap
tier (P4), and the exact-length UrbanSound8K mn10 as a static and a substitute cheap tier (P5).

Cheap sources (per-clip logits in Stage 2's ScoreDir layout):
  eff            Stage 2   --s2/scores/<task>/effp_<task>_s<seed>/<kind>/<fp32|w8a8>__<cond>.npz
  mn04, mn10     Stage 3b  --s3b/scores_cnn/<task>/<m>_<task>_s<seed>/<kind>/<m>_<bb>__<cond>.npz
  mn05 mn20 dymn04         --r/scores_cnn/<task>/...
  <m>int8a / <m>int8b      --r/scores_int8a|b/<task>/<m>[r]_<task>_s<seed>/<kind>/<m>_int8__<cond>.npz
                           (inner-validation logits exist on clean only; the analysis reads no other)
  mn10unp (urban)          --r/scores_unp/urban/mn10_urban_s<seed>/<kind>/mn10_<bb>__<cond>.npz
AST tiers: Stage 3b's (1024-frame from idea 5 Stage 2, 512/400 from Stage 2, 256/384 and the two-window mean from
Stage 3b).

Hull sets: "stage3b" (Stage 2's and Stage 3b's statics; E4 reproduction), "new" (plus the 24 sibling statics; P3,
S1, S4, E2), "new+int8a", "new+int8b" (P4b, S5), "new+unp" (P5b).

    python scripts/redesign/idea6/stage3c/analyze_3c.py --task esc50 --r R --s3b S3B --s2 S2 --i5 I5 --i7 I7

Follow-up mode (--unp-retrained DIR, urban only; the Stage 3c folder given as --s3c when --r is a new folder): the
exact-length mn10 scored with heads retrained on the exact-length embeddings (DIR/mn10unp_urban_s<seed>/<kind>/
mn10_<bb>__<cond>.npz) replaces the padded mn10 in every hull set under the name "mn10unpr", priced at exact length;
the padded mn10 stays loaded as "mn10pad" outside every hull for the static comparison (padding_retrained.csv). The
int8 and P5a sources are not loaded, E4 is not checked. Family "P5c": Stage 3b's P2 design and Stage 3c's P3 design
with the retrained exact-length tier; family "P3": this run's own clean-inner-validation pick over every design.
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
S3BDIR = HERE.parent / "stage3b"
S2DIR = HERE.parent / "stage2"
ROOT = HERE.parents[3]
for p in (ROOT, S2DIR, S3BDIR, HERE.parent):
    sys.path.insert(0, str(p))

import analyze as A  # noqa: E402  Stage 2's helpers
from analyze_3b import AST_HEADS, BUDGETS, CASC_CONDS, CHEAP_CONDS, FULL_TIERS, KINDS, NEW_CFGS, PRICE, RULES  # noqa: E402
from analyze_3b import S2_SHORT, TASKS, ni_test  # noqa: E402
from audio_shift import cond_name  # noqa: E402
from idea5b_common import mix_expected_f1  # noqa: E402

CNN_S3B = {"mn04": 384, "mn10": 960}
CNN_NEW = {"mn05": 480, "mn20": 1920, "dymn04": 384}
INT8_BASES = ["mn10", "mn04", "mn05", "mn20"]
P3_CHEAP = [("eff", "w8a8")] + [(m, "w8a8") for m in [*CNN_S3B, *CNN_NEW]]
P2_CHEAP = [("eff", "w8a8"), ("mn04", "w8a8"), ("mn10", "w8a8")]


class TestOnlyIVScoreDir(A.ScoreDir):
    """A ScoreDir whose inner-validation logits exist on clean only (the real-int8 sources)."""

    def complete(self, cfg, cond):
        if f"{cfg}__{cond}" not in self.files:
            return False
        r = self.raw(cfg, cond)
        if cond == "clean":
            return bool(np.isfinite(r["z"]).all() and all(np.isfinite(v).all() for _, v in r["iv"].values()))
        return bool(np.isfinite(r["z"]).all())

    def get(self, cfg, cond):
        if cond != "field":
            return self.raw(cfg, cond)
        key = (cfg, "field")
        if key not in self.cache:
            base = self.raw(cfg, "clean")
            at, _ = A.field_arrays(len(base["y"]))
            z = np.full_like(base["z"], np.nan)
            for c in ["clean"] + A.FIELD:
                r = self.raw(cfg, c)
                m = at == c
                z[m] = r["z"][m]
            if np.isnan(z).any():
                raise SystemExit(f"{self.d}: field rows missing for {cfg}")
            ivs = {f: (idx, np.full_like(v, np.nan)) for f, (idx, v) in base["iv"].items()}
            self.cache[key] = {"y": base["y"], "fold": base["fold"], "z": z, "iv": ivs, "crc": base["crc"]}
        return self.cache[key]


def cheap_name(src, bb, k):
    return f"eff:{bb}:{k}" if src == "eff" else f"cnn:{src}:{bb}:{k}"


def ci95(fm):
    k = len(fm)
    if k < 2 or fm.std(ddof=1) == 0:
        return float(fm.mean()), float(fm.mean())
    h = stats.t.ppf(0.975, k - 1) * fm.std(ddof=1) / np.sqrt(k)
    return float(fm.mean() - h), float(fm.mean() + h)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--r", required=True, type=Path)
    ap.add_argument("--s3b", required=True, type=Path)
    ap.add_argument("--s2", required=True, type=Path)
    ap.add_argument("--i5", required=True, type=Path)
    ap.add_argument("--i7", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-mean2", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="code check while siblings are incomplete: drop missing sibling sources, write to --out")
    ap.add_argument("--unp-retrained", type=Path, default=None,
                    help="urban only: scores of the exact-length mn10 with heads retrained on exact-length embeddings; "
                         "they replace the padded mn10 in every hull set (see the module docstring)")
    ap.add_argument("--s3c", type=Path, default=None,
                    help="the Stage 3c results folder when --r is a follow-up folder (siblings, sidecars, picks)")
    a = ap.parse_args(argv)
    if a.check_only and a.out is None:
        raise SystemExit("--check-only needs --out outside the results folder")
    unp = a.unp_retrained is not None
    if unp and a.task != "urban":
        raise SystemExit("--unp-retrained is for the urban task only")
    r3c = a.s3c or a.r
    sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6"))
    t0 = time.time()
    task = a.task
    T = TASKS[task]
    K, seeds, SL = T["K"], T["seeds"], T["short"]
    ds = T["ds"]
    out = a.out or a.r / "analysis" / task
    out.mkdir(parents=True, exist_ok=True)
    esc = ds == "esc50"

    # ── cheap sources ───────────────────────────────────────────────────────
    # src -> {"dirs": {(seed, kind): ScoreDir}, "cfg": {bb: config prefix}, "dim": int, "side": {bb: clean sidecar},
    #         "kinds": [kinds with files for every seed], "set": hull set that adds the source}
    sources = {}

    def add(src, root, tag, cfg, dim, side, cls=A.ScoreDir, hull_set="stage3b", kinds=KINDS):
        dirs = {(s, k): cls(root / tag.format(s) / k) for s in seeds for k in kinds}
        have = [k for k in kinds if all(dirs[(s, k)].files for s in seeds)]
        if not have:
            return
        sources[src] = {"dirs": dirs, "cfg": cfg, "dim": dim, "side": side, "kinds": have, "set": hull_set}

    add("eff", a.s2 / "scores" / task, f"effp_{task}_s{{}}", {"fp32": "fp32", "w8a8": "w8a8"}, 1536,
        {bb: a.s2 / "emb" / ds / f"{bb}__clean.json" for bb in ("fp32", "w8a8")})
    unp_side = {bb: r3c / "emb_cnn" / "urban8k_unpadded" / f"mn10_{bb}__clean.json" for bb in ("fp32", "w8a8")}
    for m, d in CNN_S3B.items():
        name = "mn10pad" if unp and m == "mn10" else m
        add(name, a.s3b / "scores_cnn" / task, f"{m}_{task}_s{{}}", {bb: f"{m}_{bb}" for bb in ("fp32", "w8a8")}, d,
            {bb: a.s3b / "emb_cnn" / ds / f"{m}_{bb}__clean.json" for bb in ("fp32", "w8a8")},
            hull_set="excluded" if name == "mn10pad" else "stage3b")
    if unp:
        add("mn10unpr", a.unp_retrained, "mn10unp_urban_s{}", {bb: f"mn10_{bb}" for bb in ("fp32", "w8a8")}, 960,
            unp_side, hull_set="stage3b")
        if "mn10unpr" not in sources:
            raise SystemExit(f"{a.unp_retrained}: no retrained exact-length scores")
    for m, d in CNN_NEW.items():
        add(m, r3c / "scores_cnn" / task, f"{m}_{task}_s{{}}", {bb: f"{m}_{bb}" for bb in ("fp32", "w8a8")}, d,
            {bb: r3c / "emb_cnn" / ds / f"{m}_{bb}__clean.json" for bb in ("fp32", "w8a8")}, hull_set="new")
    for m in INT8_BASES if not unp else []:
        d = {**CNN_S3B, **CNN_NEW}.get(m)
        w8 = (a.s3b if m in CNN_S3B else r3c) / "emb_cnn" / ds / f"{m}_w8a8__clean.json"
        if (r3c / "scores_int8a" / task).exists():
            add(f"{m}int8a", r3c / "scores_int8a" / task, f"{m}_{task}_s{{}}", {"int8": f"{m}_int8"}, d, {"int8": w8},
                TestOnlyIVScoreDir, "new+int8a")
        if (r3c / "scores_int8b" / task).exists():
            add(f"{m}int8b", r3c / "scores_int8b" / task, f"{m}r_{task}_s{{}}", {"int8": f"{m}_int8"}, d, {"int8": w8},
                TestOnlyIVScoreDir, "new+int8b")
    if task == "urban" and not unp and (r3c / "scores_unp" / task).exists():
        add("mn10unp", r3c / "scores_unp" / task, "mn10_urban_s{}", {bb: f"mn10_{bb}" for bb in ("fp32", "w8a8")}, 960,
            unp_side, hull_set="new+unp")
    subs = [s for s in sources if s.endswith(("int8a", "int8b", "unp"))]

    def ren(x):
        """A design's or family's cheap source in this run: the retrained exact-length mn10 stands in for mn10."""
        return "mn10unpr" if unp and x == "mn10" else x

    for m in CNN_NEW:
        if m not in sources:
            if a.check_only:
                print(f"{task}: check-only, {m} has no scores and is dropped", flush=True)
                continue
            raise SystemExit(f"{task}: {m} has no scores; the plan needs every sibling complete")

    def cheap_get(src, bb, s, k, cond):
        return sources[src]["dirs"][(s, k)].get(sources[src]["cfg"][bb], cond)

    def cheap_has(src, bb, cond):
        S_ = sources[src]
        return all(S_["dirs"][(s, k)].has(S_["cfg"][bb], cond) for s in seeds for k in S_["kinds"])

    # ── AST tiers (as Stage 3b) ─────────────────────────────────────────────
    full_root = a.s2 / "scores_ast1024" / task if task == "esc5off" else a.i5 / "scores" / task
    full_tag = "astoff1024_esc5off_s{}" if task == "esc5off" else f"ast{task}_s{{}}"
    groups = {
        "full": ({(s, h): A.ScoreDir(full_root / full_tag.format(s) / h) for s in seeds for h in AST_HEADS}, None, ""),
        "s2short": ({(s, h): A.ScoreDir(a.s2 / "scores_ast_short" / task / f"astshort_{task}_s{s}" / h)
                     for s in seeds for h in AST_HEADS}, S2_SHORT, f"@{SL}"),
        "a256": ({(s, h): A.ScoreDir(a.s3b / "scores_ast256" / task / f"ast256_{task}_s{s}" / h)
                  for s in seeds for h in AST_HEADS}, NEW_CFGS, "@256"),
        "a384": ({(s, h): A.ScoreDir(a.s3b / "scores_ast384" / task / f"ast384_{task}_s{s}" / h)
                  for s in seeds for h in AST_HEADS}, NEW_CFGS, "@384"),
    }
    if esc and not a.no_mean2:
        groups["m2"] = ({(s, h): A.ScoreDir(a.s3b / "scores_astm2" / task / f"astm2_{task}_s{s}" / h)
                         for s in seeds for h in AST_HEADS}, NEW_CFGS, "@256x2")
    src_ast = {}
    for g, (dirs, cfgs, sfx) in groups.items():
        if cfgs is None:
            cfgs = sorted({f.split("__")[0] for f in dirs[(seeds[0], "esc")].files})
        for c in cfgs:
            src_ast[c + sfx] = (dirs, c, g)

    def ast_has(name, cond):
        dirs, base, _ = src_ast[name]
        return all(d.has(base, cond) for d in dirs.values())

    def ast_get(name, s, h, cond):
        dirs, base, _ = src_ast[name]
        return dirs[(s, h)].get(base, cond)

    conds = CHEAP_CONDS + ["field"]
    cheap_all = [(x, bb) for x in sources for bb in sources[x]["cfg"]]
    cheap_conds = {cb: [c for c in conds if cheap_has(*cb, c)] for cb in cheap_all}
    ast_conds = {n: [c for c in conds if ast_has(n, c)] for n in src_ast}
    missing = [f"{x}:{bb}" for x, bb in cheap_all if not all(c in cheap_conds[(x, bb)] for c in CASC_CONDS)]
    if missing:
        raise SystemExit(f"{task}: incomplete systems on clean or field: {missing}")
    tiers = [n for n in FULL_TIERS if n in src_ast] + [f"{c}@{SL}" for c in S2_SHORT]
    tiers += [n for n, v in src_ast.items() if v[2] in ("a256", "a384", "m2")]
    if not all(all(c in ast_conds[t] for c in CASC_CONDS) for t in tiers):
        raise SystemExit(f"{task}: a tier lacks clean or field")
    p3_cheap = [(ren(x), bb) for x, bb in P3_CHEAP if ren(x) in sources]
    p2_cheap = [(ren(x), bb) for x, bb in P2_CHEAP]
    cheap_casc = [(x, "w8a8") for x, _ in p3_cheap] + [(x, bb) for x in subs for bb in sources[x]["cfg"]
                                                       if bb in ("w8a8", "int8")]
    print(f"{task}: cheap sources {list(sources)}, {len(src_ast)} AST systems, tiers {len(tiers)}", flush=True)

    # ── costs (priced) ──────────────────────────────────────────────────────
    from costs_idea6 import head_costs

    with contextlib.redirect_stdout(io.StringIO()):
        hk = "esc5" if task == "esc5off" else task
        dims = sorted({1536, 768, *(S_["dim"] for S_ in sources.values())})
        hc = {d: head_costs(d, KINDS)[hk] for d in dims}
    eff_emb = a.s2 / "emb" / ds
    lengths = np.load(eff_emb / "lengths__clean.npy")
    fbank = float(np.mean([A.ast_fbank_frontend_macs(int(n)) for n in lengths])) * 1024
    cheap_cost, crc_side = {}, {}
    for x, S_ in sources.items():
        for bb, sp in S_["side"].items():
            sd = json.loads(sp.read_text())
            cc = sd["cost_per_clip"]
            crc_side[(x, bb)] = sd["names_crc32"]
            for k in S_["kinds"]:
                cheap_cost[(x, bb, k)] = cc["frontend_bitops_mean"] + cc["backbone_bitops_rescale1024"] + hc[S_["dim"]][k]["bitops"]
    ast_cost, ast_side = {}, {}
    from idea5b_common import bitops as ast_bitops

    for n, (dirs, base, g) in src_ast.items():
        if g == "full":
            bo = ast_bitops(base, PRICE)
        else:
            L = {"s2short": SL, "a256": 256, "a384": 384, "m2": 256}[g]
            P_, T_ = A.short_tokens(L)
            mult = 2 if g == "m2" else 1
            bo = mult * A.ast_bitops_tok(base, PRICE, T_, P_)
            side_dir = {"s2short": a.s2 / "emb_ast_short" / ds, "a256": a.s3b / "emb_ast" / f"{ds}_256",
                        "a384": a.s3b / "emb_ast" / f"{ds}_384", "m2": a.s3b / "emb_ast" / f"{ds}_256x2"}[g]
            s_ = json.loads((side_dir / f"{base}__clean.json").read_text())
            un = mult * A.ast_bitops_tok(base, 0.0, *A.short_tokens(L)[::-1])
            if abs(s_["backbone_cost_per_clip"]["bitops"] - un) > 1e-6 * un or s_["max_length"] != L:
                raise SystemExit(f"{n}: sidecar bit-ops differ")
            ast_side[n] = s_["names_crc32"]
        for h in AST_HEADS:
            ast_cost[(n, h)] = fbank + bo + hc[768][h]["bitops"]
    crc_ref = json.loads((eff_emb / "fp32__clean.json").read_text())["names_crc32"]
    if any(v != crc_ref for v in crc_side.values()) or any(v != crc_ref for v in ast_side.values()):
        raise SystemExit(f"{task}: a names checksum differs: {crc_side} {ast_side}")
    (out / "costs.json").write_text(json.dumps({
        "cheap": {cheap_name(*kk): v for kk, v in cheap_cost.items()},
        "ast": {f"{n}+{h}": v for (n, h), v in ast_cost.items()}, "fbank_bitops_mean": fbank,
        "heads": {str(d): v for d, v in hc.items()}}, indent=1))

    def sys_cost(name):
        if name.startswith("ast:"):
            n, h = name[4:].split("+")
            return ast_cost[(n, h)]
        parts = name.split(":")
        return cheap_cost[("eff", parts[1], parts[2]) if parts[0] == "eff" else (parts[1], parts[2], parts[3])]

    def sys_set(name):
        """The smallest hull set a static belongs to."""
        if name.startswith("ast:") or name.startswith("eff:"):
            return "stage3b"
        return sources[name.split(":")[1]]["set"]

    SET_ORDER = {"stage3b": ["stage3b"], "new": ["stage3b", "new"], "new+int8a": ["stage3b", "new", "new+int8a"],
                 "new+int8b": ["stage3b", "new", "new+int8b"], "new+unp": ["stage3b", "new", "new+unp"]}

    # ── one pass over seeds and folds ───────────────────────────────────────
    static, sel_rows, casc, esc_rows = [], [], [], []
    preds, ys, yiv, iv_static, ivx = {}, {}, {}, {}, {}
    acs, escs, decs, au = {}, {}, {}, {}
    thr_all = {}
    ref0 = cheap_get("eff", "fp32", seeds[0], "esc", "clean")
    folds = sorted(ref0["iv"])
    n_te = np.array([np.sum(ref0["fold"] == f) for f in folds])
    A.NB_RATIO["task"] = float(np.mean(n_te / (len(ref0["y"]) - n_te)))
    for s in seeds:
        ref = cheap_get("eff", "fp32", s, "esc", "clean")
        for x, bb in cheap_all:
            for k in sources[x]["kinds"]:
                for c in CASC_CONDS:
                    r_ = cheap_get(x, bb, s, k, c)
                    if not ((r_["y"] == ref["y"]).all() and (r_["fold"] == ref["fold"]).all() and sorted(r_["iv"]) == folds
                            and all(np.array_equal(r_["iv"][f][0], ref["iv"][f][0]) for f in folds)):
                        raise SystemExit(f"{x}:{bb}:{k} {c}: rows differ")
                    if r_["crc"] is not None and r_["crc"] != crc_ref:
                        raise SystemExit(f"{x}:{bb}:{k} {c}: names checksum differs")
        for n in src_ast:
            for h in AST_HEADS:
                for c in [c for c in CASC_CONDS if c in ast_conds[n]]:
                    r_ = ast_get(n, s, h, c)
                    if not ((r_["y"] == ref["y"]).all() and (r_["fold"] == ref["fold"]).all()
                            and all(np.array_equal(r_["iv"][f][0], ref["iv"][f][0]) for f in folds)):
                        raise SystemExit(f"{n}+{h} {c}: rows differ")
        y_all = ref["y"]
        for fold in folds:
            te = np.flatnonzero(ref["fold"] == fold)
            iv = ref["iv"][fold][0]
            y, yv = y_all[te], y_all[iv]
            key = (s, fold)
            ys[key], yiv[key], ivx[key] = y, yv, iv
            civ = {}
            for x, bb in cheap_all:
                for k in sources[x]["kinds"]:
                    p = A.softmax(cheap_get(x, bb, s, k, "clean")["iv"][fold][1])
                    civ[(x, bb, k)] = p
                    pr = p.argmax(1)
                    iv_static[(cheap_name(x, bb, k), s, fold)] = (A.f1_fast(yv, pr, K), pr)
            aiv = {}
            for n in [n for n in src_ast if "clean" in ast_conds[n]]:
                for h in AST_HEADS:
                    p = A.softmax(ast_get(n, s, h, "clean")["iv"][fold][1])
                    aiv[(n, h)] = p
                    pr = p.argmax(1)
                    iv_static[(f"ast:{n}+{h}", s, fold)] = (A.f1_fast(yv, pr, K), pr)
            for x, bb in cheap_casc:
                for k in sources[x]["kinds"]:
                    p = civ[(x, bb, k)]
                    m, ac = A.margin(p), p.argmax(1)
                    for b, rate in BUDGETS.items():
                        t_ = A.thr_rate(m, rate)
                        thr_all[(x, bb, k, b, s, fold)] = t_
                        escm = m < t_
                        for e in tiers:
                            pe = aiv[(e, "esc")]
                            for r in RULES:
                                de = pe.argmax(1) if r == "ast" else ((p + pe) / 2).argmax(1)
                                sel_rows.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r,
                                                 "budget": b, "seed": s, "fold": fold, "iv_rate": float(escm.mean()),
                                                 "f1_iv": A.f1_fast(yv, np.where(escm, de, ac), K)})
            for c in conds:
                cp = {}
                for x, bb in cheap_all:
                    if c not in cheap_conds[(x, bb)]:
                        continue
                    for k in sources[x]["kinds"]:
                        z = cheap_get(x, bb, s, k, c)["z"][te]
                        p = A.softmax(z)
                        pr = p.argmax(1)
                        name = cheap_name(x, bb, k)
                        if c in CASC_CONDS:
                            preds[(name, c, s, fold)] = pr.astype(np.int16)
                        static.append({"system": name, "condition": c, "seed": s, "fold": fold,
                                       "f1": A.f1_fast(y, pr, K), "acc": float((pr == y).mean()),
                                       "cost": cheap_cost[(x, bb, k)]})
                        if (x, bb) in cheap_casc:
                            cp[(x, bb, k)] = p
                            m = A.margin(p)
                            for b in BUDGETS:
                                esc_rows.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "budget": b, "condition": c,
                                                 "seed": s, "fold": fold,
                                                 "rate": float((m < thr_all[(x, bb, k, b, s, fold)]).mean()),
                                                 "error": float((pr != y).mean())})
                            a_ = au.setdefault((x, bb, k, c, s), {"score": [], "wrong": []})
                            a_["score"].append(1 - m)
                            a_["wrong"].append(pr != y)
                ap_ = {}
                for n in src_ast:
                    if c not in ast_conds[n]:
                        continue
                    for h in AST_HEADS:
                        p = A.softmax(ast_get(n, s, h, c)["z"][te])
                        pr = p.argmax(1)
                        name = f"ast:{n}+{h}"
                        if c in CASC_CONDS:
                            preds[(name, c, s, fold)] = pr.astype(np.int16)
                        static.append({"system": name, "condition": c, "seed": s, "fold": fold,
                                       "f1": A.f1_fast(y, pr, K), "acc": float((pr == y).mean()),
                                       "cost": ast_cost[(n, h)]})
                        if h == "esc":
                            ap_[n] = p
                if c not in CASC_CONDS:
                    continue
                for (x, bb, k), p in cp.items():
                    m, ac = A.margin(p), p.argmax(1)
                    acs[(x, bb, k, c, s, fold)] = ac.astype(np.int16)
                    qs = {}
                    for e in tiers:
                        pe = ap_[e]
                        for r in RULES:
                            q = pe if r == "ast" else (p + pe) / 2
                            qs[(e, r)] = (q.argmax(1), q.max(1))
                            decs[(x, bb, k, e, r, c, s, fold)] = qs[(e, r)][0].astype(np.int16)
                    for b in BUDGETS:
                        escm = m < thr_all[(x, bb, k, b, s, fold)]
                        escs[(x, bb, k, b, c, s, fold)] = escm
                        rate = float(escm.mean())
                        for (e, r), (de, qmax) in qs.items():
                            pred = np.where(escm, de, ac)
                            casc.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r,
                                         "budget": b, "condition": c, "seed": s, "fold": fold,
                                         "f1": A.f1_fast(y, pred, K), "acc": float((pred == y).mean()), "rate": rate,
                                         "ece": A.ece_conf(np.where(escm, qmax, p.max(1)), pred == y),
                                         "cost": cheap_cost[(x, bb, k)] + rate * ast_cost[(e, "esc")]})
            print(f"{task} seed {s} fold {fold}: {time.time() - t0:.0f}s", flush=True)
        for S_ in sources.values():
            for d in S_["dirs"].values():
                d.cache.clear()
        for g in groups.values():
            for d in g[0].values():
                d.cache.clear()

    S = pd.DataFrame(static)
    S["set"] = S.system.map(sys_set)
    S.to_csv(out / "static.csv", index=False)
    CA = pd.DataFrame(casc)
    CA.to_csv(out / "cascades.csv.gz", index=False)
    pd.DataFrame(esc_rows).to_csv(out / "escalation.csv", index=False)
    rows = []
    for (x, bb, k, c, s), a_ in au.items():
        sc, w = np.concatenate(a_["score"]), np.concatenate(a_["wrong"])
        rows.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "condition": c, "seed": s,
                     "auroc_wrong": roc_auc_score(w, sc) if 0 < w.sum() < len(w) else np.nan,
                     "error": float(w.mean())})
    pd.DataFrame(rows).to_csv(out / "auroc_wrong.csv", index=False)

    # ── selection on clean inner validation: P2 (Stage 3b set, reproduction) and P3 (new set) ──
    SR = pd.DataFrame(sel_rows)
    st = pd.DataFrame([{"system": nm, "seed": s, "fold": f, "f1": v[0], "cost": sys_cost(nm), "set": sys_set(nm)}
                       for (nm, s, f), v in iv_static.items()])
    sp_iv_all = st.groupby(["system", "set"]).agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
    sp_iv_all.to_csv(out / "static_iv.csv", index=False)

    def select(cheap_fam, which, fname):
        sp_iv = sp_iv_all[sp_iv_all["set"].isin(SET_ORDER[which])]
        rng = np.random.default_rng(A.RNG_IV)
        sel_out = []
        fam = {(x, bb) for x, bb in cheap_fam}
        for (x, bb, k, e, r, b), g in SR.groupby(["cheap_src", "cheap_bb", "cheap", "expensive", "rule", "budget"],
                                                  sort=False):
            if (x, bb) not in fam:
                continue
            g = g.sort_values(["seed", "fold"])
            c_iv = float(np.mean(cheap_cost[(x, bb, k)] + g.iv_rate.to_numpy() * ast_cost[(e, "esc")]))
            v_interp, lo, hi, lam = A.hull_pair(sp_iv, c_iv)
            hull = []
            for s, f in zip(g.seed, g.fold):
                plo = iv_static[(lo, s, f)][1]
                if lam == 0.0:
                    hull.append(iv_static[(lo, s, f)][0])
                    continue
                phi = iv_static[(hi, s, f)][1]
                draws = rng.random((A.DRAWS, len(plo))) < lam
                hull.append(float(A.f1_batch(yiv[(s, f)], np.where(draws, phi[None, :], plo[None, :]), K).mean()))
            d = g.f1_iv.to_numpy() - np.array(hull)
            sel_out.append({"cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r, "budget": b,
                            "f1_iv": g.f1_iv.mean(), "iv_rate": g.iv_rate.mean(), "cost_iv": c_iv, "hull_lo": lo,
                            "hull_hi": hi, "lam": lam, "hull_f1_iv": float(np.mean(hull)), "gain_iv": float(d.mean())})
        SO = pd.DataFrame(sel_out).sort_values(["gain_iv", "cost_iv"], ascending=[False, True])
        SO.to_csv(out / fname, index=False)
        top = SO.iloc[0]
        return SO, (top.cheap_src, top.cheap_bb, top.cheap, top.expensive, top.rule, top.budget)

    SO2, p2_re = select(p2_cheap, "stage3b", "selection_stage3b.csv")
    SO3, p3 = select(p3_cheap, "new", "selection.csv")
    s3b_picks = json.loads((a.s3b / "analysis" / task / "picks.json").read_text())
    p1 = tuple(s3b_picks["P1"]["design"])
    p2 = tuple(s3b_picks["P2"]["design"])
    p2 = (ren(p2[0]), *p2[1:])
    p2_re = tuple(str(v) for v in p2_re)
    if p2_re != tuple(str(v) for v in p2):
        if not unp:
            raise SystemExit(f"{task}: P2 reselected with Stage 3b's systems gives {p2_re}, Stage 3b picked {p2}")
        print(f"{task}: P2 reselected with the exact-length tier gives {p2_re}, Stage 3b picked {p2}", flush=True)
    top3 = SO3.iloc[0]
    picks = {"P1": {"design": list(p1), "source": "Stage 3b picks.json P1 (Stage 2's H1), unchanged"},
             "P2": {"design": list(p2), "source": "Stage 3b picks.json P2, reselected here with the Stage 3b systems: equal"
                    if not unp else "Stage 3b picks.json P2 with the retrained exact-length mn10 as its cheap tier",
                    "reselected": list(p2_re),
                    "gain_iv_reselected": float(SO2.iloc[0].gain_iv), "gain_iv_stage3b": s3b_picks["P2"]["gain_iv"]},
             "P3": {"design": [str(v) for v in p3], "gain_iv": float(top3.gain_iv),
                    "runner_up_gain_iv": float(SO3.iloc[1].gain_iv),
                    "runner_up": [str(v) for v in SO3.iloc[1][["cheap_src", "cheap_bb", "cheap", "expensive", "rule", "budget"]]],
                    "cost_iv": float(top3.cost_iv), "f1_iv": float(top3.f1_iv), "iv_rate": float(top3.iv_rate),
                    "hull_iv": [top3.hull_lo, top3.hull_hi, float(top3.lam)],
                    "n_designs": int(len(SO3))}}
    (out / "picks.json").write_text(json.dumps(picks, indent=1))
    print(f"{task}: P1 {p1}  P2 {p2}  P3 {p3} (gain_iv {top3.gain_iv:.4f})", flush=True)

    # ── test comparisons ────────────────────────────────────────────────────
    from analyze import AudioStats, field_arrays, fit_lr, top_k, oracle_pred  # noqa: E402

    stems = [str(x) for x in np.load(a.i7 / "audio_stats" / T["stats"] / "clean.npz", allow_pickle=True)["names"]]
    audio = AudioStats(a.i7 / "audio_stats" / T["stats"], stems)
    test_idx = {f: np.flatnonzero(ref0["fold"] == f) for f in folds}
    noise = {}
    av = field_arrays(len(ref0["y"]))[1]
    for (s, f), iv in ivx.items():
        noise[(s, f)] = fit_lr(audio.rows("field", iv, "val"), (av[iv] != "clean").astype(int))
    s_rec = {(n, c): dict(zip(zip(g.seed, g.fold), g.f1)) for (n, c), g in S.groupby(["system", "condition"])}
    CAg = {kk: g.sort_values(["seed", "fold"]) for kk, g in
           CA.groupby(["cheap_src", "cheap_bb", "cheap", "expensive", "rule", "budget", "condition"])}
    sp_cache = {}

    def hull_for(c, cost, which):
        if (c, which) not in sp_cache:
            g = S[(S.condition == c) & S["set"].isin(SET_ORDER[which])]
            sp_cache[(c, which)] = g.groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
        sp = sp_cache[(c, which)]
        v_interp, lo, hi, lam = A.hull_pair(sp, cost)
        ok = sp[sp.cost <= cost * (1 + 1e-12)].sort_values(["f1", "cost"], ascending=[False, True])
        return v_interp, lo, hi, lam, (ok.iloc[0].system if len(ok) else None)

    def hull_f1(c, lo, hi, lam, kk):
        if lam == 0.0:
            return s_rec[(lo, c)][kk]
        return float(mix_expected_f1(ys[kk], preds[(lo, c, *kk)].astype(np.int64),
                                     preds[(hi, c, *kk)].astype(np.int64), lam, K)[0])

    ni_ref = f"ast:n4@{SL}+esc"

    def compare(design, c, which, baselines=False):
        x, bb, k, e, r, b = design
        g = CAg[(x, bb, k, e, r, b, c)]
        keys = list(zip(g.seed, g.fold))
        xv = g.f1.to_numpy()
        cost = float(g.cost.mean())
        v_interp, lo, hi, lam, best = hull_for(c, cost, which)
        hull = np.array([hull_f1(c, lo, hi, lam, kk) for kk in keys])
        row = {"hull_set": which, "cheap_src": x, "cheap_bb": bb, "cheap": k, "expensive": e, "rule": r, "budget": b,
               "condition": c, "f1": float(xv.mean()), "cost": cost, "rate": float(g.rate.mean()),
               "rate_min": float(g.rate.min()), "rate_max": float(g.rate.max()), "ece": float(g.ece.mean()),
               "hull_lo": lo, "hull_hi": hi, "lam": lam, "hull_f1_interp": v_interp, "hull_f1": float(hull.mean())}
        row.update({f"hull_{kk}": vv for kk, vv in A.tests(xv, hull, keys).items()})
        if not baselines:
            return row
        z = np.array([s_rec[(ni_ref, c)][kk] for kk in keys])
        row.update({"ni_ref": ni_ref, "ni_ref_f1": float(z.mean()), "ni_ref_cost": sys_cost(ni_ref)})
        row.update(ni_test(xv, z, keys, A.NB_RATIO["task"]))
        rnd, orc, orc_full, aud = [], [], [], []
        for kk in keys:
            pc = acs[(x, bb, k, c, *kk)].astype(np.int64)
            de = decs[(x, bb, k, e, r, c, *kk)].astype(np.int64)
            escm = escs[(x, bb, k, b, c, *kk)]
            yk = ys[kk]
            rnd.append(float(mix_expected_f1(yk, pc, de, escm.mean(), K)[0]))
            orc.append(A.f1_fast(yk, oracle_pred(yk, pc, de, int(escm.sum())), K))
            fix = (pc != yk) & (de == yk)
            orc_full.append(A.f1_fast(yk, np.where(fix, de, pc), K))
            if audio.available(c):
                sc = noise[kk](audio.rows(c, test_idx[kk[1]], "test"))
                aud.append(A.f1_fast(yk, np.where(top_k(sc, int(escm.sum())), de, pc), K))
        row["random_f1"] = float(np.mean(rnd))
        row.update({f"random_{kk}": vv for kk, vv in A.tests(xv, rnd, keys).items()})
        row["oracle_k_f1"], row["oracle_full_f1"] = float(np.mean(orc)), float(np.mean(orc_full))
        row["best_static"] = best
        zb = np.array([s_rec[(best, c)][kk] for kk in keys])
        row["best_static_f1"], row["best_static_cost"] = float(zb.mean()), sys_cost(best)
        row.update({f"static_{kk}": vv for kk, vv in A.tests(xv, zb, keys).items()})
        if aud:
            row["audio_f1"] = float(np.mean(aud))
            row.update({f"audio_{kk}": vv for kk, vv in A.tests(xv, aud, keys).items()})
        return row

    def sub_design(design, src):
        """The design with its cheap tier replaced by a substitute source (same head, tier, rule, budget)."""
        x, bb, k, e, r, b = design
        bb2 = "int8" if src.endswith(("int8a", "int8b")) else bb
        if k not in sources[src]["kinds"]:
            return None
        return (src, bb2, k, e, r, b)

    res = []
    if unp:
        p3_3c = tuple(json.loads((r3c / "analysis" / task / "picks.json").read_text())["P3"]["design"])
        p3_3c = (ren(p3_3c[0]), *p3_3c[1:])
        picks["P3_stage3c"] = {"design": list(p3_3c), "source": "Stage 3c picks.json P3 with the retrained exact-length tier"}
        (out / "picks.json").write_text(json.dumps(picks, indent=1))
        plan = [("P5c", "P2", p2, "stage3b", True), ("P5c", "P2", p2, "new", True), ("P5c", "P3", p3_3c, "new", True),
                ("P3", "P3", p3, "new", True)]
    else:
        plan = [("E4", "P1", p1, "stage3b", True), ("E4", "P2", p2, "stage3b", True), ("S4", "P2", p2, "new", True),
                ("P3", "P3", p3, "new", True)]
    for fam, base_name, d in (("P4b", "P2", p2), ("S5", "P3", p3)) if not unp else ():
        for suf, which in (("int8a", "new+int8a"), ("int8b", "new+int8b")):
            src = f"{d[0]}{suf}"
            if src in sources and sub_design(d, src) is not None:
                plan.append((fam, base_name, sub_design(d, src), which, True))
    if "mn10unp" in sources:
        for base_name, d in (("P2", p2), ("P3", p3)):
            if d[0] == "mn10" and sub_design(d, "mn10unp") is not None:
                plan.append(("P5b", base_name, sub_design(d, "mn10unp"), "new+unp", True))
    for fam, base_name, d, which, bl in plan:
        for c in CASC_CONDS:
            row = compare(d, c, which, baselines=bl)
            row["family"], row["base_design"] = fam, base_name
            res.append(row)
    R_ = pd.DataFrame(res)
    R_.to_csv(out / "selected.csv", index=False)
    # E4: P1 and P2 against the Stage 3b hull must equal Stage 3b's rows (not in the follow-up mode: the hull differs)
    repro = {}
    if not unp:
        s3b_sel = pd.read_csv(a.s3b / "analysis" / task / "selected.csv")
        s3b_sel = s3b_sel[s3b_sel.hull_set == "new"]
        for fam in ("P1", "P2"):
            for c in CASC_CONDS:
                mine = R_[(R_.family == "E4") & (R_.base_design == fam) & (R_.condition == c)].iloc[0]
                ref = s3b_sel[(s3b_sel.family == fam) & (s3b_sel.condition == c)].iloc[0]
                repro[f"{fam}_{c}"] = {k: float(abs(mine[k] - ref[k]) / (abs(ref[k]) if k == "cost" else 1.0))
                                       for k in ("f1", "cost", "hull_f1", "hull_diff", "hull_p_t_fold")}
        worst = max(v for d in repro.values() for v in d.values())
        print(f"{task}: E4 reproduction of Stage 3b, largest difference {worst:.2e} (absolute; cost relative)", flush=True)
        if worst > 1e-9:
            raise SystemExit(f"{task}: E4 does not reproduce Stage 3b: {repro}")

    e2 = [compare(d, "field", "new") for d in
          [(x, bb, k, e, r, b) for x, bb in p3_cheap for k in sources[x]["kinds"] for e in tiers for r in RULES for b in BUDGETS]]
    pd.DataFrame(e2).to_csv(out / "e2.csv", index=False)
    print(f"{task}: E2 done, {time.time() - t0:.0f}s", flush=True)

    # ── P4a / S6: real-int8 statics against the W8A8 static, non-inferiority ──
    ni_rows = []
    for src in [s for s in subs if s.endswith(("int8a", "int8b"))]:
        base = src[:-5]
        for k in sources[src]["kinds"]:
            for c in CASC_CONDS:
                keys = sorted(s_rec[(cheap_name(src, "int8", k), c)])
                xv = np.array([s_rec[(cheap_name(src, "int8", k), c)][kk] for kk in keys])
                zv = np.array([s_rec[(cheap_name(base, "w8a8", k), c)][kk] for kk in keys])
                fv = np.array([s_rec[(cheap_name(base, "fp32", k), c)][kk] for kk in keys])
                row = {"source": src, "base": base, "way": src[-1], "cheap": k, "condition": c, "f1_int8": float(xv.mean()),
                       "f1_w8a8": float(zv.mean()), "f1_fp32": float(fv.mean())}
                row.update(ni_test(xv, zv, keys, A.NB_RATIO["task"]))
                row.update({f"t_{kk}": vv for kk, vv in A.tests(xv, zv, keys).items()})
                ni_rows.append(row)
    pd.DataFrame(ni_rows).to_csv(out / "ni_int8.csv", index=False)

    # ── P5a: exact-length statics against padded ones ───────────────────────
    if "mn10unp" in sources:
        pad_rows = []
        for bb in ("fp32", "w8a8"):
            for k in sources["mn10unp"]["kinds"]:
                for c in conds:
                    if (cheap_name("mn10unp", bb, k), c) not in s_rec:
                        continue
                    keys = sorted(s_rec[(cheap_name("mn10unp", bb, k), c)])
                    xv = np.array([s_rec[(cheap_name("mn10unp", bb, k), c)][kk] for kk in keys])
                    zv = np.array([s_rec[(cheap_name("mn10", bb, k), c)][kk] for kk in keys])
                    row = {"cheap_bb": bb, "cheap": k, "condition": c, "f1_unpadded": float(xv.mean()),
                           "f1_padded": float(zv.mean()), "cost_unpadded": cheap_cost[("mn10unp", bb, k)],
                           "cost_padded": cheap_cost[("mn10", bb, k)]}
                    tt = A.tests(xv, zv, keys)
                    row.update(tt)
                    fm = np.array(json.loads(tt["fold_diffs"]))
                    row["ci95_lo"], row["ci95_hi"] = ci95(fm)
                    pad_rows.append(row)
        pd.DataFrame(pad_rows).to_csv(out / "padding.csv", index=False)

    # ── follow-up mode: retrained exact-length statics against the padded ones, every head kind ──
    if unp:
        pad_rows = []
        for bb in ("fp32", "w8a8"):
            for k in sources["mn10unpr"]["kinds"]:
                for c in conds:
                    if (cheap_name("mn10unpr", bb, k), c) not in s_rec or (cheap_name("mn10pad", bb, k), c) not in s_rec:
                        continue
                    keys = sorted(s_rec[(cheap_name("mn10unpr", bb, k), c)])
                    xv = np.array([s_rec[(cheap_name("mn10unpr", bb, k), c)][kk] for kk in keys])
                    zv = np.array([s_rec[(cheap_name("mn10pad", bb, k), c)][kk] for kk in keys])
                    row = {"cheap_bb": bb, "cheap": k, "condition": c, "f1_unpadded_retrained": float(xv.mean()),
                           "f1_padded": float(zv.mean()), "cost_unpadded": cheap_cost[("mn10unpr", bb, k)],
                           "cost_padded": cheap_cost[("mn10pad", bb, k)]}
                    tt = A.tests(xv, zv, keys)
                    row.update(tt)
                    fm = np.array(json.loads(tt["fold_diffs"]))
                    row["ci95_lo"], row["ci95_hi"] = ci95(fm)
                    pad_rows.append(row)
        pd.DataFrame(pad_rows).to_csv(out / "padding_retrained.csv", index=False)

    # ── S2: "cheap wrong, AST tier right" AUROC for every pick's cheap tier ──
    rows = []
    seen = set()
    for fam, base_name, d, which, _ in plan:
        if fam == "E4" or d in seen:
            continue
        seen.add(d)
        x, bb, k, e, r, b = d
        for c in conds:
            if c not in ast_conds[e] or c not in cheap_conds[(x, bb)]:
                continue
            for s in seeds:
                sc_, fx_ = [], []
                for f in folds:
                    te = test_idx[f]
                    p = A.softmax(cheap_get(x, bb, s, k, c)["z"][te])
                    pe = A.softmax(ast_get(e, s, "esc", c)["z"][te])
                    yk = ref0["y"][te]
                    sc_.append(1 - A.margin(p))
                    fx_.append((p.argmax(1) != yk) & (pe.argmax(1) == yk))
                sc_, fx_ = np.concatenate(sc_), np.concatenate(fx_)
                rows.append({"family": fam, "base_design": base_name, "cheap_src": x, "cheap_bb": bb, "cheap": k,
                             "expensive": e, "rule": r, "budget": b, "condition": c, "seed": s, "n_fixable": int(fx_.sum()),
                             "auroc_fixable": roc_auc_score(fx_, sc_) if 0 < fx_.sum() < len(fx_) else np.nan})
    pd.DataFrame(rows).to_csv(out / "auroc_fixable.csv", index=False)
    meta = {"task": task, "K": K, "seeds": seeds, "folds": folds, "nb_ratio": A.NB_RATIO["task"], "tiers": tiers,
            "sources": {x: {"kinds": v["kinds"], "set": v["set"], "configs": v["cfg"]} for x, v in sources.items()},
            "ast_conditions": ast_conds, "cheap_conditions": {f"{x}:{bb}": v for (x, bb), v in cheap_conds.items()},
            "picks": picks, "comparisons": [{"family": f_, "base": b_, "design": list(d_), "hull_set": w_}
                                            for f_, b_, d_, w_, _ in plan],
            "e4_reproduction": repro, "n_designs_p3": int(len(SO3)), "mean2": "m2" in groups, "seconds": time.time() - t0,
            "unp_retrained": str(a.unp_retrained) if unp else None, "s3c": str(r3c)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{task}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
