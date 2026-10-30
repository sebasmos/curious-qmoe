"""T17 selection (ANALYSIS_PLAN_T17.md sections 2 and 4): lambda of the diversity arms, (T, alpha) of the distilled arms and the best-single comparator, per bed, fold and condition family,
from INNER-VALIDATION logits and labels only (seeds pooled inside the fold). Reads no test label. python t17_select.py --run DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t17_common import *
import t13_common as T13
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); ap.add_argument("--beds", nargs="+", default=None); ap.add_argument("--maxfold", type=int, default=None); a = ap.parse_args()
sm = lambda z: (lambda e: e / e.sum(1, keepdims=True))(np.exp(z - z.max(1, keepdims=True)))
LAMS = [0.1, 0.3, 0.5]; GRID = [(T, al) for T in (1, 2, 4) for al in (0.25, 0.5, 0.75)]
SEL = {}
for bed in (a.beds or BEDS):
    D = load_bed(bed); y = D["y"]; task = TASK_OF_BED[bed]; C = costs17(bed); SEL[bed] = {}
    for fold in range(1, (a.maxfold or BEDS[bed]["nf"]) + 1):
        R = [np.load(a.run / bed / "raw" / f"s{s}_f{fold}.npz") for s in (0, 1, 2)]; SEL[bed][fold] = {}
        for cond, fam in FAMILY_OF.items():
            if (cond != "clean" and cond not in CONDITIONS[bed]) or (cond == "clean" and False): continue
            yfs = [y[r[f"famrows|{fam}"]] for r in R]
            def pair_cnt(name): return sum(int((np.mean([sm(r[f"iv|{name}|{m}|{fam}"].astype(np.float64)) for m in (0, 1)], 0).argmax(1) == yy).sum()) for r, yy in zip(R, yfs))
            def one_cnt(name): return sum(int((r[f"iv|{name}|0|{fam}"].argmax(1) == yy).sum()) for r, yy in zip(R, yfs))
            pick = lambda names, f: max(names, key=lambda n: (f(n), -names.index(n)))      # ties: first in grid order
            ncl = pick(["indep"] + [f"ncl{l}" for l in LAMS], pair_cnt); dec = pick(["indep"] + [f"dec{l}" for l in LAMS], pair_cnt)
            kA = pick([f"kdA_T{T}_a{al}" for T, al in GRID], one_cnt); kF = pick([f"kdF_T{T}_a{al}" for T, al in GRID], one_cnt)
            # best single at the cost of (a) and of (b), T13 stored experts, inner-validation logits of the same family
            best = {}
            for tag in ("a", "b"):
                elig = [e for e in T13.experts_of(task) if C["singles"][e] <= C[tag] * (1 + 1e-12)]; score = {}
                for e in elig:
                    cells = T13_CELLS[cond]; cnt = n = 0; ok = True
                    for cl in cells:
                        tcell = "clean" if bed == "tau" else cl
                        if not T13.available(task, e, tcell, [0, 1, 2]): ok = False; break
                        for s in (0, 1, 2):
                            L = T13.load(task, e, tcell, s); idx, z = L["iv"][fold]; cnt += int((z.argmax(1) == L["y"][idx]).sum()); n += len(idx)
                    if ok: score[e] = cnt / n
                best[tag] = dict(choice=max(score, key=score.get), scores=score)
            SEL[bed][fold][cond] = dict(ncl=ncl, dec=dec, kdA=kA, kdF=kF, best_single=best, n_family=int(sum(len(v) for v in yfs)),
                                        counts=dict(indep=pair_cnt("indep"), **{f"ncl{l}": pair_cnt(f"ncl{l}") for l in LAMS}, p10=one_cnt("p10"), p20=one_cnt("p20")))
    print(bed, "selected", flush=True)
(a.run / "inputs").mkdir(exist_ok=True); json.dump(SEL, open(a.run / "inputs/selection.json", "w"), indent=1)
