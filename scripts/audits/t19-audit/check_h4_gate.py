"""Check 6 (H4) and part of check 7 (gate direction, wrong-tail control, gate AUROC on test rows)."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from audit_core import R, TASK, load, get, softmax, margin, nb
from reproduce_h1_h2 import cascade, fm, folds, ratio

res = {}
for task in ("esc50", "urban"):
    K = TASK[task]["K"]
    per_fold = []
    sanity = []
    for f in folds(task):
        am, aj, aj_det, am_iv = [], [], [], []
        for s in TASK[task]["seeds"]:
            Dc, De = get(task, "n4a4:own", s, "field"), get(task, "w8a8:own", s, "field")
            Q1 = load(R / "scores" / task / "n4a4sr1@n4a4" / f"s{s}", "n4a4sr1", "field")
            Q2 = load(R / "scores" / task / "n4a4sr2@n4a4" / f"s{s}", "n4a4sr2", "field")
            r = np.flatnonzero(Dc["fold"] == f)
            y = Dc["y"][r]
            pc, pe = softmax(Dc["z"][r]), softmax(De["z"][r])
            tgt = (pc.argmax(1) != y) & (pe.argmax(1) == y)
            p1, p2 = softmax(Q1["z"][r]), softmax(Q2["z"][r])
            def js(a, b):
                m = (a + b) / 2
                kl = lambda u, v: np.sum(u * (np.log(u + 1e-12) - np.log(v + 1e-12)), 1)
                return 0.5 * (kl(a, m) + kl(b, m))
            J = js(p1, p2)
            am.append(roc_auc_score(tgt, -margin(pc)))
            aj.append(roc_auc_score(tgt, J))
            aj_det.append(roc_auc_score(tgt, js(pc, p1)))      # descriptive: deterministic vs one SR pass
            sanity.append(dict(seed=s, fold=f, sr_flip=float((p1.argmax(1) != p2.argmax(1)).mean()),
                               det_vs_sr1_flip=float((pc.argmax(1) != p1.argmax(1)).mean()), mean_js=float(J.mean()),
                               n_target=int(tgt.sum()), n=len(y)))
        per_fold.append(dict(fold=f, auroc_margin=np.mean(am), auroc_js=np.mean(aj), auroc_js_det_sr1=np.mean(aj_det)))
    P = pd.DataFrame(per_fold)
    d = P.auroc_js - P.auroc_margin
    maj = len(d) // 2 + 1
    res[task] = dict(auroc_margin=P.auroc_margin.mean(), auroc_js=P.auroc_js.mean(), diff=nb(d, ratio(task)), folds_pos=int((d > 0).sum()),
                     n_folds=len(d), passes=bool(d.mean() >= 0.02 and (d > 0).sum() >= maj),
                     auroc_js_det_vs_sr1_descriptive=P.auroc_js_det_sr1.mean(),
                     sr_sanity=pd.DataFrame(sanity)[["sr_flip", "det_vs_sr1_flip", "mean_js"]].mean().to_dict())
    # gate direction: picked budget/rule, low-margin tail (as run) vs high-margin tail (wrong-tail control)
    b, rule = (0.10, "B") if task == "esc50" else (0.30, "avg")
    lo = cascade(task, "field", b, rule)
    hi = cascade(task, "field", b, rule, tail="high")
    res[task]["gate_direction"] = dict(budget=b, rule=rule, low_tail_f1=fm(lo, "f1").mean(), low_tail_rate=lo.rate.mean(),
                                       high_tail_f1=fm(hi, "f1").mean(), high_tail_rate=hi.rate.mean())
    # escalated-clip fix rates on the field mix at the pick
    fix, harm, nesc, ntot = 0, 0, 0, 0
    for s in TASK[task]["seeds"]:
        Cc = get(task, "n4a4:own", s, "clean")
        Dc, De = get(task, "n4a4:own", s, "field"), get(task, "w8a8:own", s, "field")
        from audit_core import thr_closest
        for f in folds(task):
            thr = thr_closest(margin(softmax(Cc["iv"][f][1])), b)
            r = np.flatnonzero(Dc["fold"] == f)
            y = Dc["y"][r]
            pc, pe = softmax(Dc["z"][r]), softmax(De["z"][r])
            fused = pe if rule == "B" else (pc + pe) / 2
            e = margin(pc) < thr
            fix += int(((pc.argmax(1) != y) & (fused.argmax(1) == y) & e).sum())
            harm += int(((pc.argmax(1) == y) & (fused.argmax(1) != y) & e).sum())
            nesc += int(e.sum()); ntot += len(y)
    res[task]["escalated"] = dict(n_escalated=nesc, n=ntot, fixed=fix, harmed=harm, net_fix_rate_pct=100 * (fix - harm) / nesc)
print(json.dumps(res, indent=1, default=float))
json.dump(res, open("h4_gate.json", "w"), indent=1, default=float)
