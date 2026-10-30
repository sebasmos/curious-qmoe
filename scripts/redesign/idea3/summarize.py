"""Cross-dataset summary of idea 3 stage 1 (reads analyze.py's <name>_idea3.json files).

Go rule checks:
  A  a disagreement or SR signal beats softmax margin at finding the errors an
     expensive model fixes (AUROC (ii), H = 8), and a cascade it drives is above
     the single-model front;
  B  an SR ensemble's answer beats the best single model at equal or lower bit-ops
     (and pre-sampled storage).

usage: python summarize.py OUT.txt name=path/to/<name>_idea3.json [...]
"""
import json, sys
from pathlib import Path

D = {kv.split("=", 1)[0]: json.loads(Path(kv.split("=", 1)[1]).read_text()) for kv in sys.argv[2:]}
out = Path(sys.argv[1])
names = list(D)
DIS = ["disagree", "symKL", "vote", "var", "MI"]
lines = ["# idea 3 stage 1, cross-dataset summary", ""]

lines.append("## B: answers vs best single at <= bit-ops: system | per dataset F1, diff (wins/n, p) [diff with storage cap]")
systems = [s["name"] for s in D[names[0]]["systems"]]
for sname in systems:
    parts = []
    for n in names:
        s = next(x for x in D[n]["systems"] if x["name"] == sname)
        f1 = sum(s["f1"]) / len(s["f1"])
        parts.append(f"{n} {f1:.4f} vs {s['best_single']} {s['diff']:+.4f} ({s['wins']}/{len(s['f1'])}, p {s['p']:.3f})"
                     f" [{s['best_single_storage']} {s['diff_storage']:+.4f}]")
    lines.append(f"{sname:15s} | " + " ; ".join(parts))

for H in ("8", "esc"):
    lines.append(f"\n## A: AUROC (ii) answer wrong and {H} right: best disagreement-type signal vs the same answer's"
                 " margin vs the reference single head's margin on its own errors")
    for sname in systems:
        parts = []
        for n in names:
            a = D[n]["auroc"][sname]
            own = {"pair": "margin_avg", "probe": "margin_A", "SR": "margin_ens"}[sname.split()[0]]
            best = max((g for g in a if g in DIS), key=lambda g: a[g][H])
            parts.append(f"{n} {best} {a[best][H]:.3f} vs {own} {a[own][H]:.3f} vs single "
                         f"{a['ref_single_margin_own_errors'][H]:.3f}")
        lines.append(f"{sname:15s} | " + " ; ".join(parts))

lines.append("\n## A: best cascade driven by a disagreement-type signal, per dataset (diff vs front)")
for n in names:
    cand = [o for o in D[n]["ops"] if o["signal"] in DIS]
    b = max(cand, key=lambda o: o["diff"])
    above = [o for o in cand if o["diff"] > 0]
    lines.append(f"{n}: {len(above)}/{len(cand)} above front; best {b['system']} > {b['H']} {b['signal']} {b['rule']} "
                 f"F1 {b['f1']:.4f} {b['bitops'] / 1e6:.1f}M vs {b['best_single']} {b['best_single_f1']:.4f} "
                 f"diff {b['diff']:+.4f} ({b['wins']}/{len(D[n]['units'])}, p {b['p']:.3f}), test esc {100 * b['esc_frac_test']:.0f}%")
key = lambda o: (o["system"], o["H"], o["signal"], o["rule"])
ops = {n: {key(o): o for o in D[n]["ops"]} for n in names}
allk = set.intersection(*(set(o) for o in ops.values()))
good = sorted([k for k in allk if all(ops[n][k]["diff"] > 0 for n in names)],
              key=lambda k: -min(ops[n][k]["diff"] for n in names))
lines.append(f"\nconfigs above the front on every dataset (any signal): {len(good)} of {len(allk)}; top 20 by worst-dataset diff")
for k in good[:20]:
    lines.append(f"{k[0]} > {k[1]} {k[2]} {k[3]} | " + " ; ".join(
        f"{n} {ops[n][k]['diff']:+.4f} ({ops[n][k]['wins']}/{len(D[n]['units'])}, {ops[n][k]['bitops'] / 1e6:.1f}M)"
        for n in names))
txt = "\n".join(lines)
print(txt)
out.write_text(txt + "\n")
