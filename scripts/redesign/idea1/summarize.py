"""Cross-dataset summary of idea 1 stage 1 (reads analyze.py's <name>_idea1.json files).

For every cascade configuration (L, H, signal, rule): its paired difference to the
best single model at equal or lower bit-ops on each dataset, and whether it is
above the front on every dataset. Also the per-dataset best configuration replayed
on the other datasets (a guard against picking the luckiest of many points).

usage: python summarize.py OUT.txt name=path/to/<name>_idea1.json [...]
"""
import json, sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
D = {kv.split("=", 1)[0]: json.loads(Path(kv.split("=", 1)[1]).read_text()) for kv in sys.argv[2:]}
names = list(D)
key = lambda o: (o["L"], o["H"], o["signal"], o["rule"])
ops = {n: {key(o): o for o in d["ops"]} for n, d in D.items()}
lines = ["# idea 1 stage 1, cross-dataset summary", ""]
for n, d in D.items():
    above = [o for o in d["ops"] if o["diff"] is not None and o["diff"] > 0]
    sig = [o for o in above if o["p"] is not None and o["p"] < 0.05]
    lines.append(f"{n}: {len(d['units'])} units, front {d['front']}, {len(above)}/{len(d['ops'])} points above front, "
                 f"{len(sig)} with Wilcoxon p < 0.05 (uncorrected)")
lines.append("")
lines.append("configs above the front on every dataset (top 20 by worst-dataset diff): L>H signal rule | per dataset diff (wins/n, p, bit-ops M, test esc%)")
allk = set.intersection(*(set(o) for o in ops.values()))
good = [k for k in allk if all(ops[n][k]["diff"] is not None and ops[n][k]["diff"] > 0 for n in names)]
good.sort(key=lambda k: -min(ops[n][k]["diff"] for n in names))
lines.append(f"{len(good)} of {len(allk)} configs")
for k in good[:20]:
    lines.append(f"{k[0]}>{k[1]} {k[2]} {k[3]} | " + " ; ".join(
        f"{n} {ops[n][k]['diff']:+.4f} ({ops[n][k]['wins']}/{len(D[n]['units'])}, p {ops[n][k]['p']:.3f}, "
        f"{ops[n][k]['bitops'] / 1e6:.1f}M, {100 * ops[n][k]['esc_frac_test']:.0f}%)" for n in names))
if not good:
    lines.append("none")
lines.append("")
lines.append("best config on each dataset, replayed on the others (diff vs front):")
for n in names:
    b = max((o for o in D[n]["ops"] if o["diff"] is not None), key=lambda o: o["diff"])
    k = key(b)
    lines.append(f"best on {n}: {k[0]}>{k[1]} {k[2]} {k[3]} | " + " ; ".join(
        f"{m} {ops[m][k]['diff']:+.4f} ({ops[m][k]['wins']}/{len(D[m]['units'])})" for m in names))
lines.append("")
lines.append("per signal, mean over every (L, H, rule) of diff vs front; AUROC (i) L wrong and (ii) L wrong with"
             " H right (H = 8 | esc), averaged over L:")
for g in ["1-maxp", "margin", "entropy", "T-entropy", "mc4_var", "mc10_var"]:
    lines.append(f"{g:10s} " + " ; ".join(
        f"{n} diff {np.mean([o['diff'] for o in D[n]['ops'] if o['signal'] == g]):+.4f} "
        f"AUROC(i) {np.mean([v['L_wrong'] for kk, v in D[n]['auroc'].items() if kk.endswith('|' + g)]):.3f} "
        f"AUROC(ii) {np.mean([v['fix_8'] for kk, v in D[n]['auroc'].items() if kk.endswith('|' + g)]):.3f}"
        f" | {np.mean([v['fix_esc'] for kk, v in D[n]['auroc'].items() if kk.endswith('|' + g)]):.3f}"
        for n in names))
txt = "\n".join(lines)
print(txt)
out.write_text(txt + "\n")
