"""Check 3b: the cascade is charged the full w8a8 system (front end, FP32 patch conv, backbone, head) on escalated clips. The front end
and the FP32 patch embedding are identical in both tiers and could be reused. Credit them back and recompute the picked field-mix point."""
import numpy as np
from audit_core import TASK, cost, ast_bitops, nb
from reproduce_h1_h2 import cascade, fm, hull_folds, members, ratio

for task, b, rule in (("esc50", 0.10, "B"), ("urban", 0.30, "avg")):
    t = TASK[task]
    reuse = t["fbank"] + ast_bitops(t["frames"], 8, 8, 8)["patch"]
    df = cascade(task, "field", b, rule)
    fc = fm(df, "f1")
    for label, c in (("as run", cost(task, "n4a4:own") + df.rate.mean() * cost(task, "w8a8:own")),
                     ("front end and patch conv reused", cost(task, "n4a4:own") + df.rate.mean() * (cost(task, "w8a8:own") - reuse))):
        h, desc = hull_folds(task, "field", c, members(task, "B3"))
        r = nb(fc - h, ratio(task))
        print(f"{task} {label:32s} cost {c/1e12:.4f} T hull {h.mean():.3f} [{desc}] diff {r['mean']:+.3f} p {r['p']:.3f}")
