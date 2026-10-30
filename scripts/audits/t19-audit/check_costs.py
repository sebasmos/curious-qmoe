"""Check 3: costs from scratch against tables/costs.csv and the plan's section 4.2 table."""
import pandas as pd
from audit_core import R, TASK, NEW, LEG, cost, ast_bitops, ast_tokens, head_cost

rows = []
ref = pd.read_csv(R / "tables/costs.csv")
for task in ("esc50", "urban"):
    for c in NEW + ["L:" + l for l in LEG]:
        name = c if c.startswith("L:") else c + ":own"
        mine_p, mine_u = cost(task, name, True) / 1e12, cost(task, name, False) / 1e12
        r = ref[(ref.task == task) & (ref.config == c)].iloc[0]
        theirs_u = r.backbone_unpriced_T + r.fbank_T + r.head_T
        rows.append(dict(task=task, config=c, audit_priced_T=mine_p, t19_priced_T=r.system_priced_T, audit_unpriced_T=mine_u,
                         t19_unpriced_T=theirs_u, d_priced=mine_p - r.system_priced_T, d_unpriced=mine_u - theirs_u))
df = pd.DataFrame(rows)
df.to_csv("costs_check.csv", index=False)
print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
print("max abs diff priced", df.d_priced.abs().max(), "unpriced", df.d_unpriced.abs().max())
print("\nplan section 4.2 at 1024 frames (unpriced, no front end or head), tokens", ast_tokens(1024))
for nm, wb, ab, pb in (("W8A8 P16 stored rule", 8, 8, 16), ("W4A8 P16 stored rule", 4, 8, 16), ("W8A8 P8", 8, 8, 8),
                        ("N4A4 P8", 4, 4, 8), ("W4A6 P8", 4, 6, 8)):
    c = ast_bitops(1024, wb, ab, pb)
    print(f"{nm:22s} linear {c['linear']/1e12:.3f} qk {c['qk']/1e12:.3f} pv {c['pv']/1e12:.3f} patch {c['patch']/1e12:.3f} total {c['unpriced']/1e12:.3f}")
w8, n4, w46 = (ast_bitops(1024, *a)["unpriced"] for a in ((8, 8, 8), (4, 4, 8), (4, 6, 8)))
print(f"r N4A4/W8A8 = {n4/w8:.3f}, W4A6/W8A8 = {w46/w8:.3f}, 1/(1-r) = {1/(1-n4/w8):.3f}")
for task in ("esc50", "urban"):
    F = TASK[task]["frames"]
    a, b = ast_bitops(F, 4, 4, 8), ast_bitops(F, 8, 8, 8)
    print(task, F, "frames, tokens", ast_tokens(F), f"N4A4/W8A8 priced {a['priced']/b['priced']:.3f} unpriced {a['unpriced']/b['unpriced']:.3f}",
          f"patch conv {a['patch']/1e12:.3f} T, rescale n4a4 {a['rescale']/1e12:.3f} T, zero point {a['zp']/1e12:.3f} T")
