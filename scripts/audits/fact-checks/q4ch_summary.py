import pandas as pd, numpy as np
d = pd.read_csv("q4ch_head_transfer_raw.csv")
g = d.groupby(["cfg","cond","variant","seed"]).f1.mean().reset_index()   # mean over folds per seed
m = g.groupby(["cfg","cond","variant"]).f1.mean().unstack("variant")      # mean over seeds
fp = m.loc["fp32"]["T"]
pd.set_option("display.width", 200)
out = []
for cond in ["clean","white__0db","white__5db","white__10db","bg__0db","pink__0db"]:
    row = {"cond": cond, "FP32": fp[cond]}
    for cfg in ["w4g128","w4","w3g64"]:
        for v in ["T","Rfull","R128","O"]:
            if (cfg,cond) in m.index and v in m.columns and not np.isnan(m.loc[(cfg,cond)][v]):
                row[f"{cfg}:{v}"] = m.loc[(cfg,cond)][v]
    out.append(row)
t = pd.DataFrame(out).set_index("cond")
print(t.round(4).T.to_string())
print("\nLoss against FP32 in points, same table")
print(((t.drop(columns="FP32").sub(t["FP32"], axis=0))*100).round(1).T.to_string())
# seed-level range for w4 white 0 dB
x = g[(g.cfg=="w4")&(g.cond=="white__0db")].pivot(index="seed", columns="variant", values="f1")
print("\nw4 white 0 dB per seed (mean over folds)"); print(x.round(4).to_string())
print("fp32 per seed:", g[(g.cfg=="fp32")&(g.cond=="white__0db")&(g.variant=="T")].f1.round(4).tolist())
# paired fold-level: own vs transferred (w4, white 0), seeds averaged inside folds
from scipy import stats
dd = d[(d.cfg=="w4")&(d.cond=="white__0db")].groupby(["variant","fold"]).f1.mean().unstack("variant")
f = d[(d.cfg=="fp32")&(d.cond=="white__0db")&(d.variant=="T")].groupby("fold").f1.mean()
for v in ["T","Rfull","R128","O"]:
    diff = (dd[v]-f)*100
    print(v, "gap to FP32 by fold (pts):", diff.round(1).tolist(), "mean", round(diff.mean(),2))
for v in ["Rfull","R128","O"]:
    diff=(dd[v]-dd["T"])*100; print(v,"minus T by fold:",diff.round(2).tolist(),"mean",round(diff.mean(),2),"t p",round(stats.ttest_1samp(diff,0).pvalue,3))
