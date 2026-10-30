"""D2 subset, labels, folds, splits and noise source, rebuilt from the FSD50K metadata with my own code.
Reads dataset metadata as data only. Usage: python -I audit_leakage_d2.py <T21 results> <FSD50K meta_x> <audit out>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

R, META, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
log = open(OUT / "audit_leakage.log", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


gt = META / "FSD50K.ground_truth"
dev = pd.read_csv(gt / "dev.csv", dtype={"fname": str})
ev = pd.read_csv(gt / "eval.csv", dtype={"fname": str})
voc = pd.read_csv(gt / "vocabulary.csv", header=None, names=["i", "label", "mid"])
info_dev = json.loads((META / "FSD50K.metadata/dev_clips_info_FSD50K.json").read_text())
info_ev = json.loads((META / "FSD50K.metadata/eval_clips_info_FSD50K.json").read_text())
if not (OUT / "ontology_download/ontology.json").exists(): sys.exit("fetch https://raw.githubusercontent.com/audioset/ontology/master/ontology.json into ontology_download/ (sha256 9c685f4403eecc3ca9be37fd7285cf212feaaea6ff7229d3e7ca89e0d1f2d15d)")
onto = {n["id"]: n for n in json.loads((OUT / "ontology_download/ontology.json").read_text())}
mid2lab = dict(zip(voc.mid, voc.label))
lab2mid = dict(zip(voc.label, voc.mid))
V = set(voc.mid)


# true ancestors inside the FSD50K vocabulary, from the AudioSet ontology
def descendants(m, seen=None):
    seen = set() if seen is None else seen
    for c in onto[m]["child_ids"]:
        if c not in seen:
            seen.add(c)
            descendants(c, seen)
    return seen


desc = {m: descendants(m) & V for m in V}
anc = {m: {a for a in V if m in desc[a]} for m in V}
leaves = {m for m in V if not desc[m]}
P("ontology: vocabulary", len(V), "leaves in the FSD50K vocabulary", len(leaves), "(README says 144)")
# their data-derived ancestry
for d in (dev, ev):
    d["mids_set"] = d.mids.str.split(",").map(frozenset)
clips_of = {m: set(np.flatnonzero(dev.mids_set.map(lambda s: m in s))) for m in V}
anc_data = {a: {b for b in V if b != a and clips_of[a] <= clips_of[b]} for a in V}
leaves_data = {m for m in V if m not in set().union(*anc_data.values())}
P("data-derived leaves", len(leaves_data), "; in data but not ontology:", sorted(mid2lab[m] for m in leaves_data - leaves),
  "; in ontology but not data:", sorted(mid2lab[m] for m in leaves - leaves_data))
bad_anc = {mid2lab[a]: sorted(mid2lab[b] for b in anc_data[a] - anc[a]) for a in V if anc_data[a] - anc[a]}
P("data-derived 'ancestors' that are not ontology ancestors (class: wrongly assumed ancestors):", bad_anc)
miss_anc = {mid2lab[a]: sorted(mid2lab[b] for b in anc[a] - anc_data[a]) for a in V if anc[a] - anc_data[a]}
P("ontology ancestors not found by the data rule:", miss_anc)
# smearing check: every clip's labels closed under ontology ancestors
unsmeared = dev.mids_set.map(lambda s: any(not anc[m] <= s for m in s)).sum()
P("dev clips whose label set is not closed under ontology ancestors:", int(unsmeared))


def leaf_of(s, LV, AN):
    lv = [m for m in s if m in LV]
    if len(lv) != 1:
        return None
    return lv[0] if all(m in AN[lv[0]] for m in s if m != lv[0]) else None


for d in (dev, ev):
    d["leaf"] = d.mids_set.map(lambda s: leaf_of(s, leaves, anc))
cnt = dev.leaf.value_counts()
cls = sorted(mid2lab[m] for m in cnt[cnt >= 60].index)
P("ontology rule: qualifying dev clips", int(dev.leaf.notna().sum()), "classes with >= 60:", len(cls))
mine_pool = dev[dev.leaf.map(lambda m: m is not None and mid2lab[m] in cls)].copy()
mine_pool["label"] = mine_pool.leaf.map(mid2lab)
mine_extra = ev[ev.leaf.map(lambda m: m is not None and mid2lab[m] in cls)].copy()
mine_extra["label"] = mine_extra.leaf.map(mid2lab)
pool = pd.read_csv(R / "csv/d2/pool.csv", dtype={"uploader": str})
extra = pd.read_csv(R / "csv/d2/extra.csv", dtype={"uploader": str})
pool["fname"] = pool.name.str.replace(".wav", "", regex=False)
extra["fname"] = extra.name.str.replace(".wav", "", regex=False)
theirs_cls = sorted(json.loads((R / "csv/d2/classes.json").read_text())["classes"])
P("classes equal (ontology rule vs theirs):", cls == theirs_cls, "| only mine:", sorted(set(cls) - set(theirs_cls)), "| only theirs:", sorted(set(theirs_cls) - set(cls)))
mp = dict(zip(mine_pool.fname, mine_pool.label))
tp = dict(zip(pool.fname, pool.category))
P("pool clips: mine", len(mp), "theirs", len(tp), "only mine", len(set(mp) - set(tp)), "only theirs", len(set(tp) - set(mp)),
  "label mismatches on common", sum(mp[k] != tp[k] for k in set(mp) & set(tp)))
me = dict(zip(mine_extra.fname, mine_extra.label))
te_ = dict(zip(extra.fname, extra.category))
P("eval clips: mine", len(me), "theirs", len(te_), "only mine", len(set(me) - set(te_)), "only theirs", len(set(te_) - set(me)),
  "label mismatches", sum(me[k] != te_[k] for k in set(me) & set(te_)))
# class id mapping consistent (sorted names) and label == class_id
cid = {c: i for i, c in enumerate(theirs_cls)}
P("class_id equals sorted-name index:", bool((pool.category.map(cid) == pool.class_id).all() and (pool.label == pool.class_id).all()))
# clips the single-label rule admits whose smeared label set holds a sibling (would be multi-label in truth)
ccount = pool.category.value_counts()
P("their extra classes in the pool:", {c: int(ccount.get(c, 0)) for c in ("Fire", "Wood")}, "| Crackle, Crack in their classes:", "Crackle" in theirs_cls, "Crack" in theirs_cls)
P("their pool clips the ontology rule does not admit (no ontology leaf among their labels):", int(pool.fname.map(lambda f: f not in mp).sum()))

# folds
pool = pool.sort_values("name").reset_index(drop=True) if not pool.name.is_monotonic_increasing else pool
upl = pool.fname.map(lambda f: info_dev[f]["uploader"])
P("uploader column equals dev_clips_info:", bool((upl == pool.uploader).all()))
sgk = StratifiedGroupKFold(5, shuffle=True, random_state=42)
fold = np.zeros(len(pool), int)
for k, (_, tei) in enumerate(sgk.split(np.zeros(len(pool)), pool.class_id, upl), 1):
    fold[tei] = k
P("folds rebuilt with StratifiedGroupKFold(5, shuffle, 42) grouped by uploader equal pool.csv:", bool((fold == pool.folder).all()))
P("uploaders spanning more than one fold:", int(pool.groupby("uploader").folder.nunique().gt(1).sum()), "of", pool.uploader.nunique())
P("fold sizes", pool.folder.value_counts().sort_index().tolist())
ct = pd.crosstab(pool.category, pool.folder)
P("classes absent per test fold:", {int(f): sorted(ct.index[ct[f] == 0]) for f in ct.columns if (ct[f] == 0).any()})
P("min training clips of a class in any fold (pool minus that fold):", int((ct.sum(1).values[:, None] - ct.values).min()))
P("largest share of one class in one fold:", float((ct.values / ct.sum(1).values[:, None]).max()))
P("Freesound ids unique in pool:", pool.fname.is_unique, "| pool ids also in eval:", len(set(pool.fname) & set(extra.fname)))
P("uploaders shared between pool (dev) and eval (secondary design):", len(set(pool.uploader) & set(extra.uploader)))

# head CSVs and split files
for tier in ("mn10", "ast"):
    hn = pd.read_csv(R / f"csv/d2/heads/pool_{tier}.csv", usecols=["folder", "name", "class_id"])
    ok = (hn.name == pool.head_name).all() and (hn.folder == pool.folder).all() and (hn.class_id == pool.class_id).all()
    u_from_name = hn.name.str.split("-").str[0]
    same_group = pd.crosstab(u_from_name, pool.uploader).gt(0).sum(1).max() == 1 and pd.crosstab(pool.uploader, u_from_name).gt(0).sum(1).max() == 1
    P(f"head CSV {tier}: rows and order equal pool.csv: {bool(ok)}; name token 0 is a one-to-one uploader code: {bool(same_group)}")
n = len(pool)
bad = 0
for tier in ("mn10", "ast"):
    for s in (0, 1, 2):
        for f in range(1, 6):
            sp = json.loads((R / f"outputs/d2_pool_{tier}_s{s}/esc/fold_{f}/split.json").read_text())
            te, iv = np.asarray(sp["test_idx"]), np.asarray(sp["inner_val_idx"])
            fit = np.setdiff1d(np.arange(n), np.concatenate([te, iv]))
            for key in ("train_idx", "fit_idx", "inner_fit_idx"):
                if key in sp:
                    if not np.array_equal(np.sort(np.asarray(sp[key])), fit if key != "train_idx" else np.setdiff1d(np.arange(n), te)):
                        bad += 1
                        P("split key differs", tier, s, f, key)
            u_te, u_iv, u_fit = set(pool.uploader[te]), set(pool.uploader[iv]), set(pool.uploader[fit])
            if not np.array_equal(np.sort(te), np.flatnonzero(pool.folder == f)):
                bad += 1
                P("test rows differ from fold", tier, s, f)
            if u_te & u_iv or u_te & u_fit or u_iv & u_fit:
                bad += 1
                P("uploader overlap", tier, s, f, len(u_te & u_iv), len(u_te & u_fit), len(u_iv & u_fit))
P("split.json keys:", sorted(sp.keys()))
P("split files with a problem (test rows, uploader overlap between test, inner validation and fit):", bad, "of 30")
P("inner-validation sizes seed 0 (mn10):", [len(json.loads((R / f'outputs/d2_pool_mn10_s0/esc/fold_{f}/split.json').read_text())['inner_val_idx']) for f in range(1, 6)])

# noise source
us = pd.read_csv(Path.home() / "Documents/DATASETS/Urban8k/efficientnet/urbansound8k.csv", usecols=["name", "category"])
us_names = set(us.name.astype(str).str.rsplit(".", n=1).str[0])
P("UrbanSound8K clips in its metadata:", len(us_names))
for tier in ("mn_pool", "ast_pool"):
    bp = json.loads((R / f"emb/d2/{tier}/bg_plan.json").read_text())
    names = {v["bg_clip"] for v in bp.values()}
    P(f"{tier} bg_plan: {len(bp)} clips, keys equal pool names: {set(bp) == set(pool.fname)}, bg clips {len(names)}, all in UrbanSound8K: {names <= us_names}")
bp_m = json.loads((R / "emb/d2/mn_pool/bg_plan.json").read_text())
bp_a = json.loads((R / "emb/d2/ast_pool/bg_plan.json").read_text())
P("bg_plan identical content between tiers:", bp_m == bp_a)
us_fsid = {n_.split("-")[0] for n_ in names}
P("UrbanSound8K source Freesound ids used as background:", len(us_fsid), "| equal to an FSD50K pool clip id:", sorted(us_fsid & set(pool.fname)),
  "| equal to an FSD50K eval clip id:", sorted(us_fsid & set(extra.fname)))
log.close()
