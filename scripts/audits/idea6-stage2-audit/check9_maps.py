"""Check 9: class maps of E6 / E7 and of the two ESC-50 five-class label sets against the official metadata;
which clips the E6 and E7 heads were trained on. Read only. Writes outputs/check9_maps.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, R  # noqa: E402

E = Path.home() / "Documents/DATASETS"
esc_meta = pd.read_csv(E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/meta/esc50.csv")
us_meta = pd.read_csv(E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/metadata/UrbanSound8K.csv")
esc_csv = pd.read_csv(E / "ESC-50/ast_768/esc-50.csv", usecols=["folder", "name", "label", "category", "class_id"])
us_csv = pd.read_csv(E / "Urban8k/ast_768/urbansound8k.csv", usecols=["folder", "name", "label", "category", "class_id"])
out = {}
t2c = dict(zip(esc_meta.target, esc_meta.category))
c2c = dict(zip(us_meta.classID, us_meta["class"]))
out["esc_shared"] = {t: t2c[t] for t in (0, 42, 43, 44)}
out["us_shared"] = {c: c2c[c] for c in (3, 8, 1, 5)}
out["us_other_classes"] = {c: c2c[c] for c in sorted(c2c) if c not in (3, 8, 1, 5)}
# the CSV's label column against the metadata (by file name)
e = esc_csv.assign(stem=esc_csv.name.str.replace(".png", "", regex=False))
em = esc_meta.assign(stem=esc_meta.filename.str.replace(".wav", "", regex=False)).set_index("stem")
out["esc_label_equals_target"] = bool((e.label.to_numpy() == em.loc[e.stem, "target"].to_numpy()).all())
out["esc_folder_equals_fold"] = bool((e.folder.to_numpy() == em.loc[e.stem, "fold"].to_numpy()).all())
u = us_csv.assign(stem=us_csv.name.str.replace(".png", "", regex=False).str.replace(".wav", "", regex=False))
um = us_meta.assign(stem=us_meta.slice_file_name.str.replace(".wav", "", regex=False)).set_index("stem")
out["us_label_equals_classID"] = bool((u.label.to_numpy() == um.loc[u.stem, "classID"].to_numpy()).all())
out["us_folder_equals_fold"] = bool((u.folder.to_numpy() == um.loc[u.stem, "fold"].to_numpy()).all())
# ESC-50 five-class sets
official = esc_csv.label // 10
ct = pd.crosstab(esc_csv.label, esc_csv.class_id)
grp = {int(t): int(ct.loc[t].idxmax()) for t in ct.index}
out["esc5_repo_groups"] = {g: sorted(t for t, gg in grp.items() if gg == g) for g in sorted(set(grp.values()))}
out["esc5_repo_vs_official_classes_agree"] = int(sum(grp[t] == t // 10 for t in grp))
# official categories in the metadata order: targets 0-9 animals etc.; check with the metadata category names
out["official_groups_by_name"] = {g: [t2c[t] for t in range(10 * g, 10 * g + 10)] for g in range(5)}
off = pd.read_csv(R / "csv/esc-50_effb3pool_fp32_off5.csv", usecols=["label", "class_id"])
out["off5_class_id_equals_target_div_10"] = bool((off.class_id == off.label // 10).all())
# E6 training CSVs: only source-dataset shared-class clips
for tier in ("eff", "ast"):
    for ds in ("esc50", "urban8k"):
        d = pd.read_csv(R / f"csv/realshift/{tier}_{ds}_shared.csv", usecols=["folder", "name", "label", "class_id"])
        out[f"e6_{tier}_{ds}"] = dict(n=len(d), per_class=d.label.value_counts().sort_index().tolist(),
                                      names_in_other_dataset=int(d.name.isin(us_csv.name if ds == "esc50" else esc_csv.name).sum()))
# E6 heads: test indices of each fold equal the official fold rows, inner validation disjoint from test
chk = []
for ds in ("esc50", "urban8k"):
    d = pd.read_csv(R / f"csv/realshift/eff_{ds}_shared.csv", usecols=["folder"])
    for tier in ("eff", "ast"):
        for s in (0, 1, 2):
            run = R / f"outputs/xds_{tier}_{ds}_s{s}/esc"
            for fdir in sorted(run.glob("fold_*")):
                f = int(fdir.name.split("_")[1])
                sp = json.loads((fdir / "split.json").read_text())
                te, iv = np.array(sp["test_idx"]), np.array(sp["inner_val_idx"])
                chk.append(bool(np.array_equal(np.sort(te), np.flatnonzero(d.folder.to_numpy() == f))
                                and len(np.intersect1d(te, iv)) == 0))
out["e6_splits_ok"] = f"{sum(chk)}/{len(chk)}"
# E7: OOD sets
out["e7_ood_for_esc_tasks_us_classes"] = sorted(us_csv.loc[~us_csv.label.isin([3, 8, 1, 5]), "category"].unique().tolist())
out["e7_n_ood_for_esc_tasks"] = int((~us_csv.label.isin([3, 8, 1, 5])).sum())
out["e7_n_ood_for_urban"] = int((~esc_csv.label.isin([0, 42, 43, 44])).sum())
(OUT / "check9_maps.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
