"""Check 2: row alignment between the EfficientNet tier and the AST tier, by file name.

Reads (read only): the two embedding CSVs per dataset (meta columns only), the official dataset
metadata, every split.json of every run that feeds idea 6 (EfficientNet heads: idea 6 outputs and
the protocol 3 singles; AST heads: idea 5 Stage 2 outputs), and every score npz used by the
analysis. Writes one JSON with every comparison and its outcome.

    python check2_alignment.py OUT.json
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

H = Path.home() / "Documents"
import os; RD = Path(os.environ.get("RESULTS_DIR", str(H / "results")))  # results root (archive edit: path only)
DS = H / "DATASETS"
I6 = RD / "2026-10-22_0953_redesign-idea6-stage1_code-ec70260"
S2 = RD / "2026-10-20_2359_redesign-idea5-stage2_code-3c18c38"
P3 = RD / "2026-10-20_2127_protocol-v3-singles_code-bc4ff1c/outputs"
META = ["folder", "name", "label", "category", "class_id"]

CSV = {"esc50": (DS / "ESC-50/efficientnet_1536/esc-50.csv", DS / "ESC-50/ast_768/esc-50.csv"),
       "urban": (DS / "Urban8k/efficientnet/urbansound8k.csv", DS / "Urban8k/ast_768/urbansound8k.csv")}
TASKS = {  # task: (dataset, label column, eff run dir pattern, ast run dir pattern, seeds, eff score tag, ast score tag)
    "esc50": ("esc50", "label", I6 / "outputs/effesc50_s{s}/{k}", S2 / "outputs/astesc50_s{s}/{h}", range(5),
              I6 / "scores/esc50/effesc50_s{s}/{k}.npz", S2 / "scores/esc50/astesc50_s{s}/{h}/{c}__clean.npz"),
    "esc5": ("esc50", "class_id", P3 / "esc_singles_s{s}/{k}", S2 / "outputs/astesc5_s{s}/{h}", range(5),
             I6 / "scores/esc5/esc_singles_s{s}/{k}.npz", S2 / "scores/esc5/astesc5_s{s}/{h}/{c}__clean.npz"),
    "urban": ("urban", "class_id", P3 / "urban_singles_s{s}/{k}", S2 / "outputs/asturban_s{s}/{h}", range(3),
              I6 / "scores/urban/urban_singles_s{s}/{k}.npz", S2 / "scores/urban/asturban_s{s}/{h}/{c}__clean.npz"),
}
EFF_KINDS, AST_HEADS = ["esc", "8", "4", "bitnet"], ["esc", "4"]


def stems(names):
    return [Path(n).stem for n in names]


def main(out):
    res = {}
    meta = {}
    for ds, (ce, ca) in CSV.items():
        e = pd.read_csv(ce, usecols=META)
        a = pd.read_csv(ca, usecols=META)
        se, sa = stems(e.name), stems(a.name)
        r = {"n_eff": len(e), "n_ast": len(a),
             "same_stems_same_order": se == sa,
             "n_stem_mismatch": int(sum(x != y for x, y in zip(se, sa))),
             "same_set_of_stems": set(se) == set(sa),
             "duplicate_stems_eff": int(len(se) - len(set(se))),
             "label_equal": bool((e.label.values == a.label.values).all()),
             "class_id_equal": bool((e.class_id.values == a.class_id.values).all()),
             "folder_equal": bool((e.folder.values == a.folder.values).all()),
             "crc_eff_stems": zlib.crc32("\n".join(se).encode()),
             "crc_ast_stems": zlib.crc32("\n".join(sa).encode()),
             "eff_name_example": e.name.iloc[0], "ast_name_example": a.name.iloc[0]}
        # official metadata
        if ds == "esc50":
            m = pd.read_csv(next((DS / "ESC-50/audio_raw").rglob("esc50.csv")))
            m["stem"] = stems(m.filename)
            m = m.set_index("stem").loc[se]
            r["official_fold_equal"] = bool((m.fold.values == e.folder.values).all())
            r["official_target_equal"] = bool((m.target.values == e.label.values).all())
            # file name token convention: FOLD-CLIPID-TAKE-TARGET
            tok = [s.split("-") for s in se]
            r["name_fold_token_equal"] = bool(all(int(t[0]) == f for t, f in zip(tok, e.folder)))
            r["name_target_token_equal"] = bool(all(int(t[3]) == lab for t, lab in zip(tok, e.label)))
        else:
            m = pd.read_csv(next((DS / "Urban8k/audio_raw").rglob("UrbanSound8K.csv")))
            m["stem"] = stems(m.slice_file_name)
            m = m.set_index("stem").loc[se]
            r["official_fold_equal"] = bool((m.fold.values == e.folder.values).all())
            r["official_classID_equal"] = bool((m.classID.values == e.label.values).all())
            r["class_id_equals_label"] = bool((e.class_id.values == e.label.values).all())
        # AST sidecar checksum
        sc = json.loads((S2 / f"emb/{'esc50' if ds == 'esc50' else 'urban8k'}/w4g128__clean.json").read_text())
        r["sidecar_names_crc32"] = sc["names_crc32"]
        r["sidecar_csv"] = sc["csv"]
        res[f"csv_{ds}"] = r
        meta[ds] = e

    for task, (ds, lab, effrun, astrun, seeds, effsc, astsc) in TASKS.items():
        e = meta[ds]
        y_csv = e[lab].to_numpy()
        fold_csv = e.folder.to_numpy()
        rec = np.array([s.split("-")[1] for s in stems(e.name)]) if ds == "esc50" else None
        if ds == "urban":
            m = pd.read_csv(next((DS / "Urban8k/audio_raw").rglob("UrbanSound8K.csv")))
            m["stem"] = stems(m.slice_file_name)
            rec = m.set_index("stem").loc[stems(e.name)].fsID.astype(str).to_numpy()
        t = {"split_files_compared": 0, "split_mismatch": [], "test_not_official": [], "iv_in_test": [],
             "iv_recording_overlap": [], "score_y_mismatch": [], "score_iv_vs_split_mismatch": [],
             "score_tf_mismatch": [], "score_files_compared": 0, "iv_differs_across_seeds": None}
        ivs_by_seed = {}
        for s in seeds:
            runs = [Path(str(effrun).format(s=s, k=k)) for k in EFF_KINDS] + \
                   [Path(str(astrun).format(s=s, h=h)) for h in AST_HEADS]
            ref = None
            for run in runs:
                folds = sorted(int(p.name.split("_")[1]) for p in run.glob("fold_*"))
                sp = {f: json.loads((run / f"fold_{f}" / "split.json").read_text()) for f in folds}
                cur = {f: (sorted(v["test_idx"]), sorted(v["inner_val_idx"])) for f, v in sp.items()}
                t["split_files_compared"] += len(folds)
                for f, (te, iv) in cur.items():
                    if sorted(np.flatnonzero(fold_csv == f).tolist()) != te:
                        t["test_not_official"].append(f"{run} fold {f}")
                    if set(iv) & set(np.flatnonzero(fold_csv == f).tolist()):
                        t["iv_in_test"].append(f"{run} fold {f}")
                    fit = np.setdiff1d(np.flatnonzero(fold_csv != f), iv)
                    if set(rec[iv]) & set(rec[fit]):
                        t["iv_recording_overlap"].append(f"{run} fold {f}: {len(set(rec[iv]) & set(rec[fit]))}")
                if ref is None:
                    ref = cur
                    ivs_by_seed[s] = cur
                elif cur != ref:
                    t["split_mismatch"].append(str(run))
            # score files used by the analysis
            files = [Path(str(effsc).format(s=s, k=k)) for k in EFF_KINDS] + \
                    [p for h in AST_HEADS for p in Path(str(astsc).format(s=s, h=h, c="x")).parent.glob("*__clean.npz")]
            for p in files:
                z = np.load(p)
                t["score_files_compared"] += 1
                if not np.array_equal(z["y"], y_csv):
                    t["score_y_mismatch"].append(str(p))
                tf = z["test_fold"]
                if not np.array_equal(tf, fold_csv):
                    t["score_tf_mismatch"].append(str(p))
                for f, (te, iv) in ivs_by_seed[s].items():
                    if sorted(z[f"iv_idx_{f}"].tolist()) != iv:
                        t["score_iv_vs_split_mismatch"].append(f"{p} fold {f}")
        ss = list(ivs_by_seed)
        t["iv_differs_across_seeds"] = any(ivs_by_seed[a] != ivs_by_seed[ss[0]] for a in ss[1:])
        t["n_seeds"] = len(ss)
        t["ok"] = not any(t[k] for k in ["split_mismatch", "test_not_official", "iv_in_test", "iv_recording_overlap",
                                         "score_y_mismatch", "score_iv_vs_split_mismatch", "score_tf_mismatch"])
        res[f"task_{task}"] = t
    Path(out).write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main(sys.argv[1])
