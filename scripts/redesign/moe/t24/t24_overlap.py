"""T24 overlap check (ANALYSIS_PLAN.md C9, round plan section 12). Counts the Freesound ids of the T24 dataset that are
also in FSDKaggle2018, in FSD50K and among the UrbanSound8K recordings the mixer uses as background noise, and the
clips whose bg noise (stage 2's NoiseMaker, replayed by name) was drawn from a recording that is itself a clip of the
T24 dataset. Writes the counts and the list of test clips to drop for the sensitivity run of t24_analyze.py --drop.

A test clip is dropped when its Freesound id is in FSDKaggle2018, FSD50K or the UrbanSound8K noise bank, or when the
bg recording drawn for it has the Freesound id of any clip of the T24 dataset. A dataset without Freesound ids (D4)
gets counts of zero and an empty list, with the reason.

    python t24_overlap.py --r R --d d3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t24_common as C  # noqa: E402

from audio_shift import NoiseMaker, index_wavs  # noqa: E402

DATA = Path.home() / "Documents/DATASETS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--d", required=True)
    a = ap.parse_args()
    C.patch_loader()
    al = pd.read_csv(a.r / "csv" / a.d / "all.csv", dtype={"freesound_id": str})
    ev = pd.read_csv(a.r / "csv" / a.d / "eval.csv", dtype={"freesound_id": str})
    out = a.r / "tables" / a.d
    out.mkdir(parents=True, exist_ok=True)
    fid = al.freesound_id.fillna("").astype(str)
    if (fid == "").all():
        res = {"dataset": a.d, "freesound_ids": 0, "note": "no Freesound ids: own recordings, overlap with Freesound sources impossible by source"}
        (out / "overlap_counts.json").write_text(json.dumps(res, indent=1))
        (out / "overlap_drop.txt").write_text("")
        print(json.dumps(res, indent=1))
        return
    k18 = set()
    for f in ("train_post_competition.csv", "test_post_competition_scoring_clips.csv"):
        k18 |= set(pd.read_csv(DATA / "FSDKaggle2018/meta_x/FSDKaggle2018.meta" / f).freesound_id.astype(str))
    f50 = set()
    for f in ("dev.csv", "eval.csv"):
        f50 |= set(pd.read_csv(DATA / "FSD50K/meta_x/FSD50K.ground_truth" / f).fname.astype(str))
    u8_names = pd.read_csv(C.C21.URB_CSV, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
    u8_index = index_wavs(C.C21.URB_AUDIO)
    bank = [b for b in u8_names if b in u8_index]
    u8 = {b.split("-")[0] for b in bank}
    own = set(fid[fid != ""])
    is_test = al.folder.to_numpy() == 0
    # replay the bg draw of every eval row (the rows extracted under noise), as the extractors do it
    lens = dict(zip(ev.name, np.load(a.r / "emb" / a.d / "ast_eval" / "lengths16k.npy")))
    nm = NoiseMaker(bank, u8_index, True)
    bg_fs = {}
    for name in ev.name:
        stem = Path(name).stem
        nm(stem, "bg", int(lens[name]))
        bg_fs[name] = nm.plan[stem]["bg_clip"].split("-")[0]
    bg_own = {n for n, b in bg_fs.items() if b in own}
    sets = {"FSDKaggle2018": k18, "FSD50K": f50, "UrbanSound8K_noise_bank": u8}
    res = {"dataset": a.d, "freesound_ids": len(own), "n_clips": len(al), "n_test": int(is_test.sum())}
    drop = np.zeros(len(al), bool)
    for nme, s in sets.items():
        hit = fid.isin(s).to_numpy() & (fid != "").to_numpy()
        res[f"shared_with_{nme}"] = {"clips": int(hit.sum()), "test_clips": int((hit & is_test).sum()), "ids": int(len(own & s))}
        drop |= hit
    hit_bg = al.name.isin(bg_own).to_numpy()
    res["bg_noise_from_own_recording"] = {"eval_rows": int(hit_bg.sum()), "test_clips": int((hit_bg & is_test).sum()),
                                         "own_ids_in_noise_bank": int(len(own & u8))}
    drop |= hit_bg
    res["test_clips_dropped"] = int((drop & is_test).sum())
    (out / "overlap_counts.json").write_text(json.dumps(res, indent=1))
    (out / "overlap_drop.txt").write_text("\n".join(al.name[drop & is_test]) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
