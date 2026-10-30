"""A head trained on the cheap backbone's clean embeddings (one seed): the
cheap system with that head, and cascades whose cheap pass uses it while the
escalated pass keeps the head trained on clean FP32-backbone embeddings.

Same thresholds, points and costs as analyze_stage2.py (FP32 head, margin
gate, inner-validation thresholds under the same condition). The alternative
run must have the same folds and inner validation splits as the reference
run (same seed and protocol; checked). Records for front.py go to
<out>/records/cheaphead__<condition>.jsonl with every static system of the
reference seed as singles plus the cheap system with its own head.

    python scripts/redesign/idea5/analyze_cheap_head.py --scores SCORES/esc5 --emb-dir EMB/esc50 \
        --runs-root OUT/outputs --ref-tag astesc5_s0 --alt-tag astesc5headw4_s0 --cheap w4 --dataset esc5 \
        --out ANALYSIS/esc5_cheaphead
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_stage2 import (BUDGETS, Scores, backbone_costs, cascade_cost, ece, head_costs, macro_f1,  # noqa: E402
                            parent_of, seed_of, signal, threshold_for_rate, threshold_match)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for a in ("--scores", "--emb-dir", "--runs-root", "--out"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--ref-tag", required=True)
    ap.add_argument("--alt-tag", required=True)
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for f in sorted((args.runs_root / args.ref_tag / "esc").glob("fold_*/split.json")):
        a, b = json.loads(f.read_text()), json.loads((args.runs_root / args.alt_tag / "esc" / f.parent.name /
                                                       "split.json").read_text())
        if a["test_idx"] != b["test_idx"] or a["inner_val_idx"] != b["inner_val_idx"]:
            raise SystemExit(f"{f.parent.name}: the two runs have different splits")
    seed = seed_of(args.ref_tag)
    bcost = backbone_costs(args.emb_dir)
    hb, hbytes = head_costs(args.runs_root, args.ref_tag, ["esc"])["esc"]
    ref = {h: Scores(args.scores / args.ref_tag / h) for h in ("esc", "4")}
    alt = Scores(args.scores / args.alt_tag / "esc")
    hcost4 = head_costs(args.runs_root, args.ref_tag, ["4"])["4"]
    conds = sorted({f.split("__", 1)[1] for f in alt.files})
    if alt.has(args.cheap, "field"):
        conds.append("field")
    static, casc = [], []
    for cond in conds:
        if not alt.has(args.cheap, cond):
            continue
        for h, sc in ref.items():
            hcb, hcbytes = (hb, hbytes) if h == "esc" else hcost4
            for cfg in sorted({f.split("__")[0] for f in sc.files}):
                if not sc.has(cfg, cond):
                    continue
                r = sc.get(cfg, cond)
                for fold in sorted(r["iv"]):
                    m = r["fold"] == fold
                    static.append({"condition": cond, "method": f"{cfg}+{h}", "fold": fold,
                                   "f1": macro_f1(r["y"][m], r["p"][m].argmax(1)), "ece": ece(r["p"][m], r["y"][m]),
                                   "bitops": bcost[cfg]["bitops"] + hcb,
                                   "storage_bytes": bcost[cfg]["storage_bits"] / 8 + hcbytes})
        ra = alt.get(args.cheap, cond)
        for fold in sorted(ra["iv"]):
            m = ra["fold"] == fold
            static.append({"condition": cond, "method": f"{args.cheap}+esc@{args.cheap}", "fold": fold,
                           "f1": macro_f1(ra["y"][m], ra["p"][m].argmax(1)), "ece": ece(ra["p"][m], ra["y"][m]),
                           "bitops": bcost[args.cheap]["bitops"] + hb,
                           "storage_bytes": bcost[args.cheap]["storage_bits"] / 8 + hbytes})
        par = parent_of(args.cheap)
        for exp in (par, "fp32"):
            if not (ref["esc"].has(exp, cond) and ref["esc"].has(par, cond)):
                continue
            re_, rp = ref["esc"].get(exp, cond), ref["esc"].get(par, cond)
            for fold in sorted(ra["iv"]):
                idx, pci = ra["iv"][fold]
                pei = re_["iv"][fold][1]
                yi = ra["y"][idx]
                m = ra["fold"] == fold
                y, pct, pet = ra["y"][m], ra["p"][m], re_["p"][m]
                si, st = signal(pci, "margin"), signal(pct, "margin")
                ths = {k: (threshold_for_rate(si, v), None) for k, v in BUDGETS.items()}
                ths["match"] = threshold_match(si, yi, pci.argmax(1), pei.argmax(1),
                                               macro_f1(yi, rp["iv"][fold][1].argmax(1)))
                for point, (t, reached) in ths.items():
                    esc = st > t
                    rate = float(esc.mean())
                    cc = cascade_cost(bcost[args.cheap], bcost[exp], rate, hb, hb)
                    casc.append({"condition": cond, "method": f"{args.cheap}@own>{exp}+esc|margin", "point": point,
                                 "fold": fold, "f1": macro_f1(y, np.where(esc, pet.argmax(1), pct.argmax(1))),
                                 "ece": ece(np.where(esc[:, None], pet, pct), y), "rate": rate,
                                 "match_reached": reached, "bitops": cc["bitops"],
                                 "storage_bytes": cc["storage_bits"] / 8 + 2 * hbytes})
    S, C = pd.DataFrame(static), pd.DataFrame(casc)
    S.to_csv(args.out / "static.csv", index=False)
    C.to_csv(args.out / "cascades.csv", index=False)
    (args.out / "records").mkdir(exist_ok=True)
    for cond in S["condition"].unique():
        recs = [{"method": r.method, "dataset": f"{args.dataset}|{cond}", "seed": seed, "fold": int(r.fold),
                 "f1_macro": r.f1, "bitops_per_clip": r.bitops, "weight_bytes": r.storage_bytes,
                 "operating_point": "static", "is_single": True} for r in S[S["condition"] == cond].itertuples()]
        recs += [{"method": r.method, "dataset": f"{args.dataset}|{cond}", "seed": seed, "fold": int(r.fold),
                  "f1_macro": r.f1, "bitops_per_clip": r.bitops, "weight_bytes": r.storage_bytes,
                  "operating_point": r.point, "is_single": False} for r in C[C["condition"] == cond].itertuples()]
        with open(args.out / "records" / f"cheaphead__{cond}.jsonl", "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in recs)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
