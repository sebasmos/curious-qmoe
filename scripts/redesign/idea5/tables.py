"""Markdown tables and the Stage 1 go rule from analyze.py's analysis.json.

    python scripts/redesign/idea5/tables.py OUT/analysis.json > OUT/tables.md
"""
from __future__ import annotations

import json
import sys

GO_GAP_PTS, GO_AUROC = 2.0, 0.75
BB = ["fp32", "w8", "w4", "w4a8", "w3", "w2"]
HEADS = ["esc", "8", "4", "1", "bitnet"]


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main(path):
    rep = json.loads(open(path).read())
    print("# Idea 5 Stage 1 tables\n")
    print("Backbone bit-ops per clip (T = 1e12): " + ", ".join(
        f"{b} {rep['backbone_bops'][b] / 1e12:.1f}T" for b in BB if b in rep["backbone_bops"]) + "\n")
    for task, t in rep["tasks"].items():
        print(f"## {task}\n")
        print("Head bit-ops per clip: " + ", ".join(f"{h} {v / 1e9:.3f}G" for h, v in t["head_bops"].items()) + "\n")
        rows = t["f1_table"]
        audios = list(dict.fromkeys(r["audio"] for r in rows))
        heads = [h for h in HEADS if any(r["head"] == h for r in rows)]
        print("### Macro F1 (mean of folds); gap to FP32 backbone + FP32 head on the same audio, points\n")
        print("| audio | backbone | " + " | ".join(f"head {h}" for h in heads) + " |")
        print("|---|---|" + "---|" * len(heads))
        for a in audios:
            for b in BB:
                cells = []
                for h in heads:
                    r = [x for x in rows if x["audio"] == a and x["backbone"] == b and x["head"] == h]
                    cells.append(f"{r[0]['f1']:.3f} ({r[0]['gap_pts']:+.1f})" if r else "")
                if any(cells):
                    print(f"| {a} | {b} | " + " | ".join(cells) + " |")
        print()
        print("### AUROC for the system's errors that the reference fixes (systems with gap >= 1 point)\n")
        sigs = sorted({k for r in t["auroc"] for k in r["auroc"]})
        print("| audio | system | gap | n target | " + " | ".join(sigs) + " |")
        print("|---|---|---|---|" + "---|" * len(sigs))
        go = []
        for r in t["auroc"]:
            if r["gap_pts"] >= GO_GAP_PTS:
                best = max((v for v in r["auroc"].values() if v is not None), default=None)
                go.append((r["audio"], f"{r['backbone']}+{r['head']}", r["gap_pts"], best,
                           max(r["auroc"], key=lambda k: -1 if r["auroc"][k] is None else r["auroc"][k])))
            if r["gap_pts"] < 1:
                continue
            print(f"| {r['audio']} | {r['backbone']}+{r['head']} | {r['gap_pts']:.1f} | {r['n_target']} | "
                  + " | ".join(fmt(r["auroc"].get(s), 2) for s in sigs) + " |")
        print()
        print("### Go rule: gap >= 2 points and some signal AUROC >= 0.75\n")
        for a, s, g, best, which in go:
            print(f"- {a} {s}: gap {g:.1f}, best AUROC {fmt(best, 3)} ({which}) -> "
                  f"{'MET' if best is not None and best >= GO_AUROC else 'not met'}")
        if not go:
            print("- no system has a gap of 2 points")
        print()
        print("### Primary cascade points (fixed after the clean and background-noise ESC-50 pass, before the white/pink-noise and UrbanSound8K results)\n")
        print("Thresholds on each fold's inner validation split. "
              f"{sum(1 for c in t['cascades'] if 'curve' not in c and c['point'] != 'oracle')} honest points "
              "were computed in all; only these are claimed.\n")
        print("| audio | cascade | signal | point | escalated | F1 | bit-ops (T) | best single at <= cost | "
              "beats front | difference, 95% bootstrap (points) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for c in t["cascades"]:
            if c.get("primary"):
                bs = c["best_single_at_cost"]
                beat = "" if bs is None else f"{'yes' if c['f1'] > bs else 'no'} ({100 * (c['f1'] - bs):+.1f})"
                ci = c.get("diff_vs_front_ci95")
                ci = "" if ci is None else f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] vs {c['best_single']}"
                print(f"| {c['audio']} | {c['low']} to {c['high']} | {c['signal']} | {c['point']} | "
                      f"{c['escalation']:.2f} | {c['f1']:.3f} | {c['bops'] / 1e12:.1f} | {fmt(bs)} | {beat} | {ci} |")
        print()
        print("### Cascades (low system first, the higher system re-run on escalation)\n")
        print("| audio | cascade | signal | point | escalated | F1 | bit-ops (T) | best single at <= cost | beats front |")
        print("|---|---|---|---|---|---|---|---|---|")
        for c in t["cascades"]:
            if "curve" in c:
                gain = [(p["f1"] - p["best_single_at_cost"], p) for p in c["curve"]
                        if p["best_single_at_cost"] is not None]
                g, p = max(gain, key=lambda x: x[0])
                print(f"| {c['audio']} | {c['low']} to {c.get('high', 'fp32+esc')} | {c['signal']} | upper bound (tuned on test), best q | "
                      f"{p['q']:.2f} | {p['f1']:.3f} | {p['bops'] / 1e12:.1f} | {p['best_single_at_cost']:.3f} | "
                      f"{'yes' if g > 0 else 'no'} ({100 * g:+.1f}) |")
                continue
            if c["point"] not in ("max_f1", "oracle") and not c["point"].startswith("budget_w8+esc") \
                    and not c["point"].startswith("budget_w4+esc"):
                continue
            bs = c["best_single_at_cost"]
            beat = "" if bs is None else f"{'yes' if c['f1'] > bs else 'no'} ({100 * (c['f1'] - bs):+.1f})"
            print(f"| {c['audio']} | {c['low']} to {c.get('high', 'fp32+esc')} | {c['signal']} | {c['point']} | {c['escalation']:.2f} | "
                  f"{c['f1']:.3f} | {c['bops'] / 1e12:.1f} | {fmt(bs)} | {beat} |")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
