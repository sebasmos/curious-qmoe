#!/usr/bin/env python3
"""Compact markdown tables across datasets from router_pilot.py outputs.

    python report_tables.py <results dir> urban esc quinn
"""
import json
import sys
from pathlib import Path

import numpy as np

RES = Path(sys.argv[1])
DS = sys.argv[2:]
PAIRS = ["bitnet->4", "bitnet->esc", "1->esc", "4->esc"]
ROUTERS = ["lr_out", "mlp_out", "lr_emb", "lr_both", "mlp_both"]
out = []


def recs(ds, src):
    R = [json.loads(p.read_text()) for p in sorted((RES / "router" / ds).glob(f"rec_{src}_fold*.json"))]
    return sorted(R, key=lambda r: r["fold"])


def pt(p, key):
    if not p:
        return "-"
    return f"{p['task']} {p['method']} {p['op']} {p['mean_f1']:.4f}@{p['mean_bitops'] / 1e6:.1f}M: {cmp(p[key])}"


def cmp(v):
    return f"{v['diff']:+.4f} ({v['wins']}/{v['n']}, p={v['p']:.3f})" if v else "-"


out.append("## Defer-target AUROC on the test folds (mean over folds); router trained on the named held-in source\n")
out.append("| dataset | pair | test target rate | margin | " + " | ".join(f"{r} iv / oof / cf" for r in ROUTERS) + " |")
out.append("|" + "---|" * (4 + len(ROUTERS)))
for ds in DS:
    S = {src: recs(ds, src) for src in ("iv", "oof", "cf")}
    for pair in PAIRS:
        base = next((S[s] for s in ("iv", "oof", "cf") if S[s] and all(pair in r["pairs"] for r in S[s])), None)
        if not base:
            continue
        mar = np.mean([r["pairs"][pair]["methods"]["margin"]["auroc_test"] for r in base])
        tr = np.mean([r["pairs"][pair]["test_target_rate"] for r in base])
        cells = []
        for rt in ROUTERS:
            v = []
            for s in ("iv", "oof", "cf"):
                R = S[s]
                ok = R and all(pair in r["pairs"] for r in R) and len(R) == len(base)
                v.append(f"{np.mean([r['pairs'][pair]['methods'][rt]['auroc_test'] for r in R]):.3f}" if ok else "-")
            cells.append(" / ".join(v))
        out.append(f"| {ds} | {pair} | {tr:.3f} | {mar:.3f} | " + " | ".join(cells) + " |")

out.append("\n## Accuracy and cost front (mean over folds; diff = paired macro F1 difference, wins/n folds, Wilcoxon p)\n")
out.append("Best operating point of each family against the singles' mix hull and the margin mix hull at the same "
           "mean bit-ops. Operating points are fixed on held-in data; the 'best' point is picked here by its test "
           "diff, so these rows are an optimistic summary of each family.\n")
out.append("| dataset | source | family | point | F1 @ Mbit-ops | vs singles mix | vs margin mix | vs singles+margin mix | ops above singles mix | ops above margin mix | ops above singles+margin mix |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for ds in DS:
    sj = RES / "router" / ds / "summary.json"
    if not sj.exists():
        continue
    S = json.loads(sj.read_text())
    P = S["points"]
    for src in ("iv", "oof", "cf"):
        for fam, meths in (("margin", ["margin"]), ("router (out)", ["lr_out", "mlp_out"]),
                           ("router (emb)", ["lr_emb", "lr_both", "mlp_both"])):
            Q = [p for p in P if p["source"] == src and p["method"] in meths and p.get("vs_single_mix")]
            if not Q:
                continue
            b = max(Q, key=lambda p: p["vs_single_mix"]["diff"])
            above_s = sum(p["vs_single_mix"]["diff"] > 0 and p["vs_single_mix"]["wins"] > p["vs_single_mix"]["losses"]
                          for p in Q)
            Qm = [p for p in Q if p.get("vs_margin_mix")]
            above_m = sum(p["vs_margin_mix"]["diff"] > 0 and p["vs_margin_mix"]["wins"] > p["vs_margin_mix"]["losses"]
                          for p in Qm)
            bm = max(Qm, key=lambda p: p["vs_margin_mix"]["diff"]) if Qm else None
            Qf = [p for p in Q if p.get("vs_front_mix")]
            above_f = sum(p["vs_front_mix"]["diff"] > 0 and p["vs_front_mix"]["wins"] > p["vs_front_mix"]["losses"]
                          for p in Qf)
            bf = max(Qf, key=lambda p: p["vs_front_mix"]["diff"]) if Qf else None
            out.append(f"| {ds} | {src} | {fam} | {b['task']} {b['method']} {b['op']} | {b['mean_f1']:.4f} @ "
                       f"{b['mean_bitops'] / 1e6:.1f} | {cmp(b['vs_single_mix'])} | "
                       f"{(bm['task'] + ' ' + bm['method'] + ' ' + bm['op'] + ': ' + cmp(bm['vs_margin_mix'])) if bm else '-'} | "
                       f"{pt(bf, 'vs_front_mix')} | "
                       f"{above_s}/{len(Q)} | " + (f"{above_m}/{len(Qm)}" if Qm else "-") + " | "
                       + (f"{above_f}/{len(Qf)}" if Qf else "-") + " |")

out.append("\n## Singles and oracles (mean over folds)\n")
out.append("| dataset | bitnet | 1 | 2 | 4 | 8 | 16 | esc | Mbit-ops bitnet / 1 / 4 / 8 / esc | oracle bitnet->4 | oracle 3way |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for ds in DS:
    sj = RES / "router" / ds / "summary.json"
    if not sj.exists():
        continue
    S = json.loads(sj.read_text())
    sg = S["singles"]
    R = recs(ds, "iv")
    o4 = np.mean([r["pairs"]["bitnet->4"]["oracle"]["f1"] for r in R]), np.mean([r["pairs"]["bitnet->4"]["oracle"]["bitops"] for r in R])
    o3 = np.mean([r["three"]["oracle"]["f1"] for r in R]), np.mean([r["three"]["oracle"]["bitops"] for r in R])
    out.append(f"| {ds} | " + " | ".join(f"{np.mean(sg[m]['f1']):.4f}" for m in ["bitnet", "1", "2", "4", "8", "16", "esc"])
               + " | " + " / ".join(f"{S['costs'][m] / 1e6:.1f}" for m in ["bitnet", "1", "4", "8", "esc"])
               + f" | {o4[0]:.4f} @ {o4[1] / 1e6:.1f} | {o3[0]:.4f} @ {o3[1] / 1e6:.1f} |")
print("\n".join(out))
