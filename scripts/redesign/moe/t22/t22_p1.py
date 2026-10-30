"""T22 precondition P1 (ANALYSIS_PLAN.md D): fold 1, clean inner validation, macro F1 of the W8A8 coarse expert minus MN10-Q8 (stored),
seeds 0 to 2. Reads inner-validation rows only (the cached clean coarse view; stored iv logits). No test row is read.
    python t22_p1.py --r R --tasks esc50 urban tau"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t22_common import *  # noqa
import t22_view as V, t22_models as TM
import t13_common as T13
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--tasks", nargs="+", default=["esc50", "urban", "tau"])
ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS); ap.add_argument("--runs", type=Path, default=None); a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2))); runs = a.runs or a.r / "runs"
rows, verdict = [], {}
for task in a.tasks:
    K = TASKS[task]["K"]; X = np.load(CACHE / task / "clean_mel.npy", mmap_mode="r"); crow = np.load(CACHE / task / "clean_rows.npy"); cpos = {int(r): i for i, r in enumerate(crow)}
    for s in a.seeds:
        iv = inner_split(task, s, 1); D2 = T13.load(task, "mn10", "clean", s); D3 = T13.load(task, "astw4", "clean", s)
        assert np.array_equal(np.sort(D2["iv"][1][0]), iv) and np.array_equal(D2["iv"][1][0], D3["iv"][1][0]), "inner split differs between tiers"
        o2 = np.argsort(D2["iv"][1][0]); y = D2["y"][iv]
        m = TM.load_coarse(runs / f"{task}_f1_s{s}.pt", K); q = V.quantize_w8a8(m, "cpu")
        Xi = np.stack([X[cpos[int(r)]] for r in iv]); t0 = time.time()
        lq = TM.run(q, Xi); lf = TM.run(m, Xi); dt = time.time() - t0
        rd = a.r / "scores_p1"; rd.mkdir(parents=True, exist_ok=True)
        np.savez(rd / f"{task}_f1_s{s}_iv.npz", iv_idx=iv, y=y, logits_w8a8=lq, logits_fp32=lf)  # raw inner-validation scores for the recompute
        meta = json.load(open(runs / f"{task}_f1_s{s}.json"))
        r = dict(task=task, seed=s, n_iv=len(iv), coarse_w8a8=macro_f1(y, lq.argmax(1), K) * 100, coarse_fp32=macro_f1(y, lf.argmax(1), K) * 100,
                 mn10_q8=macro_f1(y, D2["iv"][1][1][o2].argmax(1), K) * 100, ast_q4=macro_f1(y, D3["iv"][1][1][o2].argmax(1), K) * 100,
                 train_best_iv_fp32=meta["best_iv_f1_fp32"] * 100, best_epoch=meta["best_epoch"], lr=meta["lr"], cpu_clips_per_s=len(iv) / max(dt, 1e-9) * 2)
        rows.append(r); print(r, flush=True)
    g = pd.DataFrame([r for r in rows if r["task"] == task])
    d = float(g.coarse_w8a8.mean() - g.mn10_q8.mean())
    verdict[task] = dict(coarse_w8a8=float(g.coarse_w8a8.mean()), coarse_fp32=float(g.coarse_fp32.mean()), mn10_q8=float(g.mn10_q8.mean()),
                         ast_q4=float(g.ast_q4.mean()), diff_w8a8_minus_mn10=d, holds=bool(d >= -5.0), seeds=a.seeds)
out = a.r / "tables"; out.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out / "p1_by_seed.csv", index=False)
n = sum(v["holds"] for v in verdict.values()); n_fail = sum(not v["holds"] for v in verdict.values())
V_ = {"rule": "P1: W8A8 coarse expert within 5 points of MN10-Q8, clean inner validation, fold 1, macro F1 (seed mean), on two of three tasks",
      "tasks": verdict, "n_tasks_hold": n, "tasks_scored": a.tasks, "passes": (False if n_fail >= 2 else (bool(n >= 2) if len(a.tasks) == 3 else "incomplete")),
      "note": "fails as soon as two tasks miss the bar, whatever the third gives"}
(out / "p1.json").write_text(json.dumps(V_, indent=1)); print(json.dumps(V_, indent=1))
