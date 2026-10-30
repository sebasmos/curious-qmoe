# T27 analysis plan (POST HOC): S2 against the three-model average

**POST HOC.** This check was chosen on 2026-10-29 (Irish time) after T26's results were seen. No pass bar is claimed as confirmatory; every number is descriptive evidence about where S2's gain comes from.

Branch `moe/t27` of `curious-qmoe` (worktree `~/Documents/qmoe-work/t27`), from `moe/t26` at `b6b5276`. This plan and the T27 scripts (`scripts/redesign/moe/t27/`) are committed together before any T27 number. Before this commit only syntax and import checks were run. Results folder: `~/Documents/qmoe-results/2026-10-29_<HHMM>_moe-t27_code-<this commit>/`.

Seen before this commit: every T21, T24, T25 and T26 result, including S2 minus the widened hull (+2.33 on FSDnoisy18k field, +1.74 on FSDKaggle2018 field) and T26's caveat that the widened hull has no three-model average. Not seen: the F1 of the always-on MN10-Q8 + MN20-Q8 + AST-Q4 average on either dataset, or any hull containing it.

## Question

T26's S2 (MN10-Q8; low-margin clips to MN20-Q8, pair mean; still low-margin clips to AST-Q4, mean of the three) beat T25's widened hull. The widened hull holds single models and two-model averages only, so S2's gain may come from AST-Q4's information rather than from routing. T27 asks:

- **(i)** S2 minus the always-on three-model average (MN10-Q8 + MN20-Q8 + AST-Q4 on every clip), each at its own cost.
- **(ii)** S2 minus the widened hull extended with that three-model average as an 18th static point, evaluated at S2's own cost.

## Fixed choices

- **Three-model average**: argmax of the arithmetic mean of the three softmax probability vectors (the same score space and the same rule as T26's two-model averages, always-average and S2's tier 3). Each model keeps the scaler T26 used (chosen on clean inner validation; all `refit` on both datasets). Cost per clip = MN10-Q8 + MN20-Q8 + AST-Q4 full priced costs (front end counted per model, as T25 prices pairs). MACs reported as a secondary view.
- **S2 exactly as T26**: budgets read from T26's verdict files (`tables/a/verdict.json`, `tables/c/d3/verdict.json`: r2 = 50%, r3 = 20% on both), thresholds rebuilt per (seed, fold) on clean inner validation by T26's `t26_core.s2_thr`. No reselection. Before any new number the script rebuilds every T26 static, S1 and S2 per (condition, seed, fold) and must match T26's `by_seed_fold.csv` to 1e-9 (F1 and cost), and S2 minus the widened hull must match T26's verdict to 1e-9; otherwise it stops.
- **Extended hull**: T26's `compare_all` with the three-model average added to T25's 17 statics (`hull_pair` at S2's mean test cost per condition, per (seed, fold) hull F1 by T21's exact `exp_f1`).
- **Datasets and seeds**: FSDnoisy18k (d3, T24's fixed-test design, 5 fold models, seeds 0 to 2, 947 test clips, NB ratio 0.066) and FSDKaggle2018 (d1, T21's pool design, 5 folds, seeds 0 to 4, NB ratio 0.25). Both are post hoc here.
- **Beds**: field mix (primary for reading) and clean audio (descriptive).
- **Statistics**: fold-level differences (seeds averaged within folds), paired t with the Nadeau-Bengio correction, two-sided, 95% interval (T26's `fold_test`); folds positive out of 5; sign-flip p reported beside it. No bootstrap, no Holm.
- **Also reported, for context**: the three-model average's F1 and cost, S1 (T26's 20% budget), the MN10+MN20 average, and S2 minus T26's 17-point widened hull.
- **Costs**: priced bit-ops per clip in T (10^12), T26's cost arrays.

## Recompute

`t27_recompute.py` reads only the raw score files (T21, T24, T25, T26 `scores/`), with T26's independent recompute primitives (`t26_recompute.py`: own softmax, margin, threshold, sklearn macro F1, linear envelope, Monte Carlo time-sharing at 2,000 draws, own NB test) and none of the analysis helpers, apart from the per-clip cost arrays (as T26's recompute). Budgets fixed at T26's. It recomputes (i) and (ii) on both datasets and beds; agreement within 0.01 F1 points on each difference is required, and any larger gap is reported.

## Machine and disk

CPU only, nice 10, `OMP_NUM_THREADS=2`, no extraction, no GPU. Free disk checked above 25 GB before writing.
