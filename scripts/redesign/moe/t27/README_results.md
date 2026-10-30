# T27 (POST HOC), S2 against the always-on MN10+MN20+AST average

Branch `moe/t27` of `curious-qmoe` (worktree `~/Documents/qmoe-work/t27`, from `moe/t26` at `b6b5276`), never pushed. Plan and the analysis, recompute and manifest scripts committed at `856efb5` (15:38:41, 29 Oct, Irish time = UTC) before any T27 number; the analysis started at 15:38:50 and printed its first number at 15:40:41 (`logs/analyze.log`). `ANALYSIS_PLAN.md` here equals the committed plan. **Post hoc**: chosen after T26's results were seen; no pass bar is confirmatory. Every number below comes from `tables/summary.json` and `tables/<d>/results.json`.

S2 is T26's, unchanged: budgets read from T26's verdict files (r2 50%, r3 20% on both datasets), thresholds rebuilt on clean inner validation. The three-model average (call it AVG3) is the argmax of the mean of the three softmaxes under T26's scalers (all refit), priced as the sum of the three models' full costs. Before any T27 number the script rebuilt every T26 static, S1, S2 and cascade row on field and clean (1,050 rows on d1, 630 on d3) to 1.4e-14 F1 and S2 minus T26's widened hull to 0.0.

## Results (macro F1 points; cost in T priced bit-ops per clip; NB = Nadeau-Bengio fold t-test, two-sided)

| Dataset, bed | S2 F1 at T | AVG3 F1 at T | (i) S2 minus AVG3, NB 95%, p, folds +/0/- | Extended hull at S2's cost | (ii) S2 minus extended hull, NB 95%, p, folds + | T26 S2 minus 17-point hull |
|---|---|---|---|---|---|---|
| FSDnoisy18k field | 72.59 at 0.951 | 72.73 at 2.776 | -0.15 (-0.42 to 0.12), p 0.21, 1/1/3 | MN10+MN20 to AVG3 at 0.315, 71.03 | **+1.56** (1.26 to 1.86), p 0.0001, 5 of 5 | +2.33 |
| FSDnoisy18k clean | 85.85 at 0.554 | 85.96 at 2.776 | -0.11 (-0.32 to 0.11), p 0.24, 2/0/3 | MN10+MN20 to AVG3 at 0.166, 84.42 | +1.43 (1.06 to 1.80), p 0.0004, 5 of 5 | +1.65 |
| FSDKaggle2018 field | 79.07 at 1.239 | 79.06 at 2.715 | +0.00 (-0.01 to 0.02), p 0.54, 1/4/0 | MN10+MN20 to AVG3 at 0.433, 77.90 | **+1.17** (0.72 to 1.61), p 0.002, 5 of 5 | +1.74 |
| FSDKaggle2018 clean | 94.61 at 0.478 | 94.60 at 2.715 | +0.01 (-0.02 to 0.03), p 0.34, 2/3/0 | MN10+MN20 to AVG3 at 0.141, 93.77 | +0.84 (0.51 to 1.17), p 0.002, 5 of 5 | +0.95 |

Sign-flip p for (ii) is 0.0625 everywhere (its floor with 5 folds). Fold values: (i) d3 field -0.44, -0.16, -0.16, 0.03, 0.00; d1 field 0.00, 0.00, 0.02, 0.00, 0.00 (S2 and AVG3 make the same predictions on almost every d1 test clip; with four of five fold differences exactly zero, the d1 interval for (i) means identical predictions, not a tight estimate). (ii) d3 field 1.55, 1.34, 1.72, 1.36, 1.82; d1 field 1.28, 1.29, 1.22, 1.31, 0.74.

Reading: (i) S2 is level with always running all three models (within 0.15 points on average, not significant; on FSDnoisy18k field three of five folds are negative, -0.44, -0.16 and -0.16, so the interval does not show equivalence) at 34% (d3) and 46% (d1) of AVG3's field cost, 20% and 18% on clean. (ii) Adding AVG3 to the hull cuts S2's margin from +2.33 to +1.56 on d3 field and from +1.74 to +1.17 on d1 field, so about a third of T26's gain was AST's information; the rest survives against random routing between the MN10+MN20 pair and AVG3 at S2's cost, so margin-based routing accounts for it. At S2's cost the extended hull is that straight line on every bed.

Corrected after the independent audit of 29 Oct (`2026-10-29_audit-t27`): this paragraph first said "no fold-level difference" for (i), which is wrong for FSDnoisy18k field. No number changed.

Context (field; clean in brackets):

| System | FSDnoisy18k | FSDKaggle2018 |
|---|---|---|
| S1 (20%) | 69.24 at 0.057 (83.29 at 0.046) | 76.88 at 0.065 (93.55 at 0.043) |
| MN10+MN20 average | 70.25 at 0.112 (84.11) | 77.01 at 0.110 (93.64) |
| always-average (MN10+AST) | 70.23 at 2.695 (84.60) | 77.74 at 2.635 (93.84) |
| AVG3 | 72.73 at 2.776 (85.96), 53.9 G MACs | 79.06 at 2.715 (94.60), 52.8 G MACs |
| S2 tier 2 / tier 3 rates | 68% / 39% (53% / 24%) | 70% / 48% (50% / 20%) |

## Own recompute

`t27_recompute.py` reads only the raw score files through T26's independent recompute (own softmax, margin, cut, sklearn F1, linear envelope, 2,000-draw Monte Carlo time-sharing, own NB), budgets fixed at T26's, AVG3 added to its statics. Worst absolute difference from the analysis over S2 F1, AVG3 F1, (i), (ii), the extended hull's F1 and S2 minus the 17-point hull: 0.0027 points (bar 0.01); (i) agrees to 1e-12; same hull endpoints and weights on all four beds; worst p difference 1.5e-5 (`tables/recompute_vs_analysis.csv`, `tables/recompute_agreement.json`).

## Commands (worktree, `R` = this folder, `T` = scripts/redesign/moe/t27; nice 10, OMP_NUM_THREADS=2, CPU only)

```
python T/t27_analyze.py --r $R
python T/t27_recompute.py --r $R
python T/t27_manifest.py $R 856efb5
```

## Deviations

None from the plan. The analysis took about 6 minutes; the recompute about 8. No extraction, no GPU, nothing written outside this folder and the worktree apart from `~/Documents/qmoe-work/t27-R.txt` (the folder path, as earlier T runs keep).

## Files

`ANALYSIS_PLAN.md`; `tables/summary.csv`, `tables/summary.json` (both datasets and beds); `tables/<d>/results.json`, `by_seed_fold.csv` (every static including AVG3, S1, S2, cascade per seed and fold), `statics_by_condition.csv`, `s2_vs_extended_hull.csv`; `tables/recompute.json`, `recompute_vs_analysis.csv`, `recompute_agreement.json`; `logs/`; `scripts/` (copy of `scripts/redesign/moe/t27` at the final commit); `run_manifest.json`.
