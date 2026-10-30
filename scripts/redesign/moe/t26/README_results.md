# T26, MN10-Q8 then MN20-Q8 against the widened hull (Parts A, B, C)

Branch `moe/t26` of `curious-qmoe` (worktree `~/Documents/qmoe-work/t26`, from `moe/t25` at `23175a9`, `moe/t24` merged in as `be5431f`), never pushed. Plan and every script committed at `6ecea17` (09:12, 29 Oct) before any T26 number; `ANALYSIS_PLAN.md` here equals the committed plan. No amendment. Every number below comes from `tables/a/verdict.json`, `tables/b/compare_by_condition.csv`, `tables/c/<d>/precondition.json`, `tables/c/<d>/verdict.json`, `tables/c/verdict_combined.json` and the `*_by_seed_fold.csv` files beside them.

S1: MN10-Q8, then MN20-Q8 on low-margin clips, average of the two. S2: MN10-Q8, then MN20-Q8 averaged, then AST-Q4 on clips still low-margin after the pair, average of the three. Budgets chosen on clean inner validation against the widened hull. Widened hull: T25's 17 statics (every small CNN at 8 bits, every two-CNN average, AST-Q4, always-average), evaluated at each system's own cost.

## Verdict (Part C, pre-registered)

| Rule | FSDnoisy18k (d3) | TUT Acoustic Scenes 2017 (d4) |
|---|---|---|
| Precondition: MN20-Q8 minus MN10-Q8, field mix, at least 1.0 (mean over the five fold models) | +2.14 (fold models 1.05, 3.17, 3.34, 1.56, 1.60). PASS | +0.51 (1.40, -0.63, 0.83, 1.17, -0.20). FAIL, stopped |
| H1-C: S1 minus the widened hull, field mix, at least 1.0, NB, Holm over d3 | +0.39, NB 95% -0.84 to 1.61, p 0.43 (Holm 0.43, family of 22: 1.0), bootstrap -0.80 to 1.53, 2 of 5 folds (-0.41, -0.06, -0.13, 1.66, 0.86), sign-flip p 0.50. **FAIL** | not tested |
| S2 minus the widened hull, field mix (secondary, not in Holm) | +2.33, NB 1.87 to 2.80, p 0.0001, 5 of 5 | not tested |
| Clean audio (secondary) | S1 +0.28 (NB -0.53 to 1.08, 4 of 5); S2 +1.65 (1.34 to 1.97, p 0.0001, 5 of 5) | |

**H1-C fails.** The Holm set is d3 alone. Own recompute from the raw score files (`tables/c/d3/recompute.json`) agrees: at the analysis's 20% budget it gives +0.385 (p 0.433) against the analysis's +0.386 (p 0.432). With its own linear inner-validation hull, the recompute picks the 50% budget instead (gain on inner validation -0.507 against -0.573; the analysis's Monte Carlo hull had 20% at -0.503 and 50% at -0.504, a near tie), and S1 is then +0.69 (p 0.072, 5 of 5). That also misses the 1.0 bar.

On d3 S1 sits 1.01 points below the MN10+MN20 average (NB -1.96 to -0.06, 0 of 5 folds), at half its cost. The hull at S1's cost (0.057 T) is a mix of MN05+MN10 and MN05+MN20, and S1 barely clears it.

## Part A, FSDKaggle2018 (POST HOC: MN20-Q8's scores there were seen in T25)

T21's pool design, 5 folds, seeds 0 to 4, NB ratio 0.25. T25's statics, T21 cascade and widened-hull gains reproduced to 1.4e-14 first. Budgets: S1 20%, S2 (50%, 20%).

| Bed | System | F1 | Cost (T) | Escalated | Widened hull at its cost | Minus hull | NB 95% | p | Folds + | Bootstrap 95% |
|---|---|---|---|---|---|---|---|---|---|---|
| field | S1 | 76.88 | 0.065 | 47% | MN05+MN10 to MN10+MN20, 75.00 | +1.88 | 0.77 to 2.98 | 0.009 | 5 of 5 | 1.44 to 2.31 |
| field | S2 | 79.07 | 1.239 | 70% (AST 48%) | MN10+MN20 to always-average, 77.33 | +1.74 | 1.19 to 2.29 | 0.0009 | 5 of 5 | 1.46 to 2.06 |
| field | T21 cascade | 77.57 | 1.154 | 47% | 77.30 | +0.27 | -1.08 to 1.62 | 0.61 | 3 of 5 | |
| clean | S1 | 93.55 | 0.043 | 20% | MN04+DyMN04 to MN10+DyMN04, 92.34 | +1.21 | 0.74 to 1.67 | 0.002 | 5 of 5 | 0.82 to 1.66 |
| clean | S2 | 94.61 | 0.478 | 50% (AST 20%) | 93.67 | +0.95 | 0.60 to 1.29 | 0.002 | 5 of 5 | 0.69 to 1.24 |

S1 against the MN10+MN20 average (77.01 at 0.110 T): -0.14 field (NB -0.35 to 0.08), -0.09 clean, at 59% (field) and 40% (clean) of the pair's cost. S2 against the pair: +2.05 field (p 0.002), +0.97 clean. S2 against the T21 cascade: +1.49 field (p 0.072, 5 of 5), +0.95 clean (p 0.008). Own recompute: same budgets, same hull endpoints, S1 +1.87 field and +1.21 clean, S2 +1.74 and +0.95.

## Part B, development data (DESCRIPTIVE, stored scores)

| Task, bed | S1 F1 at T | S1 minus hull (NB 95%), folds + | S2 F1 at T | S2 minus hull (NB 95%), folds + | MN10+MN20 F1 at T |
|---|---|---|---|---|---|
| ESC-50 field | 87.07 at 0.060 | +1.27 (0.35 to 2.18), 5 of 5 | 89.05 at 1.31 | +1.39 (1.01 to 1.76), 5 of 5 | 87.30 at 0.125 |
| ESC-50 clean | 96.30 at 0.045 | +0.68 (-0.04 to 1.40), 5 of 5 | 96.90 at 0.69 | +0.52 (-0.02 to 1.06), 5 of 5 | 96.35 |
| UrbanSound8K field | 81.60 at 0.040 | +0.23 (-1.04 to 1.51), 6 of 10 | 83.34 at 0.68 | +1.10 (0.50 to 1.70), 9 of 10 | 82.03 at 0.102 |
| UrbanSound8K clean | 87.71 at 0.036 | +0.45 (-1.07 to 1.97), 7 of 10 | 88.93 at 0.50 | +0.94 (0.03 to 1.85), 9 of 10 | 87.87 |
| TAU device shift | 46.60 at 0.145 | +0.73 (0.06 to 1.40), 5 of 5 | 49.27 at 4.78 | +0.68 (-0.12 to 1.49), 5 of 5 | 46.57 at 0.246 |
| TAU device A | 69.95 at 0.123 | +1.82 (1.33 to 2.31), 5 of 5 | 71.51 at 3.67 | +0.72 (0.06 to 1.38), 5 of 5 | 70.79 |

Budgets: S1 10%, 10%, 30%; S2 (30, 20), (50, 20), (50, 50) on ESC-50, UrbanSound8K, TAU. The TAU hull has only MN10-Q8, MN20-Q8, AST-Q4, always-average and MN10+MN20 (no MN04, MN05, DyMN04 there); with costs fixed per clip, S1's hull on TAU is the straight line from MN10-Q8 to MN10+MN20, which is random routing at S1's rate. UrbanSound8K uses the padded MN10 so all five CNNs read the same audio (T5 used the unpadded one).

## Part C detail

d3 field mix (5 fold models, seeds 0 to 2, 947 test clips, NB ratio 0.066), budgets S1 20%, S2 (50%, 20%), every scaler refit:

| System | F1 | Cost (T) | MACs (G) | Escalated |
|---|---|---|---|---|
| MN10-Q8 | 64.72 | 0.031 | 0.25 | |
| MN20-Q8 | 66.86 | 0.081 | 0.94 | |
| MN05+MN10 | 68.65 | 0.049 | 0.34 | |
| S1 | 69.24 | 0.057 | 0.55 | 36% |
| MN10+MN20 | 70.25 | 0.112 | 1.19 | |
| S2 | 72.59 | 0.951 | 18.2 | 68% (AST 39%) |
| T24's cascade (MN10-Q8 to AST-Q4, 30%) | 70.05 | 1.190 | 23.3 | 50% |
| AST-Q4 | 66.12 | 2.664 | 52.7 | |
| always-average | 70.23 | 2.695 | 52.9 | |

S2 beats the T24 cascade by +2.54 (NB 1.72 to 3.37, 5 of 5) at 80% of its cost. The T24 cascade against the widened hull is -0.21 (NB -0.83 to 0.42).

Caveat on S2: at S2's cost the d3 hull point is the MN10+MN20 static, because always-average (70.23) sits below the pair (70.25). The widened set has no three-model average (T25 left triples out), so S2's gain mixes AST's extra information with routing. Section 14 did not list S2 against an MN10+MN20+AST always-average; it was not computed here and is the recommended next check.

d4 (precondition failed; descriptive only, partial hull of MN10-Q8, MN20-Q8, AST-Q4, always-average, MN10+MN20, `tables/c/d4/descriptive_partial`): field S1 56.90 at 0.166 T, +0.95 over the partial hull (NB 0.43 to 1.46, 5 of 5), which is the straight line from MN10-Q8 to MN10+MN20 (random routing at S1's rate); S1 minus MN10+MN20 (57.06 at 0.246 T) -0.16; S2 58.07 at 4.00 T, +0.96 (0.77 to 1.14).

## Costs (priced bit-ops per clip, T; field mix, clean in brackets)

| System | FSDKaggle2018 | FSDnoisy18k | TUT 2017 |
|---|---|---|---|
| S1 | 0.065 (0.043) | 0.057 (0.046) | 0.166 (0.142) |
| S2 | 1.239 (0.478) | 0.951 (0.554) | 4.004 (2.998) |
| MN10+MN20 average | 0.110 | 0.112 | 0.246 |
| T21 cascade (T24's on d3, d4) | 1.154 (0.435) | 1.190 (0.737) | 2.837 (2.029) |

S1 with one shared front end (descriptive): 0.061 (0.042), 0.054 (0.044), 0.154 (0.133). MACs per clip are in the `by_seed_fold.csv` files. These are bit-op proxies from the cost model, not measured latency; every tier is W8A8 or w4g128 as in T21.

## Commands (worktree, `R` = this folder, `T` = scripts/redesign/moe/t26)

```
python T/t26_part_a.py --r $R; python T/t26_recompute.py --r $R --part a
python T/t26_part_b.py --r $R
R=$R D=d3 SEEDS="0 1 2" MODELS=mn20 TAG=pre bash T/t26_cpu_chain.sh      # extraction, checks, heads, scores, precondition, recompute
R=$R D=d4 SEEDS="0 1 2 3 4" MODELS=mn20 TAG=pre bash T/t26_cpu_chain.sh
R=$R D=d3 SEEDS="0 1 2" MODELS="mn04 mn05 dymn04" TAG=wide bash T/t26_cpu_chain.sh
python T/t26_part_c.py --r $R --d d3 --seeds 0 1 2; python T/t26_recompute.py --r $R --part c --d d3 --seeds 0 1 2
python T/t26_part_c.py --r $R --d d4 --seeds 0 1 2 3 4 --partial; python T/t26_part_c.py --r $R --combine d3 d4
python T/t26_manifest.py $R 6ecea17
```

## Deviations

- Code checks before the plan commit, all declared in the plan: T25 reproduction (`--check-only`), smoke runs with permuted labels, Part C dry runs with T24's MN10 files standing in for the new CNNs, a 4-clip extractor check. Every scratch output was deleted.
- Plan C10 said two CPU chains at once. To shorten the wait, a third CPU process ran beside them twice, with the same script and arguments: d4 eval cells white 0 dB, white 10 dB and pink 0 dB from 09:48 (the d4 chain then skipped those finished cells), and d3 stage 2's eval cells from 13:10 (the chain skipped them). Every process ran at nice 10 with 2 threads, the GPU idle. No result depends on which process wrote a file; the embedding checks passed (22 and 22 at stage 1, 55 at stage 2).
- The GPU was not used. Swap use stayed at about 10.4 GB, above section 2's 9.5 GB ceiling for a GPU process.
- The recompute picks S1's budget with a linear inner-validation hull and chose 50% on d3 where the analysis chose 20%; both fail H1-C (above). `tables/c/d3/recompute_at_analysis_budget.json` is the recompute with S1 forced to 20%, run after the verdict as a check by `t26_recompute_fixed_budget.py` (committed after the verdict); it gives +0.385, p 0.433.
- Another process on the machine took free disk from 45 GB to 35 GB during the run; T26 wrote about 3 GB. The guard never fired.
- d3's test clips are almost all FSD50K clips (T24's overlap check); no model trains on FSD50K. d4 is acoustic scenes, the task type of TAU.

## Files

`ANALYSIS_PLAN.md`; `tables/a` (Part A), `tables/b` (Part B), `tables/c/<d>` (precondition, verdict, by condition, by seed and fold, budget selection, scaler choice, recompute), `tables/c/verdict_combined.json`; `csv/<d>` (T24's all.csv and eval.csv, head CSVs), `emb/<d>`, `outputs/` (heads), `scores/<d>`, `logs/`, `scripts/` (copy of `scripts/redesign/moe/t26` at the final commit), `run_manifest.json`.
