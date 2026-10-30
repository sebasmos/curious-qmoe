# T25, the fixed T21 cascade against a wider equal-cost hull (FSDKaggle2018)

Branch `moe/t25` of `curious-qmoe` (worktree `~/Documents/qmoe-work/t25`, from `moe/t21` at `7957360`), never pushed. Plan and every script committed at `cc3b048` (22:47, 28 Oct) before any T25 embedding was kept on disk or scored (a 40-clip extractor check ran before it, see deviations); manifest script `9d4f0ae`. `ANALYSIS_PLAN.md` here equals the committed plan (diffed). Numbers come from `tables/d1/verdict.json`, `tables/d1/t25_by_condition.csv`, `tables/d1/t25_statics_by_condition.csv` and `tables/d1/recompute.json`.

## Design

T21's cascade unchanged: MN10-Q8 then AST-Q4, margin gate, average rule, budget 20%, scalers refit and refit, all read from T21's stored scores and tables. Added to the hull: MN04-Q8, MN05-Q8, MN20-Q8, DyMN04-Q8 (stage 3c loaders, simulated W8A8, protocol 3 FP32 heads, T21's folds, seeds 0 to 4, scaler rule, mixer), and the 10 pairwise averages of the five CNNs, each pair priced as the sum of both CNNs' full costs. Hull: T21's code with 17 systems. Every new tier chose `refit`.

## Checks

- T21's cascade, MN10-Q8, AST-Q4 and always-average rebuilt on every (condition, seed, fold): worst difference 1.4e-14. T21's narrow-hull gains reproduced (+2.83 field, +2.07 clean).
- 55 embedding checks pass (`logs/check_emb_d1.json`): names, rows, finite, clip lengths and background plan equal to T21's. Every head reproduced its summary.json F1 when scored.
- Separate recompute (`scripts/t25_recompute.py`, own F1, hull, Monte Carlo time-sharing and test): field +0.268 against +0.268, clean +0.002 against +0.001, same hull, same scalers.

## Verdict

| Rule | Result | Verdict |
|---|---|---|
| H1-wide: field mix, cascade minus widened hull at least 1.0, NB, Holm over datasets (D1 only) | +0.27, NB 95% -1.08 to 1.62, p 0.61 (Holm 0.61, family of 21: 1.0), bootstrap -0.12 to 0.64, 3 of 5 folds positive (fold values 1.25, -0.05, 0.15, 0.66, -0.67), sign-flip p 0.50 | FAIL |
| H2-wide: clean | +0.00, NB 95% -0.56 to 0.56, p 0.99, 1 of 5 folds | FAIL |
| Widened hull at the cascade's cost | MN10+MN20 average to always-average, weight 0.41 on the field mix (0.13 clean); hull F1 77.30 field, 93.66 clean | |
| Best two-CNN average (chosen on inner validation, field and clean): MN10+MN20 at its own cost | field 77.01 at 0.110 T against the cascade 77.57 at 1.154 T (cascade +0.56, NB -1.70 to 2.82, p 0.53, 4 of 5 folds); clean 93.64 against 93.66 at 0.435 T (+0.03, p 0.91) | descriptive secondary |

The pair costs 9.5% of the cascade on the field mix and comes within 0.56 points of it. Every point on the line from MN10+MN20 to always-average lies close to the cascade, so routing adds about 0.3 points over blind time-sharing once two small CNNs are averaged.

Descriptive sensitivities, not tests: pairs priced with one shared front end, field +0.27 (same hull); unpriced MACs view, field +0.26.

Second dataset: T24 finished with both datasets failing their precondition (FSDnoisy18k +1.40, TUT 2017 -0.64), so the Holm set is FSDKaggle2018 alone. No descriptive wider-hull rows were added there: T24 stored only MN10 and AST scores, and adding the four CNNs would mean extracting them on every clip of both datasets.

The H1-wide pass flag is in `tables/verdict_combined.json` (Holm over the datasets run); `tables/d1/verdict.json` holds the per-dataset numbers.

## Commands (worktree, `R` = this folder)

```
R=$R bash scripts/redesign/moe/t25/t25_cpu_chain.sh        # extract (CPU), check, CSVs, heads, scores
python scripts/redesign/moe/t25/t25_analyze.py --r $R --d d1; python .../t25_analyze.py --r $R --combine d1
python scripts/redesign/moe/t25/t25_recompute.py --r $R --d d1; python .../t25_manifest.py $R
```

## Deviations

- Before the plan commit, a 40-clip code check of the extractor ran on the first 40 pool rows (clean and bg 10 dB, MN10, MN04, DyMN04) into a scratch folder that was then deleted; no head, logit or F1 was computed. The plan states this.
- The plan header says "written 22:50"; the commit is at 22:47. The time was an estimate written before the commit.
- Before scoring, both analysis scripts were dry-run on a scratch folder where T21's own MN10 score files stood in for the four new tiers (code debugging only; no T25 embedding, logit or F1 existed then). The scratch folder was deleted.
- The CNN extraction ran on the CPU in one process at nice 10 with 2 threads. Other projects' jobs shared the CPU, so it took about 4 hours (22:47 to 02:59). No GPU was used.
- No run on a second dataset (above).
