# T26 analysis plan: the mixture with the right pair of experts (MN10-Q8 then MN20-Q8)

Written 2026-10-29 (Irish time) by the T26 agent and committed on branch `moe/t26` (worktree `~/Documents/qmoe-work/t26`, from `moe/t25` at `23175a9`, with `moe/t24` at `4631bc9` merged in as `be5431f` so that T24's fixed-test head option, scorer and analysis helpers are on the branch) before any T26 number exists. The plan and every analysis script (`scripts/redesign/moe/t26/`) are in this one commit. The results folder is `~/Documents/qmoe-results/2026-10-29_<HHMM>_moe-t26_code-<this commit>/`.

What has been seen before this commit: the published results of T21 (FSDKaggle2018 and FSD50K), T24 (FSDnoisy18k and TUT 2017: MN10-Q8, AST-Q4, always-average, the T21-design cascade and the narrow hull, the precondition values), T25 (FSDKaggle2018: every small CNN at 8 bits, every two-CNN average and the widened hull at the T21 cascade's cost, including MN20-Q8's FSDKaggle2018 scores and the MN10+MN20 average), stage 3c and T5 (development-data statics, including MN10-Q8 and MN20-Q8 on ESC-50, UrbanSound8K and TAU). Part A is therefore post hoc. What has not been seen: any S1 or S2 number on any dataset, and any MN20-Q8 (or MN04, MN05, DyMN04) embedding, head or score on FSDnoisy18k or TUT 2017.

Code checks run before this commit, none of which produced a T26 number:
1. `t26_part_a.py --check-only` rebuilt every static of the widened set and the T21 cascade on FSDKaggle2018 from the stored score files and reproduced T25's `t25_by_seed_fold.csv` (3,600 rows, worst F1 difference 1.4e-14, worst relative cost difference 2.2e-16) and T25's H1-wide and H2-wide numbers (difference 0.0). These are T25's published numbers; no S1 or S2 prediction was made.
2. `t26_part_a.py --smoke` and `t26_part_b.py --smoke` ran the S1 and S2 code paths with every label replaced by a fixed random permutation (chance-level F1 about 2 to 10 points), into a scratch folder that was then deleted.
3. `t26_part_c.py` (precondition and full modes), `t26_recompute.py --part c` and the combine step ran on a scratch folder where T24's own MN10-Q8 score files of FSDnoisy18k stood in for the four new CNNs (so MN20 minus MN10 is 0.0 by construction; it reproduced T24's MN10, AST, always-average and cascade rows to 7e-15). The scratch folder was deleted.
4. A 4-clip extractor check: `t26_extract.py` on the first 4 rows of TUT 2017's eval.csv, MN10 and MN20 at W8A8, clean and bg 10 dB, into a scratch folder that was deleted. Embeddings only (no head, logit or F1). Clip lengths and the background plan equalled T24's; MN10's W8A8 embedding differed from T24's stored one by at most 1.6% of the largest value, because the dynamic 8-bit activation scales depend on the batch (4 clips here, 128 in T24), the effect T25's plan documents. T24's stored MN10-Q8 scores are used throughout; MN10 is never re-extracted.

Amendments are new commits with a date, what changed and what had been seen by then.

## A. Section 14 of the round plan, verbatim

Source: `mit-projects/36-qmoe/blindspot-reviews/reviews/review-fable5-1-max-effort.md` (coordinator, last commit `c28ae11`, 2026-10-29 08:40 UTC).

## 14. T26: the mixture with the right pair of experts (owner's go, 2026-10-29)

T25 showed that MN10-Q8 and MN20-Q8 averaged come within 0.56 points of the T21 cascade at 9.5% of its cost. T26 asks whether routing by the cheap expert's own uncertainty still pays when the second expert is MN20-Q8 instead of AST-Q4. Part A is post hoc, because MN20-Q8's FSDKaggle2018 scores were seen in T25; Parts B and C are written before their data are scored. This section was committed before any T26 number.

| Item | Exact requirement |
|---|---|
| Systems | S1: MN10-Q8, then MN20-Q8 on low-margin clips, average rule (the T21 design with MN20-Q8 in place of AST-Q4). S2: three tiers, MN10-Q8, then MN20-Q8 averaged, then AST-Q4 on clips still low-margin after the pair (two budgets, chosen jointly on clean inner validation from {5, 10, 20, 30, 50%} for each). Budgets and scalers on clean inner validation only, T21's folds, seeds and mixer. |
| Comparators | The widened T25 hull (every small CNN at 8 bits, every two-CNN average, AST-Q4, always-average), the MN10+MN20 average alone, the T21 cascade. |
| Part A (post hoc, FSDKaggle2018, stored T21 and T25 scores) | S1 and S2 minus the widened hull, on the field mix and clean, Nadeau-Bengio. Labelled post hoc everywhere it is reported. |
| Part B (descriptive, development data) | The same on ESC-50, UrbanSound8K and TAU from stored stage 3c and T5 scores where MN10-Q8 and MN20-Q8 both exist with matching folds; skipped and said so where they do not. |
| Part C (pre-registered, fresh data) | FSDnoisy18k and TUT Acoustic Scenes 2017 (T24's folds, heads recipe and mixer). Extract MN20-Q8 and the other small CNNs needed for the widened hull there. Precondition: MN20-Q8 beats MN10-Q8 by at least 1.0 on the field mix (otherwise no headroom; report and stop for that dataset). H1-C: S1 minus the widened hull at least 1.0 on the field mix, Nadeau-Bengio, Holm over the datasets that pass the precondition. S2 is secondary. Family transparency number 22. |
| What the paper may claim | Part C pass: a two-expert quantized mixture of small CNNs routed by uncertainty beats every equal-cost alternative on fresh data. Part C fail or stop: the paper reports Part A and B as post hoc and descriptive, and the T25 finding stands as the cost result. |
| Audit | A Part C pass gets an independent audit agent before it is called Accepted; a fail needs the agent's own recompute. |

## B. Section 2 of the round plan (rules for every slice), verbatim

## 2. Rules for every slice (copy into every agent prompt)

| Rule | Exact requirement |
|---|---|
| Code location | The single code folder is `/Users/cajas.sebastian/Desktop/repositories/curious-qmoe` (branch `paper/quantized-gate`, with `moe/*` branches). Never work on the Desktop. Create a worktree outside it: `git -C /Users/cajas.sebastian/Desktop/repositories/curious-qmoe worktree add -b moe/t19 ~/Documents/qmoe-work/t19 moe/t13` (same for t20, t21). Archive worktrees to `~/Documents/qmoe-archive/worktrees/` when done; never delete branches. |
| Git identity | Commit as `Sebastian Cajas <sebasticajas@gmail.com>`. No Co-Authored-By line of any kind. Never push. |
| Documents | Do not edit Overleaf (`curious-moe-overlead`), `index.md`, `master-table.md`, `BACKLOG.md` or any `paper-versions` folder. The coordinator logs results. |
| Pre-registration | Write `ANALYSIS_PLAN.md` in the results folder and commit it on the branch BEFORE any extraction that produces test-fold scores and before any test statistic. Amendments are new commits, dated, listing what changed and what had already been seen. Every rule in sections 4 to 6 goes into the plan verbatim. |
| Family | Campaign family for T19 and T20 is 17 (T1, T1b, T1c, T2b, T7, T9, T12, T13, T14, T15, T16, T17, T18, T3, T5, T19, T20). Report Holm within the experiment and the family-of-17 number next to it. T21 is a confirmatory single test per dataset on unseen data; its claim rests on that single test (alpha 0.05 two-sided, Holm over its own datasets); report the family-of-17 number alongside for transparency. |
| Protocol | Official folds (ESC-50 5, UrbanSound8K 10, TAU 5 location-grouped over device A); inner validation grouped by source recording (ESC-50: second token of the file name; UrbanSound8K: fsID; TAU: location); thresholds and budgets chosen on clean inner validation only; heads trained on clean training clips with the protocol 3 recipe (`scripts/benchmark.py`); macro F1; seeds ESC-50 0 to 4, UrbanSound8K 0 to 2, TAU 0 to 2. |
| Comparators | The hull is the time-sharing hull of every static system available at that cost plus the always-average point(s); the best single model is chosen on inner validation of the same condition family (field mix, noise, shift), never on clean validation. Random time-sharing and a random gate at the same escalation count are reported as well. |
| Statistics | Fold level (seeds averaged within folds), paired t with the Nadeau-Bengio correction, sign-flip floor stated (0.0625 with 5 folds), bootstrap 95% intervals, Holm over the tasks of the primary test. No pair-level p values in the verdict. |
| Costs | Priced bit-ops: MACs times weight bits times activation bits, group and channel scales priced at 32 x 32 per rescale MAC, mel and fbank front ends included (`QWave/costs.py`, idea 5b `common.py`, stage 3c cost tables). Report the unpriced view as secondary. Measured CPU latency where a real kernel exists; say plainly that no int4 activation kernel exists on this Mac, so A4 costs are a bit-op proxy. |
| Noise and shift | Stage 2's exact mixing code and plan (`bg_plan.json`) so every system sees the same audio: background (UrbanSound8K clips for ESC-50 and the reverse), pink and white at 20, 10, 5, 0 dB; the field mix; the three non-additive corruptions (reverberation RT60 0.3, 0.6, 1.2 s; clipping 0.5, 0.25, 0.1; low-pass 4, 2, 1 kHz) from `2026-10-24_1507_redesign-idea6-final_code-ed82400`; TAU device shift. |
| GPU | Apple M2 Pro, 32 GB. One MPS process at a time, enforced by the lock: `mkdir ~/Documents/qmoe-work/.mps-lock` (atomic). If it fails, another agent holds the GPU: retry every 60 s inside Bash calls of at most 9 minutes. Write your agent name and PID into the lock directory, release with `rmdir` in a shell trap on every exit, and release between processes so others can interleave. Fixed input shapes (one graph). Check `sysctl vm.swapusage` before each process and wait if used exceeds 9.5 GB. CPU jobs at `nice -n 10` with `OMP_NUM_THREADS=2`, one at a time while the GPU is busy. |
| Results | `~/Documents/qmoe-results/<YYYY-MM-DD_HHMM>_moe-t19_code-<commit>/` with `run_manifest.json` (commit, machine, software versions, dataset sha256), `ANALYSIS_PLAN.md`, `README.md` (provenance, commands, verdict per rule, deviations), `scripts/`, `logs/`, `tables/`. Big caches may live beside it but are not copied. |
| Writing | Plain English, no em dashes, no "not X but Y", no colon titles, every number traceable to a file. |
| Audit | A PASS is not "Accepted" until an independent auditor agent (own code, read-only) reproduces the numbers, checks leakage and costs, and recomputes the hull. The coordinator launches it. |



Section 11 adds, for the extension rounds this one follows: GPU access only through `~/Documents/qmoe-work/fair_gpu_run.sh <tag> <command>`; free disk above 25 GB (stop and report below it). The coordinator's T26 prompt adds: proceed only while `memory_pressure` shows free at or above 25%; commits local only; recompute any fail from the raw score files with a separate script before reporting it; if H1-C passes, say so and stop (the coordinator launches the audit). The campaign family for the transparency number is 22.

## C. Choices fixed here, before any T26 number

### C1. The two systems (exact)

Softmax probabilities of each tier's FP32 head (logits under the tier's chosen scaler); margin = top-1 minus top-2 probability; "average" = the arithmetic mean of the probability vectors, prediction its argmax.

- **S1** (T21's design with MN20-Q8 in place of AST-Q4): every clip runs MN10-Q8. A clip whose MN10 margin is below the threshold t runs MN20-Q8 and takes the mean of the two; every other clip keeps MN10-Q8's prediction. t per (seed, fold) = stage 2's `thr_rate` on the fold model's clean inner-validation MN10 margins at the chosen rate.
- **S2** (three tiers, T23's `thr_b` and `system_b` with MN20-Q8 as the middle tier): tier 2 as S1 at rate r2 (threshold t2); among the tier-2 clips, those whose margin of the MN10+MN20 mean is below t3 run AST-Q4 and take the mean of the three probability vectors. t3 = `thr_rate` on the pair-mean margins of the clean inner-validation clips that reached tier 2, at rate min(1, r3 x N / N2), so that r3 of all inner-validation clips reach AST-Q4; when r3 is at or above r2 every tier-2 clip runs AST-Q4 (those pairs stay in the grid). AST-Q4 is T21's (Part A) or T24's (Part C) w4g128 at the clips' real length; on the development data stage 2's w4g128 at 512 frames (ESC-50) and 400 frames (UrbanSound8K) and option B's w4g128 at 1024 frames (TAU, 10 s clips).
- Code: `t26_core.py` (`s1_pred`, `s2_thr`, `s2_pred`).

### C2. Budgets and scalers (clean inner validation only)

- S1 budget from {5, 10, 20, 30, 50%}. **S2 budgets: the 25 pairs (r2, r3) with r2 and r3 each in {5, 10, 20, 30, 50%}, chosen jointly**, r3 a share of all clips as in C1.
- Rule (T21's, with T23's member set): for each candidate, the system's mean clean inner-validation macro F1 over (seed, fold) minus the inner-validation hull at the system's mean inner-validation cost, where the inner-validation hull is stage 2's `hull_pair` over the widened static set of C3 (each static's mean clean inner-validation F1 and mean per-clip priced cost over (seed, fold)) and its F1 per (seed, fold) is the Monte Carlo time-sharing F1 (stage 2's `DRAWS` = 200 draws, generator seeded with `RNG_IV` = 12346, started fresh for S1 and again for S2, candidates in the order listed). The candidate with the largest gain wins; ties go to the lower inner-validation cost. The budget is never re-chosen on test data or on a shifted condition.
- Scalers: MN10-Q8 and AST-Q4 keep the scaler T21 chose (Part A, refit and refit) or T24 chose (Part C, refit and refit on both datasets). MN04, MN05, MN20, DyMN04 on FSDKaggle2018 keep T25's choice (refit for all four). On FSDnoisy18k and TUT 2017 each new tier gets T21's rule: `refit` if its mean clean inner-validation macro F1 over folds and seeds is strictly higher than `fp32fit`, else `fp32fit`. The development-data score files hold one scaler per head; nothing is chosen there.

### C3. Comparators and the widened hull (T25's code)

- **Widened static set** (T25, 17 systems): MN10-Q8, AST-Q4, always-average (MN10-Q8 with AST-Q4), MN04-Q8, MN05-Q8, MN20-Q8, DyMN04-Q8, and the ten two-CNN averages of {MN04, MN05, MN10, MN20, DyMN04}. On TAU only MN10-Q8, MN20-Q8, AST-Q4, always-average and MN10+MN20 exist (C6).
- **Hull at each system's own cost**: per condition, the mean test macro F1 and mean priced cost of each static over the (seed, fold) cells; stage 2's `hull_pair` at the system's mean test cost on that condition (the best static at or below that cost, or the best straight line between a static below and one above); per (seed, fold) the hull's F1 is the exact expected macro F1 of random time-sharing between the two endpoints at that weight (T21's `exp_f1`: `mix_expected_f1` when every class is in the test fold, else 4,000 Monte Carlo draws). This is T25's `hull_of` with the cost argument set to S1's, S2's or the cascade's own cost (`t26_core.compare_all`).
- Also reported, each at its own cost: the MN10+MN20 average alone; the T21 cascade (MN10-Q8 then AST-Q4, average rule) with its own pre-chosen budget (Part A: T21's 20%, thresholds rebuilt from the stored clean inner-validation logits; Part C: T24's 30% on FSDnoisy18k and 20% on TUT 2017); S2 against S1. No T21-design cascade is built on the development data (no budget was chosen for it there under this design).

### C4. Pricing

Priced bit-ops per clip at the clip's own length, averaged over the clips scored, exactly T25's C4 and T21's C5: each CNN = EfficientAT front end at 32 x 32 + W8A8 backbone with one rescale per output element at 32 x 32 (DyMN04 adds its kernel aggregation at 32 x 32) + the FP32 head; MN10-Q8 and AST-Q4 from T21's (Part A) and T24's (Part C) `per_clip_costs`; a two-CNN average is the sum of both CNNs' full costs (front end counted twice); always-average is MN10-Q8 plus AST-Q4.
- S1 per clip: MN10-Q8 + (escalated) x MN20-Q8's full cost, front end counted again (T25's pair pricing).
- S2 per clip: MN10-Q8 + (tier 2) x MN20-Q8 + (tier 3) x AST-Q4.
- Secondary: the unpriced view (MACs per clip) for every system; S1's priced cost with one shared front end (escalated clips do not pay MN20's front end), cost only, not a test.
- Development data: the stored mean per-clip priced costs (stage 3c `analysis/<task>/costs.json`, option B `analysis/tau/costs.json`); a pair is the sum; no MACs view there.

### C5. Part A (FSDKaggle2018, POST HOC)

T21's D1 pool design: 5,310 manually verified clips, 5 stratified outer folds, head seeds 0 to 4. Score files: T21 `scores/d1/pool_mn10_w8a8_s*.npz`, `pool_ast_w4g128_s*.npz`; T25 `scores/d1/pool_{mn04,mn05,mn20,dymn04}_w8a8_s*.npz`. Conditions: clean, bg/pink/white at 10 and 0 dB, the field mix (T21's mixer, test seed 5001, validation seed 5002). Before any S1 or S2 number the script reproduces T25's per-(condition, seed, fold) table and T25's H1-wide and H2-wide to 1e-9 and stops otherwise. Reported: S1 minus the widened hull and S2 minus the widened hull on the field mix and on clean audio, with the Nadeau-Bengio 95% interval, p, the sign-flip p, the clip-level bootstrap interval (T21's, 2,000 draws per fold) and folds positive; S1 and S2 against the MN10+MN20 average and the T21 cascade; costs. Labelled post hoc everywhere; no pass or fail is attached. Code: `t26_part_a.py`.

### C6. Part B (development data, DESCRIPTIVE)

Stored score files with matching rows, folds and inner-validation rows (asserted per seed and fold):

| Task | Folds, seeds | MN04-Q8 | MN10-Q8 | MN05-Q8, MN20-Q8, DyMN04-Q8 | AST-Q4 | Beds |
|---|---|---|---|---|---|---|
| ESC-50 (50 classes) | 5, 0 to 4 | stage 3b | stage 3b | stage 3c | stage 2 `astshort` w4g128 at 512 frames | clean, field mix |
| UrbanSound8K | 10, 0 to 2 | stage 3b | stage 3b (padded 4.04 s, the input stage 3c's MN05, MN20 and DyMN04 read) | stage 3c | stage 2 `astshort` w4g128 at 400 frames | clean, field mix |
| TAU 2020 Mobile | 5 location-grouped, 0 to 2 | none | option B | MN20 option B; MN05 and DyMN04 none | option B w4g128 (1024 frames) | device A, other devices (shift) |

T5 and T13 paired the unpadded MN10 (`stage3c-unpadded`) with the padded MN20 on UrbanSound8K; T26 uses the padded MN10 so that all five CNNs read the same audio. TAU has no MN04, MN05 or DyMN04 and no field mix, so its hull is the five statics listed in C3 and its shifted bed is the device shift; the stored TAU heads were trained on device A. Budgets as C2 on clean inner validation. Nadeau-Bengio ratio n_test / n_train per fold, averaged (TAU: device A rows). No bootstrap, no test is attached to any number. Code: `t26_part_b.py`.

### C7. Part C (FSDnoisy18k d3 and TUT Acoustic Scenes 2017 d4, PRE-REGISTERED)

- **Design**: T24's unchanged. all.csv and eval.csv copied from T24's results folder (checksummed in the manifest); the five fold models of the fixed-test design (`cross_validation.fixed_test=true`), one official test set (d3: 947 clips; d4: 1,620 clips); inner validation on clean-label rows of the held-out fold; T24's reader of the prepared 16 kHz audio; stage 2's mixer and field mix (test seed 5001, validation seed 5002) over eval.csv rows; T24's MN10-Q8 and AST-Q4 score files reused unchanged, its scalers and its cascade budget. Seeds: d3 0 to 2, d4 0 to 4 (T24's). NB ratio: T24's per-fold n_test / (n_fit + n_iv), averaged (0.066 and 0.346), asserted equal to T24's.
- **Extraction** (`t26_extract.py` = `t25_extract_cnn.py` with `t24_common.patch_loader`): on all.csv clean at FP32 (head training) and W8A8 (inner validation, clean test, refit scaler); on eval.csv at W8A8 under bg, pink and white at 10 and 0 dB plus the 5 dB cells for the rows either field assignment needs. CPU, nice 10, 2 threads, chunks of 128 clips, each clip at its own length. Checks before any score (`t26_check_emb.py`): names checksum and row count equal to the CSVs, finite on every needed row, clip lengths equal to T24's MN10 lengths in every condition, background plan equal to T24's.
- **Heads and scores**: T24's `t24_train_heads.sh` call unchanged (protocol 3, fixed test, CPU) through `t26_train_heads.sh` (worktree path only); T24's `t24_score.py` unchanged with the tier table widened (`t26_score.py`): both scalers, every condition, clean and field inner-validation logits, every fold model must reproduce its summary.json F1 first.
- **Staged order (declared here)**: stage 1 extracts, heads and scores MN20-Q8 only on both datasets and evaluates the precondition. Stage 2 extracts MN04, MN05 and DyMN04 only on a dataset whose precondition passed, then runs H1-C there. A dataset that fails the precondition is reported and stopped: no stage 2 there; a descriptive S1 and S2 against the partial hull it has (MN10-Q8, MN20-Q8, AST-Q4, always-average, MN10+MN20) may be reported, labelled descriptive, never a test.
- **Precondition**: MN20-Q8 minus MN10-Q8 on the field mix of the test set is at least 1.0 macro F1 point, read as the mean over the five fold models with seeds averaged within each fold model (as T24 read its precondition); every fold model's value and the count at or above 1.0 are reported. The analysis rebuilds T24's MN10, AST, always-average and cascade rows to 1e-9 first. Code: `t26_part_c.py --precondition`.
- **H1-C (primary)**: on the field mix, S1 minus the widened equal-cost hull (C3) is at least 1.0 macro F1, fold-level paired t with the Nadeau-Bengio correction, two-sided, alpha 0.05, Holm over the datasets that pass the precondition. PASS on a dataset when the mean difference is at least 1.0 and its Holm p is below 0.05. Reported beside it: the NB 95% interval, the clip-level bootstrap interval (T24's, the same resampled test clips for every fold and seed, 2,000 draws), the sign-flip p and its floor 0.0625, folds positive, and the family-of-22 number min(1, 22 x raw p).
- **Secondary (not in Holm)**: S2 minus the widened hull on the field mix; both systems on clean audio; S1 and S2 against the MN10+MN20 average and against T24's cascade at their own costs; which statics form the hull; costs.
- If no dataset passes the precondition, H1-C is not tested and Part C is reported as stopped.

### C8. Statistics (every part)

Fold level (seeds averaged within folds or fold models), paired t with the Nadeau-Bengio correction (stage 2's `tests`), interval from T21's `nb_ci`, sign-flip p with its floor, two-sided. No clip-level or pair-level p in any verdict. Holm only over Part C's datasets that pass the precondition. Family number 22 for transparency.

### C9. Own recompute and audit

`t26_recompute.py` imports none of the analysis helpers (own softmax, margin, threshold, scaler rule, sklearn macro F1, own budget choice with a linear inner-validation hull, own hull over the widened statics with Monte Carlo time-sharing at 2,000 draws, own Nadeau-Bengio test); it shares only the per-clip cost arrays. It recomputes Part A (S1 and S2 minus the hull, field and clean), Part C's precondition, and Part C's S1 and S2 minus the hull (field and clean) from the raw score files. It runs whatever the verdict; a fail is reported only after it agrees. If H1-C passes on any dataset, the agent reports the pass and stops; the coordinator launches the independent audit. The linear inner-validation hull may pick a different budget on a near tie; any difference is reported.

### C10. Machine, order, disk

Order: this commit; Parts A and B (CPU, stored scores); Part C stage 1 on both datasets; stage 2 only where the precondition passes; recompute; README and manifest. Part C runs on the CPU only, in two chains at once (d3 and d4, each one process at a time, nice 10, 2 threads) while the GPU is idle, as T24 ran two CPU jobs with the GPU idle; no MPS process is used (FSDnoisy18k has many clip lengths, which compile one graph each on MPS, and swap use was 10.4 GB at plan time, above section 2's 9.5 GB ceiling for a GPU process). Every chain step checks free disk above 25 GB (stop below it) and `memory_pressure` free at or above 25% (wait below it). Part C writes about 2 to 4 GB (embeddings, head CSVs, scores); no audio is written.
