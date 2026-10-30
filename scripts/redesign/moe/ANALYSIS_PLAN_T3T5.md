# T3 (curiosity) and T5 (humility): the cheap model's margin as a predictor and as an abstain signal, on stored arrays

Written and committed on 25 Oct 2026 (branch `moe/t3t5`, from `moe/t13`, worktree `~/Documents/qmoe-work/t3t5`) before any test statistic of T3 or T5 exists. What exists at this commit:
the T13 loaders (`t13_common.py`) and nothing else new. The input check (shapes, row-order checksums, finiteness; no accuracy), the pre-selection of the best single model
(inner-validation clips only) and the scoring script are committed after this file, in that order, and the scoring script is committed before it is run.
Machine: CPU, `nice -n 10`, 2 threads (OMP, MKL and OpenBLAS set to 2). Nothing is extracted from audio. The only fitting is the small routers, regressions and noise detectors
described below, on stored embeddings, stored logits and stored audio statistics.

## 0. What is already known, and what this plan claims

Both experiments are CONFIRMATORY ON SEEN DATA. The headline numbers were visible before this plan was written and are not new:

| Known quantity | Value | Source |
|---|---|---|
| Cheap margin (mn10 W8A8) predicts the AST's errors, UrbanSound8K | AUROC 0.86 (training folds) and 0.84 (test folds), 10 of 10 folds each | `2026-10-24_2013_moe-difficulty-pilot_code-b77b98e/README.md`, `t1a/` |
| Mixture margin (pair) predicts the mixture's own errors, shifted pools | AUROC 0.865 (ESC-50), 0.810 (UrbanSound8K), 0.717 (TAU shift) | `2026-10-25_0145_moe-t13_code-eaaedc9/README.md`, epistemic table |
| Jensen-Shannon disagreement adds at most +0.0045 AUROC to the mixture margin; abstaining on JS is worse than on the margin | risk-coverage area JS minus margin +1.8 to +4.7 points | same README |
| Margin detects unseen classes, cheap tier | AUROC 0.941 (ESC-50 task, mn20) and 0.862 (UrbanSound8K task, mn10); noise_lr 0.730 and 0.406 | `2026-10-24_1507_redesign-idea6-final_code-ed82400/analysis/e7_ood.csv`, BACKLOG -57 |
| Margin detects the other dataset's clips | 0.690 (mn20), 0.575 (mn10); noise_lr 0.652, 0.366 | same run, `e6_detect.csv` |
| Margin gate beats random and the noise detector at equal deferral count; deferral rises with noise at a clean threshold | idea 7 (22 Oct) and final round part B | `2026-10-22_1105_redesign-idea7-stage1_code-e2af723/README.md`, final README |
| Margin AUROC for the cheap tier's own errors | 0.80 to 0.94 (ESC-50), 0.81 to 0.87 (UrbanSound8K) across conditions | final round `analysis/b3_gate_auroc.csv` |

The margin is the max-softmax family baseline of Hendrycks and Gimpel (2017), here top1 minus top2 of the softmax. The expected result is that it keeps working at low precision.
It is not a new signal, and nothing here is written up as one. What this plan pre-registers is only the comparators, the contrasts, the cross-fitting, the pass and stop rules, and the
multiplicity handling. I also looked at the T13 tables (margin against JS on the mixture, the rise with shift) before designing T5, which is why the T13 pair and thresholds are reused
instead of chosen again. I computed nothing on the T3/T5 targets before this commit.

## 1. Family and multiplicity

The project family is **15** experiments (T1, T1b, T1c, T2b, T7, T9, T12, T13, T14, T15, T16, T17, T18, T3, T5). T3 and T5 each have three task-level primary tests (one per task).
The index rows say "Holm, family of 15", so the decision uses Holm's step-down with m = 15: the three p-values of an experiment sorted ascending are multiplied by 15, 14 and 13,
a running maximum is taken, and the result is capped at 1. This assumes the other 12 slots of the family sit above ours, the conservative case. The T13 convention (Holm inside the
experiment, here m = 3, with a x15 line beside it) is reported next to it in every table. **This is an interpretation of "Holm accordingly"; the verdict is printed under both and the
coordinator can pick the other.** Each task-level p is the p of the weaker contrast (an intersection-union test, as in T13 section 6).

## 2. Data, roles, conditions

Cheap model: mn10 with W8A8 backbone and the FP32 `esc` head (UrbanSound8K: exact-length mn10, T13's choice), the T1a cheap tier. Strong model: the AST of T13's pre-selected pair, read from
`inputs/t13_preselected.json` of the T13 run: `astw4` (AST w4g128, 1024 frames) on ESC-50 and UrbanSound8K, `astfp32` on TAU. The pair is the uniform arithmetic mean of the two softmax
outputs ("mixture"). Heads: one per seed, ESC-50 seeds 0 to 4 (5 folds), UrbanSound8K seeds 0 to 2 (10 folds), TAU seeds 0 to 2 (5 folds). All logits are the stored ones of earlier
rounds, read through `t13_common.py`. Experts of one task and seed share the inner split.

Tasks and primary conditions (as T13): **ESC-50 (50 classes)**, shifted pool = the 12 additive noise cells (bg, pink, white at 20, 10, 5, 0 dB); **UrbanSound8K**, shifted pool = bg 10 dB and
bg 0 dB (the only noise cells with complete AST scores); **TAU shift** = the test rows of every device other than A. Clean (UrbanSound8K and ESC-50 clean clips; TAU device A rows) is a
separate descriptive row with the same machinery. The field mixture and the nine non-additive corruptions have no stored AST at 1024 frames, so they enter only the cheap-model
abstain-rate arm of T5 (section 6). Gaps stay gaps; nothing is extracted. The 5-group ESC-50 task is not run (the brief names three tasks).

Inputs per clip and cell, all stored:
* cheap logits and AST logits (as above), true labels, folds, inner-validation indices and logits (the `iv` entries);
* the 960-d mn10 W8A8 embedding of the same cell: ESC-50 `stage3b/emb_cnn/esc50/mn10_w8a8__<cell>.npy`; UrbanSound8K `stage3c/emb_cnn/urban8k_unpadded/mn10_w8a8__<cell>.npy`
  (exact-length, matching the scored head); TAU `optionB/emb_cnn/tau/mn10_w8a8__clean.npy`. Row order is checked against the score files by the names checksum in the sidecar json;
* idea 7's per-clip audio statistics (`idea7/audio_stats/{esc50,urban}/<cell>.npz`, `optionB/audio_stats/tau/clean.npz`), aligned by `names`.

Reproduction check before any contrast: cheap and AST accuracy of the cells used equal the T13 `cell_fold_acc.csv` values (tolerance 1e-9).

## 3. T3 targets and predictors

Per clip, cell, seed: p_c, p_s the cheap and strong softmax, p_m = (p_c + p_s)/2.
* **A** = AST wrong (strong argmax differs from the label).
* **B** = cheap wrong and AST right (the recoverable errors). **Primary target.**
* **C** = CE(p_c) minus CE(p_m), CE = minus log of the true-class probability; positive when averaging helps (counterfactual gain).

Predictors, each producing a score where higher means "the AST is more likely to help":
1. **margin**: minus the cheap probability margin (top1 minus top2). Needs no AST run, which is the point of the claim.
2. **router (idea 2 comparator)**: an MLP on the standardized 960-d cheap embedding. Primary R1: one hidden layer of 256, ReLU, Adam, L2 alpha 1e-4, batch 256, early stopping
   (10 percent held out, 10 rounds), at most 100 epochs, `random_state` = 1000 + seed. Sensitivity R2 (target B only): hidden 64, alpha 1e-1. No tuning on test. Classifier for A and B,
   regressor (same settings) for C.
3. **disagreement**: Jensen-Shannon divergence between p_c and p_s (natural log, equal weights). It needs the AST output, so it is not available before running the strong model; it is
   the ceiling comparator the claim is tested against.
4. **regression on class-agnostic features** (separate finding, not in the pass): logistic regression (C = 1, standardized) for A and B, ridge (alpha 1) for C, on the features that exist
   in stored arrays: cheap margin, cheap entropy, embedding L2 norm, embedding variance across dimensions, `floor_db` (idea 7's noise-floor estimate: loudest-10-percent-of-frames over
   quietest-10-percent), `level_db` (loudness), `flatness_db` (median spectral flatness). **Seven of the eight requested features.** Clipping fraction is not stored; `crest_db` is
   stored but is a different quantity and is not substituted.

Cross-fitting. **Primary**: for test fold f, fit on the same pool's clips of the other folds (all pool cells stacked), same seed, predict the held-out fold (T13's primary setting; it favours
the router and the regression, so the margin beating them is the conservative direction). Each clip's logits come from the head of its own test fold, so the fitting targets are
out-of-fold. **Secondary (deployable)**: fit on the clean clips of the other folds only, predict the held-out fold's shifted cells (router R1 on target B, regression on A, B and C).
No label of fold f enters any fit that predicts fold f.

Metrics per (seed, fold): AUROC of each score for A and B; for C the Spearman correlation with the score and the AUROC of C > 0. Seeds are averaged inside a fold; the fold is the unit.
AUROCs are computed on the stacked clips of the pool (the stream a deployed system would see).

**Pass rule (target B, per task)**: (a) the mean over folds of AUROC(margin) minus AUROC(R1) is at least +0.05, and the same against AUROC(JS); (b) each contrast is at least
+0.05 in a majority of folds (3 of 5, 6 of 10); (c) Nadeau-Bengio two-sided p of the weaker contrast, adjusted as in section 1, below 0.05. T3 passes if all three tasks pass.
The R2 router contrast and the A and C versions are reported, not decisive. **Regression finding**: the regression's B-AUROC minus the margin's is at least +0.02 on average and in a
majority of folds; reported per task with NB p, outside the family.
**Stop**: the margin loses to the primary router on any task, meaning mean AUROC(margin) minus AUROC(R1) below 0 with the router ahead in a majority of folds. Label order: STOP, else PASS
(three tasks), else NO PASS; the per-task pass count is always printed.

## 4. T5 systems and the best single model

Systems scored on the same clips: the cheap model alone (predictions and margin), the mixture of the pair (predictions and margin), the best single model (below), random abstention, and
the model-free noise detector.

**Best single model**, pre-selected before any test clip: among the five stored experts with priced cost at or below the pair's cost and stored in every cell of the task's shifted pool, the
one with the highest mean **inner-validation** accuracy over the pool's cells (folds, seeds and cells averaged). This is the inner validation of the same condition family, not the clean
validation. For TAU the inner validation holds device A clips only (heads are trained on device A), so the shifted family has no inner validation; the choice falls back to clean inner
validation and says so. Output `inputs/t3t5_preselected.json`, committed before the scoring script.

**Noise detector** (idea 7's `noise_lr`, exactly as `defer.py` fits it): per (seed, fold), a logistic regression (C = 1, standardized inputs) on the ten audio statistics of the fold's
inner-validation clips under the field validation assignment, target "clip is corrupted", never sees a model output. Applied unchanged to the test cells; the score is its predicted
probability, and it abstains on the highest first. For TAU the detector is the option B shift detector, fitted as `analyze_optb.py` fits it (inner-validation device-A rows against the
other devices' rows of those locations, target "not device A"). It was fitted for additive noise; applied to other cells it is a baseline, not a fair detector, and the README says so.

**Random abstention**: expected risk at every coverage equals the error rate, so its area under the risk-coverage curve is the error rate, computed exactly.

## 5. T5 risk against coverage

Risk-coverage curve (Geifman and El-Yaniv): sort the clips by confidence, risk at coverage k/n is the error rate of the k most confident. Area (AURC) = mean over k of the risk, in points of
error. Confidence is the margin (cheap margin for the cheap system, mixture margin for the mixture, the single's own margin for the best single), minus noise_lr for the detector, constant
for random. Computed on the stacked clips of the fold's pool (the mixed stream) per (seed, fold), seeds averaged inside the fold. Also reported: excess AURC (AURC minus the system's own error
rate, which removes accuracy and keeps ranking), selective accuracy at 80 and 90 percent coverage, and error-detection AUROC. Max-softmax (Hendrycks and Gimpel's exact score) is a
sensitivity arm for the cheap and mixture systems. Per-cell AURC averaged over cells is a secondary view.

**Pass rule, per task, on the cheap model**: AURC(margin) lower than AURC(random) and AURC(noise_lr) by a mean fold difference below 0, in a majority of folds for each, NB p of the weaker
contrast, adjusted as in section 1, below 0.05, on all three tasks. The same contrasts on the mixture (mixture margin against random and noise_lr on the mixture's errors) are reported.
**Mixture clause**: the mixture margin "does not lose" to the best single's margin on a task if the mean fold difference of AURC (mixture minus single, points) is at most +0.5 and not
(mean above 0 with NB p below 0.05, Holm over the 3 tasks, m = 3). AURC decides; excess AURC and AUROC are descriptive. The mixture is more accurate than most singles, so its AURC is lower
partly for that reason; the excess AURC column shows the ranking part separately.

## 6. T5 abstain rate under shift

Threshold per system fixed on the clean inner-validation clips of each (seed, fold): abstain when the margin is below the 30th percentile (primary, T13 and idea 7's convention); the 10th percentile is
secondary. The abstain rate is taken on the held-out test clips of each cell. **Monotonicity check, pre-registered, exceptions listed:**
* Additive noise steps, cheap margin: for each task and noise type, rate(step k) minus rate(step k-1) along clean, 20, 10, 5, 0 dB (ESC-50 and UrbanSound8K: bg, pink, white, all cells with stored
  mn10 scores). Mixture margin: the same steps on the cells where the AST is stored (ESC-50 all; UrbanSound8K clean, bg 10, bg 0).
* Device shift, TAU: for each non-reference device d, rate(d) minus rate(device A), same test folds.
* Unit: fold differences, NB two-sided p, Holm over the steps of the (task, system) set. A step is an **exception** if its mean fold difference is not above 0 or its Holm-adjusted p is not below 0.05.
  Raw p are shown beside. **Pass (c)** = no exception among the cheap-margin additive-noise steps and the TAU devices at the 30 percent threshold. Non-additive corruptions (reverberation,
  clipping, low-pass, three severities each, cheap margin only, steps clean to first severity and between severities) are exploratory and their exceptions are listed, since the final round
  already found the reverberation and mild-clipping exceptions on UrbanSound8K.

## 7. T5 unseen-class and other-dataset detection (descriptive, no pass rule)

Reported for the same signals: cheap margin, mixture margin, random, noise_lr.
(i) Stored results of the final round (`e7_pairs.csv`, `e6_detect_pairs.csv`, cheap tier with the 4-bit head or its picks), reused unchanged and labeled known.
(ii) Recomputed with this plan's cheap head (`esc`) and, where the stored arrays allow it, the mixture: AST w4g128 embeddings of the other dataset's clips (idea 5 stage 2 `emb/<ds>/w4g128__clean.npy`)
through the stored `esc` heads, which is a linear map with no training. Unseen-class sets are those of the final round (UrbanSound8K clips outside the shared classes scored by the ESC-50
heads, and ESC-50 clips outside the shared classes scored by the UrbanSound8K heads); in-domain clips are the test fold. If any array is missing for the mixture, that cell is reported as a gap.
AUROC per (seed, fold), fold means, NB intervals for margin minus noise_lr and margin minus 0.5.

## 8. Stop and verdict labels

T3: STOP, PASS or NO PASS as section 3. T5: PASS if (a) the three task-level contrasts pass section 5, (b) there is no exception in section 6 (c), (c) the mixture clause holds on all tasks; otherwise
NO PASS, with each failed clause named. Per-task and per-clause results are always printed. Every arm and every condition is reported, including those that go against the claim.

## 9. Deviations and limits stated now

1. Clipping fraction is not stored, so the regression has seven features.
2. TAU has no shifted inner validation, so the best single for TAU is chosen on clean inner validation.
3. The best single is chosen by inner-validation accuracy (the model-selection quantity), not by inner-validation AURC.
4. UrbanSound8K AST pairs rest on two shifted cells; the cheap-only arms (abstain rate) use more.
5. The learned router is trained on the pooled shifted clips of other folds (favourable to it) and, secondarily, on clean clips only (deployable). Hyperparameters are fixed, not tuned.
6. The noise detector was built for additive noise and TAU's device labels; elsewhere it is a baseline.
7. Seen-data confirmation: the p-values here test comparators, not the existence of the margin signal.
