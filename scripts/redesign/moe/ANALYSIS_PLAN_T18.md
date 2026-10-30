# Difficulty mixture of experts, T18: a hierarchy over scale and modality, routed coarse to fine

Written and committed on 25 Oct 2026 (branch `moe/t15`, from the T13 head `94f6af6`) before any view was extracted, any head was trained on a new view and any test clip was scored.
It shares the clean extraction of the views with T15 (`ANALYSIS_PLAN_T15.md`; data, conditions, same-audio check, heads, costs, statistics and honesty clauses of its sections 2, 3, 5 and 9 hold here
unless restated). Hardware and the GPU lock are as in T15. The owner proposed the design on 25 Oct 2026.

## 1. Question and family

A tier-1 expert that reads a coarse view of the audio (lower sample rate, fewer and slower frames, fewer mel rows) is cheap. Tier 2 is the full-resolution mel MN10-Q8, tier 3 the AST w4g128.
If the margin of what has answered so far decides whether the next, finer scale runs, does the hierarchy beat the equal-cost hull, the best single model and the always-average of all tiers, and does
escalation to finer scales rise with shift? The cost axis is resolution and modality, not bits (T9) and not token masking (T7).

Family. T18 is one of the 15 experiments listed in the T15 plan. It has **3 primary tests** (ESC-50 with 50 classes, UrbanSound8K, TAU shift). Holm with family size 15 as in T15: the 3 p-values are
sorted and compared with 0.05/15, 0.05/14, 0.05/13 (other experiments enter as p = 1). The README also shows Holm over the 3 tests alone and the same result with T15 and T18 pooled
(6 tests, thresholds 0.05/15 to 0.05/10).

## 2. Tier-1 candidates (extracted in the clean pass; both on mn10 W8A8, 960-d embedding, `esc` head)

The audio is resampled from 16 kHz to 8 kHz (torchaudio default sinc, low-pass at 4 kHz). Front end at 8 kHz: pre-emphasis 0.97, symmetric Hann window of 200 samples, n_fft 256, hop 160 (20 ms),
centred with reflect padding, power spectrum. This reproduces the bin spacing (31.25 Hz) and the 25 ms window of the native 32 kHz front end below 4 kHz, with half the native frame rate. Mel rows
are the native Kaldi 128-row filters restricted to the 129 columns that exist (0 to 4 kHz); a row is kept if all its weights lie in those columns (about 78 rows, counted and reported), then
`(log(mel + 1e-5) + 4.5)/5`.

| Id | View | Map into mn10 |
|---|---|---|
| LR-A | low-resolution, layout kept | 128 rows: the kept rows plus the remaining rows set to the floor value; frames at 50 per second |
| LR-B | low-resolution, cropped | only the kept rows (about 78); frames at 50 per second |

Costs are priced from the real map size and the 8 kHz front end (resampler taps, pre-emphasis, per frame window, 256-point real FFT, power, nonzero mel weights, log and scale). The stored low-pass
rows give a prior: with heads trained on full-band clean audio, a 4 kHz low-pass costs mn10 about 3 points of macro-F1 on UrbanSound8K and ESC-50 (final round `b_urban`, `b_esc50`).

**Precondition**, inner validation only, as in T15 section 5 (fold 1 of UrbanSound8K and of ESC-50, seeds 0 to 2, clean inner-validation macro-F1, same clips): a candidate qualifies if it is within
5.0 points of the full-resolution mel MN10-Q8 on both datasets. Tier 1 is the cheaper (priced) of the qualifying candidates. If none qualifies the hierarchy is not run and the verdict is STOP (precondition).
A candidate that qualifies on one dataset only is reported and not used. TAU uses the tier chosen on the two datasets.

## 3. The hierarchy

Tiers: t1 = the chosen low-resolution expert, t2 = MN10-Q8 (stored, full-resolution mel), t3 = AST w4g128 at 1024 frames (stored). All heads are the `esc` FP32 heads of the same seed.
For a clip: run t1 and take p1. Gate 1: if margin(p1) < tau1, run t2 and take p12 = mean(p1, p2); otherwise the output is p1. Gate 2: if margin(p12) < tau2, run t3 and the output is
p123 = mean(p1, p2, p3); otherwise p12. Margin = top-1 minus top-2 probability of the current output, so "the tier that answered" is the running average (variant used in the primary).
The variant where gate 2 reads the margin of t2's own output is a secondary arm.

Thresholds. For each outer fold and seed, tau1 is the quantile of the margin of p1 on the clean inner-validation clips of that fold such that the fraction q1 of those clips escalates; tau2 is the
quantile of margin(p12) among the clean inner-validation clips that reached gate 2 such that the fraction q2 of them escalates. TAU uses its clean device-A inner validation, UrbanSound8K and
ESC-50 their clean inner validation. The thresholds are then applied unchanged to the test rows of every condition. Grid: (q1, q2) in {0.1, 0.3, 0.5} x {0.1, 0.3, 0.5}.

Operating point. Per task, the setting analysed as primary is chosen on inner-validation clips of that task's shifted pool (T15 section 6: UrbanSound8K {field mix, bg 0 dB}; ESC-50 {field mix, bg 0,
pink 0, white 0}; TAU clean device-A inner validation, forced by the data) as the largest inner-validation accuracy minus the hull's inner-validation accuracy at the setting's mean priced cost; ties to
the cheaper. This is fixed before any test clip is scored (`t18_preselect.py`, committed first). All nine settings are reported on every cell.

Cost of the hierarchy: per clip c1, plus c2 when it reached tier 2, plus c3 when it reached tier 3, each tier priced with its head as in T15; the mean over the test rows of the pool gives the cost used by
the comparators.

## 4. Comparators and primary tests

For each task, on the pooled shifted test accuracy, per outer fold, seeds averaged inside a fold (accuracy primary, macro-F1 beside it):
(1) the equal-cost hull: Stage 2's `hull_pair` over the singles {t1, MN10-Q8, MN20-Q8, AST FP32, AST w4g128, short AST (not TAU)} at the hierarchy's mean cost on the pool, built on test accuracy
as in T13; (2) the best single at equal or lower cost, chosen on inner-validation accuracy of the same pool (clean device-A for TAU) among the same singles; (3) the always-average of the three tiers
(cost c1 + c2 + c3, every clip).
Nadeau-Bengio corrected t, two-sided, n_test/n_train as in T15. A task passes if all hold: (a) the three mean contrasts (hierarchy minus each comparator) are at least +1.0 point; (b) each is at least
+1.0 in a majority of folds (3 of 5, 6 of 10; 4-of-5 and 7-of-10 tabled); (c) the largest of the three p-values clears the Holm threshold of section 1; (d) escalation rises with shift: the fraction of test
rows that reach tier 2 and the fraction that reach tier 3 are higher on the shifted pool than on clean test rows (UrbanSound8K and ESC-50: clean cell; TAU: device-A rows of the test fold), in a majority of folds
for both, with Nadeau-Bengio p below 0.05 for both. T18 passes if at least two of the three tasks pass.
Stop rules: **S-a** the precondition of section 2 fails; **S-b** on every task no setting of the grid is more than 0.5 points above the uniform pair {MN10-Q8, AST w4g128} in pooled shifted accuracy.
Verdict: PASS, NO PASS or STOP.

Reported regardless of verdict: precondition table for both candidates, priced cost of each tier and of every setting, mean cost per setting and cell, escalation rates per cell, every setting against
the three comparators on every cell, the error correlation (phi) of t1 with t2 and with t3, oracles (best tier per clip), the own-tier margin variant, and measured CPU latency per tier (batch 1,
2 threads, median of 50 after 10 warm-ups, FP32 kernels, labelled as such) with the expected latency of each setting from its escalation rates.
Not tested here: a 4-bit tier 1, a downsampled-waveform CNN, fine-tuning a tier, learned gates.
