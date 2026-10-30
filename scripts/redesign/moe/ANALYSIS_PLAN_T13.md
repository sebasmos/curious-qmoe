# Difficulty mixture of experts, T13: a fixed uniform mixture of diverse quantized experts, with their disagreement as the epistemic signal

Written and committed on 25 Oct 2026 (branch `moe/t13`, from `moe/t1c`) before any test clip was scored. What exists at this commit: the loader and input check
(`t13_common.py`, `t13_check_inputs.py`), which compare row, fold and inner-validation index arrays across experts and compute no accuracy. The pair
pre-selection (`t13_preselect.py`, reads only clean inner-validation logits) is committed after this file and before the scoring script; the scoring script is
committed before it is run. Machine: CPU, `nice -n 10`, 2 threads. Nothing is trained or extracted: T13 reads the stored per-clip logits of earlier rounds.

## 1. Question and family

Does a fixed uniform average of the probabilities of two or three diverse stored experts, with no learned gate, beat the best single model and the time-sharing hull
at equal or lower priced cost under shift, and does the disagreement among the experts say where the mixture is wrong?
Motivation (T1c, post hoc, no test attached): always-average of the cheap model and the AST head beat the fixed FP32 AST by +0.75 to +1.57 points under shift.

Family. T13 is a pre-registered experiment of the difficulty mixture-of-experts question. Experiments counted: T1, T1b, T2b, T7, T1c, T13 = **6** (7 counting the T9 any-precision
kill check, which tests a different design). Holm runs inside T13 over its **4 primary tests** (section 6). As a conservative extra line the README also reports each
primary Holm p multiplied by 6 (Bonferroni over experiments, capped at 1); the decision uses the within-T13 Holm as in T1c.

## 2. Data: stored arrays only

Each score file holds the logits of every clip in its test fold (from that fold's head) and of the fold's inner-validation clips. Heads are the stored FP32 `esc` heads
(protocol 3 recipe), one run per seed (ESC-50 and ESC-5 seeds 0 to 4, UrbanSound8K and TAU seeds 0 to 2). Experts of one task and seed share the inner split; the input check
asserts that labels, folds and inner-validation indices are identical across experts and seeds.

| Expert (label) | Backbone and head | Stored in |
|---|---|---|
| mn10 (MN10-Q8) | EfficientAT mn10, W8A8 backbone, `esc` head. UrbanSound8K: exact-length (unpadded) mn10, as in the final round | stage 3b `scores_cnn`; UrbanSound8K stage 3c unpadded `scores_unpr`; non-additive cells: final round `scores`; TAU: option B |
| mn20 (MN20-Q8) | EfficientAT mn20, W8A8 backbone, `esc` head | stage 3c `scores_cnn`; final round; option B |
| astfp32 | AST FP32 at 1024 frames, `esc` head | idea 5 stage 2 `scores`; option B |
| astw4 | AST w4g128 at 1024 frames, `esc` head | idea 5 stage 2 `scores`; option B |
| astshort | AST w4g128 at 512 (ESC) or 400 (UrbanSound8K) frames, own `esc` head. Not stored for TAU | stage 2 `scores_ast_short`; final round |

Interpretation notes. "Q8" is read as the W8A8 backbone with the FP32 `esc` head (the E1 of T1c). The five-group ESC-50 task is the stage 3c `esc5` task. The padded mn10 on
UrbanSound8K is not used. Costs: priced bit-ops per clip of the repo cost module (`analysis/<task>/costs.json` of stage 3c; option B for TAU), backbone plus head; the cost
of a mixture is the sum over the experts run, which is all of them on every clip.

Conditions (cells). `clean`; background, pink and white noise at 20, 10, 5, 0 dB (12 cells); `field` (the field mixture of earlier rounds, test assignment seed 5001,
inner-validation seed 5002); the three non-additive corruptions at three severities each (reverberation RT60 0.3, 0.6, 1.2 s; clipping at 0.5, 0.25, 0.1 of peak; low-pass at
4, 2, 1 kHz; 9 cells); UrbanSound8K, ESC-50 (50 classes), ESC-50 (5 groups), TAU clean (device A rows of the test fold) and TAU shift (every other device's rows of the test fold).
A cell is used for an expert only if its stored test and inner-validation logits are complete and finite for every seed (`availability.csv` from the input check). Known
gaps, from the T1c amendment and the final-round design: UrbanSound8K AST scores are complete only for clean, bg 10 dB, bg 0 dB and field; AST FP32 and AST w4g128 at 1024 frames
are not stored for the non-additive corruptions; the non-additive corruptions are not stored for ESC-5 or TAU; no short AST for TAU. A combination is evaluated only on
cells where all its experts are stored. Nothing is extracted to fill a gap.

Systems scored. All 5 single experts; every combination of two or three experts (ESC-50, ESC-5, UrbanSound8K: 10 pairs and 10 triples; TAU: 6 pairs and 4 triples); for each
combination the uniform arithmetic mean of the experts' softmax probabilities (the mixture, primary) and the uniform geometric mean (control: exp of the mean log probability,
renormalised). No learned gate, no threshold, no weights.

## 3. Metrics, folds, tests

Accuracy in percent (primary; the unit of T1, T1b, T2b, T1c); macro-F1 beside it. The unit of every test is the outer fold, seeds averaged inside a fold. Paired fold differences,
Nadeau-Bengio corrected resampled t-test, two-sided: t = mean(d) / sqrt((1/K + n_test/n_train) s_d^2), K folds (5 or 10), n_test/n_train the mean over folds of the condition's
test rows over the training rows (ESC: 1/4, UrbanSound8K: 1/9, TAU: shift or clean device rows of the test fold over device-A training rows). "Majority of folds" = at least 3 of 5
or 6 of 10 folds with a fold difference of at least the stated margin. The tables also show the count at 4 of 5 and 7 of 10 (T1/T2b rule); no verdict is changed by it unless stated.

## 4. Comparators (all at equal or lower priced cost than the mixture)

1. Each expert alone (every expert, descriptive).
2. Best single at equal or lower cost: among the single experts whose cost is at most the mixture's cost and whose cells include every cell of the condition set, the one with the highest
   clean inner-validation accuracy (mean over folds and seeds). Fixed before any test clip. Descriptive companion: the best single chosen per condition by test accuracy (an oracle choice).
3. Time-sharing hull of the fixed experts: Stage 2's `hull_pair` on the mean accuracy over all (seed, fold) pairs of the cell, at the mixture's cost, over the single experts stored in that cell
   (cheaper or equal expert alone, or a random time-share of one cheaper and one dearer expert, mixing weight chosen by the cost). The expected accuracy of a random time-share is
   linear in the weight, hence exact. The pair and weight are chosen on the condition's own test accuracy (the repo's convention, favourable to the hull). For a pooled condition the hull is
   built per cell and averaged over the cells. Descriptive companion: the wider repo hull (all static systems of stage 3b and 3c, macro-F1) is not rebuilt; the earlier cascade picks are
   listed instead.
4. Earlier cascade picks: stage 3c pre-registered pick P3 (macro-F1 and cost on clean and field) for ESC-50, ESC-5, UrbanSound8K, and the option B pick on TAU clean and shift, as stored in
   their `selected.csv`; compared to the primary pair's mixture in macro-F1, descriptive.
Reproduction check (T0 style, before any contrast): the macro-F1 of every single expert on every cell present in the stored `static.csv` of stage 3c or option B must equal the stored value
(tolerance 1e-9); a mismatch is reported in the README and the cell is not used until understood.

## 5. Pre-selection of the pair (clean inner validation only, no test clip)

For every pair of experts of a task: the uniform-average accuracy on the clean inner-validation clips (mean over folds and seeds), its cost c (sum of the two priced costs), and the
accuracy of the hull of the single experts on the same clean inner-validation clips at cost c. "Accuracy per cost" is operationalised as the gain over the hull at the pair's own cost
(the selection statistic of stage 2 and 3: accuracy against cost along the front). The literal ratio accuracy / cost always picks the cheapest pair, which is not informative. The pair
with the largest gain is the pre-selected pair of the task; ties by lower cost. `t13_preselect.py` writes the full table and `t13_preselected.json`, committed before the scoring script.
Best single at equal or lower cost (comparator 2) is also fixed there for every combination. Triples are not eligible for the primary; they are reported in full.

## 6. Pre-registered primary

Four primary tests, one per condition set: ESC-50 (50 classes) shifted pooled, ESC-5 shifted pooled, UrbanSound8K shifted pooled, TAU shift.
Shifted pooled for the pair = the unweighted mean accuracy over the shifted cells stored for both of its experts: 12 additive noise cells plus the 9 non-additive cells where stored (the field mixture
is a separate row, not in the pool, because it re-uses the same clips). The cell list used is written in the output. Pass for a condition, all of:
(a) the mixture's mean fold accuracy exceeds the best single at equal or lower cost by at least +0.5 points and exceeds the hull by at least +0.5 points;
(b) in a majority of folds the fold difference to each comparator is at least +0.5;
(c) Nadeau-Bengio p of the weaker of the two contrasts, Holm over the four primary conditions, below 0.05;
(d) cost of the mixture at or above the comparators' costs is by construction equal or lower for the comparators; the cost is reported and the comparison is at the mixture's cost.
Report beside the verdict: macro-F1 version of (a), the geometric-mean control, every pair on every cell, cost in bit-ops, and measured latency where available (section 8).
Per-cell contrasts for the pre-selected pair (each noise cell, field, each corruption, clean) are descriptive with raw p; they are not in the Holm.

## 7. Secondary: the epistemic signal (descriptive tests, raw p, not in the Holm)

For each combination and cell, per clip: mixture probability p_mix, mixture margin m = top1 - top2 of p_mix; **JS** = generalised Jensen-Shannon divergence of the experts' probabilities
(entropy of the mean minus mean of the entropies, natural log, equal weights; for two experts the ordinary JS); **maxdiff** = max over experts of the max-probability minus min over experts
of the max-probability (for two experts the absolute difference). Error = mixture argmax differs from the label.
(i) Prediction of the mixture's errors. AUROC of -m alone, JS alone, maxdiff alone; logistic combinations (standardised, C = 1) V1 = [m, JS], V2 = [m, maxdiff], V3 = [m, JS, maxdiff],
cross-fitted by fold: fitted on the same cell's clips of the other folds (same seed), scored on the held-out fold (primary version); and fitted on the clean clips of the other folds,
scored on the held-out fold of the cell (transfer version, secondary). Added AUROC = AUROC(V1) minus AUROC(-m), per fold, seeds averaged. The signal "adds" if, for the pre-selected pair on the
pooled shifted cells of a task, the mean added AUROC is at least +0.02 and positive in a majority of folds; the NB p of the fold differences is reported.
(ii) Rise with shift (humility curve). Threshold on the clean inner-validation clips of each fold and seed so that 30 percent of them are flagged (JS above its 70th percentile; margin below its
30th percentile); also 10 percent as a secondary. The flag (abstain or escalate) rate on the held-out test clips of every cell, by condition. Test: flag rate of the pooled shifted cells minus the
clean flag rate, per fold, NB.
(iii) Risk-coverage. Abstaining on the largest JS against abstaining on the smallest margin (and on the cross-fitted V1 score): area under the risk-coverage curve (mean error of the k most
confident clips over all k) and accuracy at 80 and 90 percent coverage, per cell; fold differences JS minus margin, NB.

## 8. Cost and latency

Cost: priced bit-ops (section 2), mixture = sum of its experts. Measured latency (median ms per clip, 2 threads, batch 1) from the full latency run of stage 3c unpadded
(`2026-10-23_1919_redesign-idea6-latency-full_code-cbf6be6/tables/steps.csv`), only where every expert was measured: mn10 and mn20 on ONNX Runtime int8 (`mn10_ort_int8`, `mn20_ort_int8`), the
short AST as `ast_<L>_n4_kleidiai` (closest real kernel to w4g128) and as `ast_<L>_fp32`; plus one mel front end if any CNN runs, one AST fbank if any AST runs, and the heads. The AST at 1024 frames
(FP32 or w4g128) was not timed, so mixtures with it have no measured latency (reported as n/a). TAU was not timed. ESC-5 uses the ESC-50 timings (same clips). The sum is sequential on one CPU.

## 9. Stop rule

T13 stops, whatever else is reported, if either holds:
S1. For every task, the pre-selected pair's mixture is less than +0.3 points above the best single at equal or lower cost in every shifted cell of the pool and in the pooled condition (no shifted condition shows a +0.3 gain).
S2. The disagreement adds no AUROC: the added AUROC of V1 over the margin is below +0.02 on the pooled shifted cells for every task.
Verdict labels: PASS (at least one primary condition passes (a) to (c)); STOP (S1 or S2 fires and no primary passes); otherwise NO PASS. The per-condition pass count is always printed.

## 10. Deviations from the brief, stated now

1. "Accuracy per cost" for the pair selection is the gain over the hull at the pair's cost (section 5); the plain ratio is degenerate.
2. Shifted pooled uses every stored shifted cell of the pair, so its composition differs by task and pair; the cell list is reported. Corruptions enter only where stored for the pair's experts.
3. Hull cells are chosen on the test accuracy of the cell (favourable to the hull).
4. Accuracy is the primary metric because T1 to T1c used it; the repo's earlier hull was in macro-F1, and the macro-F1 version is reported for the primary.
