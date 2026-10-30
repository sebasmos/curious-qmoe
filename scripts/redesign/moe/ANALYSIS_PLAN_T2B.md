# Difficulty mixture of experts, T2b: a learned gate over the stored experts

Written and committed on 24 Oct 2026, branch `moe/t2b` of curious-qmoe (from `moe/t1b`, clean tree), before any gate is fitted and before any test fold is scored.
Scripts: `scripts/redesign/moe/t2b_*.py`. Source plans: PLAN.md (`~/Documents/qmoe-results/2026-10-24_2001_difficulty-moe-plan/`, section 3 and design D2, experiment T2),
the pilot and T1b plans (same folder family). Machine: CPU only, `nice -n 10`, 2 threads in total.

## 1. Status of this test

T2b is an amendment written after seeing T1b. T1b showed that the cheap model E1 and the FP32 AST head err on different clips (flagged clips: E1 61.3, uniform FP32 AST head 63.4,
oracle of the two 76.9) while the FP32 AST head is not better than E1 on average. T2b asks whether a learned gate can capture that complementarity. It is not an independent test and it
counts in the family of tests on UrbanSound8K: {T1 primary, T1b gate, the T2b arms}. The T2b verdict uses Holm over the T2b arms (section 7) as instructed; the wider family is stated in the report.
T2 in PLAN.md section 9 was conditional on T1 or T1b passing. Neither passed. T2b is run anyway at the owner's request, and it does not revive D1.

## 2. Experts (stored embeddings, heads only)

| Name | What | Cost per clip, bit-ops (repo cost module, stage 3c `analysis/urban/costs.json`) |
|---|---|---|
| E1 | mn10 W8A8 embedding (960-d), head of the pilot | `cnn:mn10:w8a8:esc` = 2.836e10 |
| E2c | AST w4g128, 400 frames (stage 2 `emb_ast_short`), uniform head | `ast:w4g128@400+esc` = 2.086e12 |
| E2b | AST w4g128, 1024 frames (idea 5 stage 2), uniform head | `ast:w4g128+esc` = 6.980e12 |
| E2a | AST FP32, 1024 frames (idea 5 stage 2), uniform head | `ast:fp32+esc` = 1.3365e14 |

E1 test logits, E1 out-of-fold logits, the inner split (`fit_p`, `iv_p`) and the flagged set come from the pilot's `t1/raw/s<seed>_f<fold>.npz` (flagged = E1 probability margin below tau_1, 30 percent
rate on inner validation, identical to T1 and T1b). E2a, E2b, E2c test logits are the uniform AST-only heads of T1b (`urban/raw_strength`). Nothing from those two runs is retrained.
The single new computation is the out-of-fold logits of E2a, E2b and E2c on the training clips of each outer fold, which a gate needs as training data: the pilot's cross-fit recipe (5 grouped folds,
`StratifiedGroupKFold` by source recording, same random state as the pilot's E1 cross-fit, library head, 30 epochs, last-epoch logits). These touch training folds only.
Known mismatch, shared by every arm: out-of-fold heads see 80 percent of the training clips and run a fixed 30 epochs, the test-time heads see about 90 percent and are early-stopped.

Cost of a system on a clip: E1 is always run (the gate reads it), so a gated system pays 2.836e10 plus the cost of every other expert it runs. A single expert E2x used alone pays its own cost only.
Always-average pays all four. The gate network itself is under 1 percent of E1 and is not priced. Costs are per clip, averaged over test clips, then over folds.

## 3. Features and what is available when

- Cheap features (known after E1 runs, so a gate can use them to decide which other experts to run): E1 logits (10), E1 probability margin, E1 max probability, E1 entropy, and the cheap model's
  penultimate features, taken as PCA to 32 dimensions of the mn10 W8A8 960-d embedding (PCA fitted on the gate-fit rows). 45 dimensions.
- Full features (need every expert to have run): the cheap features plus, for each of E2c, E2b, E2a, logits (10), margin and entropy, plus the six pairwise Jensen-Shannon divergences of the four
  probability vectors. 87 dimensions.
- No feature uses a label. Features are standardized with the gate-fit rows.

A gate that reads full features has already paid for every expert, so its cost is the cost of all four. Such arms are the information ceiling and are reported, but they cannot satisfy the cost clause of the
success rule against the best single expert. The arms that decide which experts to run read cheap features only.

## 4. Arms

| ID | Arm | Features | Cost |
|---|---|---|---|
| A1 | Margin threshold (reference): run E1; if its probability margin is below tau, run E2a and use E2a's output, else E1's | E1 margin | E1 + rate x E2a |
| A2h | Hard router | cheap | E1 + the expert chosen (if not E1) |
| A2f | Hard router, information ceiling | full | all four |
| A3h | Soft mixture, experts with weight at least 0.05 are run | cheap | E1 + experts with weight >= 0.05 (primary); all four (also reported) |
| A3f | Soft mixture, information ceiling | full | all four |
| A4 | Always-average: mean of the four softmax vectors | none | all four |
| A5 | Stacking: multinomial logistic regression on the 40 concatenated logits | all logits | all four |

Singles E1, E2c, E2b, E2a are reported beside the arms. Extra reference, not part of the rule: average of E1 and E2a only (the repo's original ensemble).

Hard router (A2, both feature sets). An MLP (one hidden layer of 64, ReLU, dropout 0.2) outputs a distribution pi over four actions: use E1, run E2c, run E2b, run E2a (decision is the arg max at test time).
Every training clip has a known out-of-fold outcome for every action, so the training loss is the expected cost-sensitive loss L = mean over clips of sum_a pi_a (err_a + lambda c_a), with err_a = 1 if the expert of
action a is wrong on the clip and c_a the action's cost divided by the E2a cost. A clip on which no expert is correct has loss lambda c_a only, so the cheapest action is best. This is a cost-sensitive classifier whose
target is the cheapest correct expert. For A2f lambda = 0 (the experts are already paid) and the action only selects whose output to use.

Soft mixture (A3, both feature sets). An MLP of the same shape outputs weights w over the four experts (softmax), output p = sum_k w_k p_k. Loss: cross-entropy of p, plus lambda x sum_k w_k c_k, plus beta x the mean entropy of w
(minimized, which makes the weights decisive). At test time for A3h the experts run are E1 and every expert with w >= 0.05; the output is the mixture over the experts run with their weights renormalized. For A3f the mixture is over all four, lambda = 0.

Margin threshold (A1). tau is the rate-quantile of the out-of-fold E1 margins on inner validation.

Training and selection. Gates are trained on the `fit_p` rows (out-of-fold expert outputs), with Adam-W (lr 1e-3, weight decay 1e-4, batch 256, at most 60 epochs) and the epoch with the lowest loss on the `iv_p` rows (grouped inner validation by source recording).
Hyper-parameters are chosen on `iv_p` only, never on a test fold:
- lambda in {0, 0.02, 0.05, 0.1, 0.2, 0.4} (A2h, A3h); beta in {0, 0.1} (A3h, A3f); A2f: lambda 0; stacking C in {0.01, 0.1, 1, 10}; A1 rate in {0.05, 0.10, ..., 0.95, 0.99}.
- Budget B = cost of the best single expert, where the best single expert is the one with the highest out-of-fold accuracy over the training clips of that (seed, fold). Among grid points whose inner-validation cost is at most B,
  take the one with the highest inner-validation accuracy (ties: lower cost, then smaller lambda, then smaller beta). If no point is feasible, take the cheapest. Arms whose cost is the sum of all four (A2f, A3f, A4, A5) are exempt (no budget).
The scripts are split so that labels of test clips are never loaded by the fitting step: `t2b_fit.py` writes decisions and probabilities for the test clips, `t2b_score.py` is the only script that reads test labels,
and it is run once, after this file is committed and the fit step has finished.

## 5. Evaluation

UrbanSound8K, 10 official folds, seeds 0, 1, 2. Per fold, accuracy of every arm on all test clips ("overall") and on the test clips E1 flags ("flagged"), averaged over the three seeds; the fold is the unit of every test.
Also macro F1 (overall), cost per clip, share of clips for which each expert is run (hard router action shares, soft run shares), and for A2h the gate's accuracy at naming a correct expert.
Comparators for an arm: (a) the best single expert (chosen per seed and fold from out-of-fold training accuracy, test accuracy of that expert, its own cost as budget); (b) always-average A4.
Contrast per fold = arm minus comparator, in points. Nadeau-Bengio corrected two-sided t, K = 10, n_test/n_train taken as the mean ratio over folds, 95 percent interval from the same standard error, 10,000-draw bootstrap interval over folds, Cohen's d_z.

Oracle ceiling. Per fold: accuracy of "some expert among the four is correct" (overall and flagged), mean cost of the cheapest correct expert (E1 when none is correct), and the oracle of E1 and E2a (the T1b number, for continuity).
Fraction of the ceiling captured = (arm accuracy minus best single accuracy) / (four-expert oracle accuracy minus best single accuracy), on all clips and on the flagged clips, reported per arm.

## 6. Pre-registered success and stop rules (applied exactly)

An arm passes if, for each of the two comparators (best single expert and always-average):
1. mean cost of the arm (over folds) is at most the mean cost of the comparator;
2. overall contrast has a positive mean of at least +0.5 points and is at least +0.5 in at least 7 of 10 folds;
3. flagged contrast has a mean of at least +1.0 point and is at least +1.0 in at least 7 of 10 folds;
4. the Nadeau-Bengio p of the overall contrast is below 0.05 after Holm over the arms, where the arm's p is the larger of its two comparator p-values (intersection-union) and the Holm family is
   {A1, A2h, A2f, A3h, A3f, A5}, six arms (A4 is a comparator, the singles are not gates).
A3h is judged on its primary cost (experts with weight at least 0.05). Its all-four cost is reported with the same pass flags. Nothing is tuned after this file is committed; a failed arm stays reported.
Stop rule: if the best gate arm (among A1, A2h, A2f, A3h, A3f, by mean overall accuracy) beats always-average by less than +0.3 points (this includes being below it), T2b stops with that verdict and the secondary datasets are not run.
I read "within 0.3 points" as a gain below 0.3, because an arm far below always-average cannot be a win either. Verdicts: PASS (an arm passes), STOP (stop rule), otherwise FAIL (no arm passes, stop rule did not fire).

## 7. Secondary datasets

Run only if the stop rule does not fire. TAU 2020 Mobile (5 location-grouped folds, clean device and shift rows, experts E1, E2a, E2b since no short AST exists for TAU, flagged set from the pilot's `tau/raw`)
and ESC-50 (5 folds), same arms where the experts exist, same code, same fold-count scaled to 4 of 5, reported as secondary with no confirmatory claim. Any change needed for their inputs is
written as an amendment and committed before their test folds are scored.

## 8. Reporting

Per arm overall and flagged contrasts against both comparators with intervals and fold pass counts, cost relative to the best single expert and to always-average, oracle fraction captured, verdict, next step.
Results folder `~/Documents/qmoe-results/2026-10-24_<HHMM>_moe-t2b_code-<commit>/` with manifest, README, scripts, logs.
