# T9, D4 any-precision kill check: analysis plan

Written and committed on 24 Oct 2026 before any test fold was scored. Branch `moe/anyprecision-check` of curious-qmoe, from
`paper/quantized-gate`. Experiment T9 of `~/Documents/qmoe-results/2026-10-24_2001_difficulty-moe-plan/PLAN.md` (design D4).
CPU only, `nice -n 10`, 2 threads, nothing extracted, nothing trained. Every number is computed from per-clip logits that earlier
runs stored.

## Question

D4 asks whether one family of 3, 4 and 8-bit AST models, used as three difficulty tiers, beats random time-sharing of the static
systems. The tier-trained version (each width trained on its own tier) needs training and is not run here. The check that stored
arrays allow is the inference-time half: send each clip to the cheapest precision whose margin passes a threshold, and compare with
the hull of the static front. If no pre-registered pick beats the hull by 1.0 point on two tasks, D4 is closed.

## Inputs (read only)

* Idea 5 Stage 2 (`2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/scores`): AST at 1024 frames, configs w3g64, w4g128, w8g64
  (direct, group-wise) and n3g64, n4g128 (nested), esc head (trained on FP32 embeddings, applied to every precision).
* Stage 2 of idea 6 (`scores_ast_short`): real-length AST (512 frames on ESC-50, 400 on UrbanSound8K), w3g64, w4g128, w8g64, n4.
* Option C (`2026-10-23_2133_redesign-idea6-optionC_code-d6e7913`): 3-bit tiers at real length, transferred (`scores_tr`: n3g64) and
  with their own heads (`scores_own`: w3g64o, n3g64o), the stored per-system costs (`analysis/<task>/costs.json`, priced view) and
  the stored static table (`analysis/<task>/static.csv`), which this run must reproduce.
* The static systems of the hull: every system of option C's hull set "new" (CNN and AST statics of Stages 2, 3b, 3c and option C,
  the exact-length mn10 on UrbanSound8K), loaded from the same score folders by the same code path as option C's `analyze_c.py`.
* Tasks: ESC-50 with 50 classes, ESC-50 with five groups, UrbanSound8K. Conditions: clean and the field mixture (the half-clean,
  half-noise assignment of Stage 2). Seeds 0 to 4 (ESC-50) and 0 to 2 (UrbanSound8K), the protocol-3 outer folds.

## Settings (the family, six per task)

Each setting is a triple (3-bit, 4-bit, 8-bit) of stored AST systems, esc head.

| id | 3-bit | 4-bit | 8-bit | length |
|---|---|---|---|---|
| F-direct | w3g64 | w4g128 | w8g64 | 1024 |
| F-nested | n3g64 | n4g128 | w8g64 | 1024 |
| R-direct-tr | w3g64 | w4g128 | w8g64 | real, transferred 3-bit head |
| R-direct-own | w3g64 with its own head | w4g128 | w8g64 | real |
| R-nested-tr | n3g64 | n4 | w8g64 | real, transferred 3-bit head |
| R-nested-own | n3g64 with its own head | n4 | w8g64 | real |

The nested chains use two stored parents at 1024 frames (n3g64 comes from w8g64, n4g128 from w8g128), so the memory claim "one stored
parent" does not hold for F-nested; only the cost question is tested. The README says so.

## The three-way gate

For a test clip: run the 3-bit tier. If its margin (top-1 minus top-2 probability) is at least t1, the clip stays at 3 bits. Otherwise
run the 4-bit tier from scratch; if its margin is at least t2 the clip stays at 4 bits; otherwise run the 8-bit tier from scratch.
Final prediction by rule `tier` (the last tier that ran) or rule `avg` (mean of the probabilities of the tiers that ran).

Difficulty tiers and thresholds. The tier boundaries are quantiles of the margins on the inner-validation rows of the clean data of
each outer fold. Those rows belong to the training folds and were not used to fit that fold's head, so they are the out-of-fold
training margins; no test row of any condition enters t1 or t2. t1 is the margin below which a share b1 of the 3-bit
inner-validation margins falls, b1 in {5, 10, 20, 30, 50}%. t2 is the margin below which a share b2 of the 4-bit
inner-validation margins falls, b2 in {20, 30, 50, 70}%; a clip reaches the 8-bit tier only through both gates, as in option C's
three-tier cascade E1. Thresholds are fixed per (seed, fold) on clean audio and applied unchanged to the field mixture.

Designs: 6 settings x 2 rules x 5 values of b1 x 4 values of b2 = 240 per task.

## Cost

Priced bit-ops view of option C: frontend + backbone + head, one 32 x 32 rescale per output element of every quantized layer (group
scales priced), values taken from option C's `costs.json`. A design's mean cost is c3 + r1 c4 + r2 c8, where r1 and r2 are the
measured shares of clips that reach the 4-bit and 8-bit passes on the condition being scored (no reuse of the earlier pass: every
escalated pass is a full forward pass). A nested child needing k+1 planes: for the system's own nested tiers the weight bits are priced
at k+1 (n3g64 as 4-bit weights with group 64, n4g128 as 5-bit with group 128, n4 per-channel as 5-bit), recomputed with the same bit-ops
formula as option C. Nested statics inside the hull keep option C's stored plain price (the hull stays as strong as before).

## Hull

Per condition and design: the best static system or two-system random time-sharing at the design's mean cost, on mean F1 of the
statics (option C's `hull_pair`), with the exact expected macro F1 of the mixture on each test fold (`mix_expected_f1`). The static
front is the full "new" set of option C. The always-average points of the setting join the set: the unweighted mean of the softmax
outputs on every clip of 3+4, 3+8, 4+8 and 3+4+8, each costing the sum of its tiers (nested tiers at k+1). Inner-validation selection
uses the same hull on the inner-validation rows (200 random draws per fold, seed `RNG_IV`, as in Stage 2).

## Selection (one pick per task)

On clean inner validation only: the design with the largest mean over (seed, fold) of macro F1 minus the inner-validation hull F1 at the
design's inner-validation cost; ties go to the lower cost. One design per task, with no look at any test fold.

## Test and decision

The pick is scored once on the clean and the field test folds. Difference to the hull per (seed, fold), seeds averaged inside folds,
two-sided t over fold means, Nadeau-Bengio corrected t (variance x (1/k + n_test/n_train)), exact sign-flip p. Holm over the three
tasks, separately for the field mixture (H1, primary) and clean (H2, secondary).

D4 passes the kill check when, on the field mixture, at least two of the three picks have a mean difference to the hull of at least
+1.0 macro-F1 point and a Holm-adjusted Nadeau-Bengio p below 0.05. Otherwise D4 is closed. The number of tasks that reach +1.0 point
alone (no test) is reported next to it. A pass would be a surprise and would go to the audit of the plan (T11) before it is believed.

## Checks before the verdict

* Static reproduction: for every system of the hull set, the macro F1 per (seed, fold, condition) computed here equals option C's
  `static.csv` to 1e-9. Any mismatch stops the run.
* The selection and test code is committed before the test phase runs; `logs/plan_commit.txt` in the results folder holds the commit
  hash and the time of the test phase.

## Descriptive (no verdict)

* All 240 designs per task against the hull on both conditions: the best difference, the number with a difference of +1.0 point or
  more, raw fold-t p.
* The pick with the nested tiers priced plain (k bits), and the pick against a hull whose nested statics are also priced at k+1.
* Share of clips ending at each precision on clean and field; accuracy of each precision on the clips assigned to each tier.
