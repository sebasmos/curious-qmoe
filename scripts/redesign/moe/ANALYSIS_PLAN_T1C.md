# Difficulty mixture of experts, T1c: specialization and gating where the strong model is clearly stronger

Written and committed on 25 Oct 2026, branch `moe/t1c` of curious-qmoe (from `moe/t7`, clean tree), before any head of T1c is trained and before any T1c test fold is scored.
Scripts: `scripts/redesign/moe/t1c_*.py`. Source plans: PLAN.md (`~/Documents/qmoe-results/2026-10-24_2001_difficulty-moe-plan/`), the pilot plan (T1; its Part B definitions apply unchanged unless
this file says otherwise), `ANALYSIS_PLAN_T1B.md` (strength gate), `ANALYSIS_PLAN_T2B.md` (gate arms and rule), `ANALYSIS_PLAN_T7.md`.
Machine: CPU only, `nice -n 10`, 2 threads in total. Stored embeddings only, nothing is extracted.

## 1. Status of this test and the family

T1c is an amendment written after seeing T1 (stopped on UrbanSound8K at -2.37 points and on TAU shift at -1.71), T1b (stopped at the strength gate: E2a minus E1 on flagged
UrbanSound8K clips +2.06 points but positive in 5 of 10 folds), T2b (no gate beat always-average, best gate arm -0.25 points) and T7 (masked AST, stopped, 6.96 points below the line).
All four verdicts were known when this plan was written, so T1c is not an independent test. The reason for it: those four ran on clean UrbanSound8K, where the full 4-bit AST (86.1) is
no better than the small CNN (86.5), so a specialist had little to specialise on. The AST is clearly stronger under shift (stage 3b, 3c and final-round results: low-SNR noise, the field
mix, TAU 2020 Mobile devices where the CNN trails by about 7 points). T1c repeats the specialization test (T1), the strength gate (T1b) and the gated-mixture test (T2b) on those conditions.

Family. The family of experiments on this question now counts five: T1, T1b, T2b, T7, T1c. T9 (the any-precision kill check) tests a different design (D4) and is listed but not counted; the
verdict says "5, or 6 with T9". Inside T1c, Holm is applied over the T1c conditions (section 3, five conditions). For every raw p below 0.05 that survives Holm the README adds the bound
alpha/5 = 0.01 (Bonferroni over the five experiments) as a descriptive line; it never changes a verdict.

## 2. Stored inputs (inventory, read only)

All paths under `~/Documents/qmoe-results/`. "Cells" are clean plus {bg, pink, white} x {10, 5, 0} dB, full rows (every clip).

| Role | UrbanSound8K (8,732 clips, 10 folds) | ESC-50 (2,000 clips, 50 classes, 5 folds) | TAU 2020 Mobile (23,035 rows, 5 folds) |
|---|---|---|---|
| E1 cheap, mn10 W8A8, 960-d | stage 3b `emb_cnn/urban8k/mn10_w8a8__<cell>.npy` | stage 3b `emb_cnn/esc50/mn10_w8a8__<cell>.npy` | option B `emb_cnn/tau/mn10_w8a8__clean.npy` |
| E2a strong, AST FP32, 1024 frames, 768-d | idea 5 stage 2 `emb/urban8k/fp32__<cell>.npy` | idea 5 stage 2 `emb/esc50/fp32__<cell>.npy` | option B `emb_ast/tau/fp32__clean.npy` |
| E2b, AST w4g128, 1024 frames | idea 5 stage 2 `emb/urban8k/w4g128__<cell>.npy` | idea 5 stage 2 `emb/esc50/w4g128__<cell>.npy` | option B `emb_ast/tau/w4g128__clean.npy` |
| E2c, AST w4g128, short | idea 6 stage 2 `emb_ast_short/urban8k/w4g128__<cell>.npy` (400 frames) | idea 6 stage 2 `emb_ast_short/esc50/w4g128__<cell>.npy` (512 frames) | none stored |
| Cost module | stage 3c `analysis/urban/costs.json` | stage 3c `analysis/esc50/costs.json` | option B `analysis/tau/costs.json` |

Present and not used: mn20 W8A8 (stage 3c cells for both datasets; option B for TAU), the AST n4 and w8g64 variants, and the three non-additive corruptions of the final round (reverberation,
clipping, low-pass) for mn10/mn20 and AST w4g128 at 400/512 frames. The final-round corruptions have no FP32 AST embedding; using them as test beds would need FP32 AST extraction at 400 or 512
frames for nine conditions on each dataset (MPS, not done here). TAU has no short AST embedding, so TAU has three experts, not four. The final-round folders hold corrupted conditions only, so
clean data always comes from the folders above. `t1c_check_inputs.py` checks shapes, finiteness, the names checksum against the CSV row order and the presence of every cell, and writes a sha256 for
every file used.

## 3. Test beds, conditions, flagged clips

Seeds 0, 1, 2 averaged inside a fold; the fold is the unit of every test. UrbanSound8K: 10 official folds. ESC-50: 5 official folds, label = the true 50 classes (CSV column `label`), inner split
grouped by source recording (name token 1). TAU: 5 location-grouped folds as in the pilot, training on device-A rows of the training folds.

Conditions (test clips only; heads never see a shifted clip):

| ID | Bed | Condition | Test rows of a fold |
|---|---|---|---|
| U-field | UrbanSound8K | field mix: half the clips clean, the rest spread over the nine noise cells, assignment `audio_shift.field_assignment(n, 5001)` | one row per test clip |
| U-0dB | UrbanSound8K | 0 dB noise pooled over bg, pink and white | each test clip three times (once per noise type) |
| E-field | ESC-50 | field mix, same function, n = 2000 | one row per test clip |
| E-0dB | ESC-50 | 0 dB noise pooled over bg, pink and white | three rows per clip |
| T-shift | TAU | the fold's rows of the eight non-reference devices | the pilot's `cond == 1` rows |

Descriptive only, never tested: clean (every bed; TAU: device A rows), each of bg, pink, white at 0 dB alone. Clean thresholds everywhere.

Flagged: a test row is flagged when E1's probability margin (top-1 minus top-2 softmax probability, on that row's own embedding) is below tau_1, the pilot's threshold: the value that escalates 30
percent of the clean inner-validation clips (`analyze.thr_rate`, as in `run_t1.py`). The field-mix validation seed 5002 is not used. Gate and contrast accuracies are computed on flagged rows only.
The Nadeau-Bengio ratio is n_test clips / n_train clips (TAU: rows of the condition / training rows), mean over folds.

## 4. Training: clean data only

E1, every E2 head and every specialist train on clean training clips only (TAU: clean device-A training rows). Nothing about shifted audio reaches a training set, a selection set or a threshold.
Recipe, splits, cross-fit difficulty labels, tiers, S_2 (the training clips whose out-of-fold E1 margin is below the 30th percentile) and the selection set (inner-validation clips with m_1^oof
below tau_1^oof) are the pilot's Part B definitions. UrbanSound8K and TAU read the pilot's `t1/raw` and `tau/raw` for `tr, fit_p, iv_p, S2, tier, z1_oof, tau1_oof` and retrain E1 (same seed, same call);
the retrained clean test logits must equal the pilot's `z1_te` (check R1, tolerance 1e-5, reported). ESC-50 has no pilot files, so `t1c_jobs.py` recomputes the same pipeline for it (cross-fit by
StratifiedGroupKFold(5) on recordings, 30 epochs, J = 5). Heads: the pilot head with the class count as an argument (`heads.fit_head(ncls=...)`, 50 for ESC-50; default unchanged), hidden sizes
[640, 320], the `config/urban8k.yaml` recipe for all three beds (the `esc50.yaml` train block holds the same values).

Heads trained per (seed, fold): E1 (macro F1 selection on the inner validation part), U-A heads on E2a, E2b and E2c (AST embedding only, uniform weights, accuracy selection on the flagged
inner-validation clips; exactly the T1b "U-A" head), and the T1 arms of section 6 on E2a. Each head is applied to the test rows of every cell, and conditions are assembled from cells afterwards.
Jobs do not read a test label; only the scoring scripts do.

## 5. Strength gate (T1b rule, per condition)

For each condition, on the flagged rows: accuracy of E2a's uniform head U-A minus accuracy of E1, seeds averaged inside a fold. E2a is "stronger" in that condition when the mean over folds is at
least +2.0 points and the difference is positive in at least 7 of 10 folds (UrbanSound8K) or 3 of 5 folds (ESC-50, TAU; a majority of five). This is the T1b rule with the fold count adapted. A
literal reading, "beats by 2.0 points in 7 of 10 folds", would ask each of those folds to clear +2.0; the count of folds at or above +2.0 is reported beside every gate row, and the verdict follows
the T1b rule as written here. Secondary gate rows, never decisive: E2b (4-bit) and E2c minus E1. The oracle (E1 or E2a right) is reported.
A condition whose gate fails gets no T1 contrast and no T2b arms, and its p-value enters the Holm family as 1. Each condition is gated on its own; a bed is reported with the gate result of each of
its conditions.

## 6. T1 on the conditions that pass the gate

Arms (pilot B6, E2 = E2a, no E3): input B (AST embedding, z_1, m_1): U (twin), M0, R05, R10 (primary), R20, F (focal, gamma 2), T (tier mask); input A (AST embedding only): U-A (twin of the A arms),
M0-A, R10-A. Specialist weights over S_2 as in the pilot. Test inputs use z_1 and m_1 of E1 on the condition's own row.
Primary contrast per condition: R10 (input B) minus U (input B), accuracy on the flagged rows, in points, per fold; Nadeau-Bengio corrected two-sided t (df K-1), 95 percent interval, d_z, 10,000-draw
bootstrap over folds, counts of folds at or above +0.5 and +1.0.
Rule (the T1 rule): PASS when the mean contrast is at least +1.0 point, at least +1.0 in a majority of folds (more than half: 6 of 10, 3 of 5) and the NB p is below 0.05 after Holm over the five T1c
conditions; STOP when the mean contrast is below +0.3; otherwise INCONCLUSIVE. Secondary arms: raw NB p and Holm within the eight secondary arms of a condition, exploratory, never change a verdict.
The ESC-50 clean contrast is reported as the control (the T1 text predicts under 0.5 there), not decided.

## 7. T2b on the conditions that pass the gate

Run on every gate-passing condition whatever the T1 verdict (a gated mixture does not need a specialist). Arms, features, losses, grids, selection and rules are ANALYSIS_PLAN_T2B.md sections 3 to 6
verbatim, with these changes:
1. Experts: UrbanSound8K and ESC-50 use E1, E2c, E2b, E2a (four); TAU uses E1, E2b, E2a (three; actions and mixture weights over three experts). Expert heads are the uniform AST-only heads.
2. Gate training data are the clean out-of-fold logits of the experts on the training clips (`t1c_jobs.py --phase oof`, the T2b recipe: 5 grouped folds, 30 epochs, last-epoch logits); grids and
   selection use clean inner validation only. A fitted gate is then applied to a condition's test rows, whose features come from that condition's own expert outputs. The PCA-32 of the mn10
   embedding is fitted on clean gate-fit rows and applied to the condition's rows.
3. Costs from the cost module of the bed (section 2); the budget B is the cost of the best single expert, chosen by clean out-of-fold accuracy.
4. Comparators: best single expert (that choice) and always-average, both evaluated on the same condition rows. Flagged rows as in section 3. Overall means all rows of the condition.
5. Pass conditions per comparator: cost clause; overall mean at least +0.5 and at least +0.5 in a majority of folds (6 of 10, 3 of 5); flagged mean at least +1.0 and at least +1.0 in a majority
   of folds; NB p of the overall contrast below 0.05 after Holm over the six arms (arm p = larger of its two comparator p-values), and then below 0.05 after Holm over the T1c conditions applied to
   each condition's smallest arm-adjusted p (failed-gate conditions carry p = 1). "Majority" replaces "7 of 10".
6. Stop rule per condition: the best gate arm among A1, A2h, A2f, A3h, A3f by mean overall accuracy beats always-average by less than +0.3 points: STOP for that condition.
7. Stacking for 50 classes uses the same C grid. The information-ceiling arms (A2f, A3f, A5) are reported and cannot pass the cost clause against the best single expert.
Hyper-parameters never see a shifted row. A failed arm stays reported.

## 8. Verdict and report

Per condition: gate, T1 verdict, T2b verdict, all arms and contrasts. Overall: PASS if any condition has a T1 PASS or a T2b arm passing both Holm steps; STOP if every gate-passing condition stops on
T1 (mean below +0.3) and on T2b; if no condition passes the gate, the verdict is "no condition where the strong model is clearly stronger on the flagged clips, in these embeddings". The report states the
family size (5, or 6 with T9), the Holm family (five conditions) and the next step, and ends with a summary of at most 250 words.

## 9. Order and test-label policy

1. This file, committed. 2. Scripts and `t1c_check_inputs.py`, committed, before any head is trained. 3. `t1c_jobs.py` phases `strength` (E1, U-A heads, ESC-50 cross-fit, logits on all cells), `oof`,
`arms`; none reads a test label. All arm heads are trained for every job (that costs seconds and scores nothing). 4. `t1c_strength.py` is the first script that reads test labels; the gate verdicts are
written before the T1 and T2b scoring scripts run, and those score only the conditions that passed. 5. T2b fit scripts read no test label. 6. Nothing is tuned after this file is committed and no
run is repeated with other settings.

## 10. Deviations stated in advance

1. E1 is mn10 W8A8 on both datasets (the T1/T1b cheap model); the stage 3c ESC-50 pick uses mn20-Q8, which is not used here.
2. E2a (AST FP32, 1024 frames) is the single decisive strong expert in all gates, as in T1b. The pilot's TAU E2 was the 4-bit AST; the TAU gate here uses FP32 and has E2b as a secondary row.
3. The ESC-50 recipe is the UrbanSound8K recipe with 50 classes. The ESC-50 inner-validation part is about 160 clips and its flagged part about 50, so head selection on it is noisy for every arm alike.
4. The "0 dB" condition pools three noise types of the same clips; the Nadeau-Bengio ratio counts clips, not rows.
5. The fold-count rules use "majority" as instructed; T1 and T2b used 7 of 10 and 4 of 5. Counts at the old thresholds are reported in the tables.
6. The 4-bit AST at 400 or 512 frames (E2c) is used only in T2b and as a secondary gate row.
7. No new extraction, so no FP32 AST rows for the three non-additive corruptions.

## Amendment 1 (25 Oct 2026, written after `t1c_check_inputs.py` and before any head is trained or any T1c test fold is scored)

The input check found that the stored UrbanSound8K AST embeddings (E2a, E2b, E2c) hold every clip only for clean, bg 10 dB and bg 0 dB. For bg 5 dB and for pink and white at 10, 5 and 0 dB they hold only the
rows the field mix needs (the 485 test-assigned rows of seed 5001 and the 485 validation-assigned rows of seed 5002, about 940 finite rows per cell). The stored mn10 embeddings and all ESC-50 embeddings
are complete. Consequences, fixed now:
1. U-0dB is bg at 0 dB only (every test clip once, all four experts stored). Pink and white at 0 dB on UrbanSound8K would need AST extraction of about 8,250 further clips per noise type for each of the
   three AST configurations; that is not done. E-0dB is unchanged (bg, pink and white pooled, three rows per clip).
2. The field mix is unaffected: its rows are exactly the stored ones. The input check asserts that for the partial cells the test-assigned rows are finite.
3. Descriptive conditions on UrbanSound8K are clean and bg 0 dB; on ESC-50 clean, bg 0, pink 0, white 0.
No rule, threshold or arm changes. The family stays at five conditions.
