# Difficulty mixture of experts, first pilot (T0, T1a, T1) on UrbanSound8K: analysis plan

Written and committed on 24 Oct 2026 before any head of this pilot was trained and before any UrbanSound8K test fold was scored by
this pilot. Branch `moe/difficulty-pilot` of curious-qmoe (from `paper/quantized-gate`). Scripts: `scripts/redesign/moe/`.
Source plan: `~/Documents/qmoe-results/2026-10-24_2001_difficulty-moe-plan/PLAN.md` (sha256 16b982d078f0df3974a3497b4c50d4b9192affa31806f23a5b4a913bfc215791).
Machine: CPU only, `nice -n 10`, 2 threads. The stored score files that T0 reads were produced and seen in earlier rounds; T0
reproduces them and tunes nothing. Every number that T1a and T1 report is computed after this commit.

## Part A. Rules copied verbatim from PLAN.md

### A1. Experiment table rows T0, T1a, T1 (PLAN.md section 9, columns: ID, what it tests, data and compute, pre-registered success rule, result that counts as proof, result that stops it)

| T0 | Reproduce the baselines and the oracle from stored arrays: single-model front, hull, always-average, cheapest-correct-model oracle, per-clip correctness table | Stored scores for ESC-50, UrbanSound8K, TAU. CPU, minutes | Recomputed hull and gate numbers match BACKLOG -52, -57, -59 to three decimals | Match, plus the oracle headroom per task written down | A mismatch: stop and find out why before anything else |
| T1a | Do out-of-fold difficulty labels from the cheap model predict the strong model's errors? Spearman between the models' out-of-fold margins, AUROC of cheap-model difficulty for strong-model errors, on the training folds and then on test folds | Stored embeddings, grouped cross-fitting of heads. CPU, under an hour per task | Spearman >= 0.3 and AUROC >= 0.6 in at least 7 of 10 UrbanSound8K folds | Both thresholds met on UrbanSound8K and TAU | Spearman < 0.15 on both: difficulty is model-specific, D1 and D2 stop |
| T1 | Specialization ablation: E2/E3 trained on the escalated set vs. uniform vs. focal vs. difficulty-tier mask vs. replay ε sweep, same inputs and architecture, evaluated on the clips the cheap model flags | Stored embeddings (cheap CNN + AST), heads in seconds. CPU, a few hours per task | Specialist beats uniform twin by >= 1.0 point on the flagged clips (UrbanSound8K), 7 of 10 folds, NB p < 0.05 | The contrast holds on UrbanSound8K and TAU-shift and is under 0.5 on ESC-50 clean | Contrast < 0.3 on UrbanSound8K: specialization does not exist in this stack, D1 stops, write the negative result into the humility paper |

### A2. Order of work and the first pilot (PLAN.md section 9)

The order is T0, T1a, T1, T2, T3, T4, T5, T6, then T7 and T8 in parallel with T6 if T1 passes, T9 any time (it costs an hour), T10 only on the condition stated, T11 last. If T1a or T1 stops, T2 to T6 are skipped, because a router cannot create a specialization the experts do not have, and the project moves to the framing in section 10.

**First pilot to launch.** T0, T1a and T1 on UrbanSound8K, from the stored mn10 and AST embeddings (folders `emb_cnn/urban8k*` and `emb_ast/urban8k_400` under `2026-10-24_1507_redesign-idea6-final_code-ed82400`). Heads train in seconds, so the full set of specialization arms runs on a laptop CPU in an afternoon. Start by checking that every clip has an embedding in both backbones and that the folds match the protocol-3 splits; I did not open the files. Write the T1a and T1 thresholds into the repo before the first run.

### A3. Difficulty definition and labels (PLAN.md section 3)

**Difficulty definition and labels (training folds only).**
1. For each training fold, run grouped inner cross-fitting: split the training clips into J groups by source recording (J = 5), train the cheap head on J-1 groups and score the held-out group at every epoch e = 1..T. Result: an out-of-fold trajectory of margin and true-class probability for every training clip.
2. Difficulty score d_i = 1 - p̄_i with p̄_i = (1/T) * sum_e p_{y_i}^{(e)}(i) (the data-map confidence, Swayamdipta et al.). Variants kept for T1a: AUM_i = (1/T) * sum_e m_y^{(e)}(i) (Pleiss et al.), forgetting count (Toneva et al.), EL2N (Paul et al.).
3. Tier labels from the training-fold quantiles of d: easy if d_i < q_e, hard if d_i >= q_h, medium otherwise. Defaults q_e = 0.5 and q_h = 0.8 (quantiles of the training-fold distribution), tuned only on inner validation.
4. Pairwise label for routing: a_i = 1 if the cheap expert is wrong and the next expert is right (the clip is worth escalating), from the same out-of-fold models.
No test fold clip ever enters any score, threshold, or quantile.

### A4. Common bar, item 2 (PLAN.md section 3)

2. Fold-level paired differences, tested with the Nadeau-Bengio corrected resampled t-test: t = mean(d) / sqrt( (1/K + n_test/n_train) * s_d^2 ), d the per-fold difference in accuracy (macro-F1 reported beside it), K folds. For 5-fold n_test/n_train = 1/4; for 10-fold UrbanSound8K 1/9.

### A5. Design D1 prediction and stage 1 rule (PLAN.md section 4)

**Falsifiable prediction.** On UrbanSound8K (10 official folds), E2 trained on S_2 beats the same E2 trained uniformly by at least 1.0 accuracy point on the clips with m_1 below τ_1, and the full system beats the hull by at least 1.0 point at equal cost. On ESC-50 clean audio the same contrast is within +/- 0.5 points. If E2-on-S_2 gains under 0.3 points on UrbanSound8K, D1 is dead.

**Stages.**
- Stage 1 (under a day, stored embeddings, CPU): T0, T1a, T1 on UrbanSound8K. Go rule: specialization contrast >= +0.5 points on the escalated subset in at least 7 of 10 folds, with the cross-model difficulty correlation (T1a) above 0.3 Spearman. No-go otherwise.

### A6. D1 architecture, training and loss (PLAN.md section 4)

**Architecture.** E1 sees the cheap embedding (1536-d EfficientNet-B3 or the MN10 vector). E2 and E3 see [AST embedding (768-d), z_1, m_1]; E3 additionally sees z_2. Gates are margin thresholds τ_1, τ_2 (the repo's current gate), so any gain comes from expert training and not from routing. Router inputs: only m_1 and m_2. Precisions: E1 8-bit, E2 and E3 group-wise 4-bit heads or 8-bit, as the repo's cost tables allow.

**Training.** Stage A: train E1 on all clips, cross-fit for out-of-fold scores. Stage B: compute difficulty d_i and the escalated sets S_2 = {i : m_1^{oof}(i) < τ_1} and S_3 = {i : m_2^{oof}(i) < τ_2}. Stage C: train E2 on S_2 (with a small replay of the rest) and E3 on S_3.

**Loss.**
L_k = (1 / Σ_i w_{ik}) * Σ_i w_{ik} * (1 - p_{k,y_i}(i))^γ * CE(p_k(i), y_i), with
- w_{i1} = 1 for all i (E1 sees everything and must stay a generalist),
- w_{i2} = 1[i in S_2] + ε * 1[i not in S_2],
- w_{i3} = 1[i in S_3] + ε * 1[i not in S_3],
- ε in [0, 0.2] the replay weight, γ >= 0 the focal exponent (γ = 0 gives plain CE), (1 - p_{k,y})^γ down-weights clips the expert already gets right.
The "mask" version sets ε = 0. The "difficulty tier" version replaces S_k by the tier labels from section 3. Both are run in T1 (experiment table).


## Part B. Exact definitions used in this pilot

Everything below is fixed now. Where a definition fills a gap in Part A it says so. Deviations from PLAN.md are listed in B9 and
will be repeated in the results README.

### B1. Data, folds, seeds

- UrbanSound8K, 8,732 clips, 10 classes, the 10 official folds, leave one fold out (protocol 3: the inner validation split of a
  training set is grouped by source recording, token 0 of the clip name, `QWave.datasets.inner_split`, 10 percent).
- Row order is the CSV `~/Documents/DATASETS/Urban8k/ast_768/urbansound8k.csv`. Step 1 (`s1_check_inputs.py`) checks that the CSV
  folds and labels equal the official metadata, that the embeddings and the score files have 8,732 rows in that order with the same
  names checksum, and writes the file list with sha256.
- Seeds 0, 1, 2. A seed fixes the inner splits, the cross-fitting groups, the initialisation and the loader order. Seeds are
  averaged inside a fold; the fold is the unit of every test (as in stage 3c).
- Stored embeddings (clean, read only): E1 `mn10_w8a8` (960-d, stage 3b `emb_cnn/urban8k`), E2 AST `w4g128` at 256 frames (768-d,
  stage 3b `emb_ast/urban8k_256`), E3 AST `w4g128` at 384 frames (stage 3b `emb_ast/urban8k_384`). The `emb_cnn/urban8k*` and
  `emb_ast/urban8k_400` folders of the final round hold corrupted conditions only, so they are not used.

### B2. The head recipe (all E1, E2, E3 heads and all cross-fitting heads)

`QWave.qmoe_layers.BitNetExpert(num_bits=32)` (the `esc` FP32 head: Linear, LayerNorm, ReLU, Dropout twice, FP32 classifier), hidden
sizes [640, 320], dropout 0.1953403862875243, AdamW lr 0.0005793146438537801, weight decay 0.0051275736171219844, label smoothing
0.10851194368924916, inverse class frequency weights (from the fit part), batch size 64, gradient clipping 1.0, at most 100 epochs,
ReduceLROnPlateau (factor 0.6156433661410728, patience 5) and early stopping (patience 15), standard scaling fit on the fit part.
This is `config/urban8k.yaml` and `QWave/train_utils.build_training_recipe`. The training loop is written in
`scripts/redesign/moe/heads.py` because the library loop takes no per-sample weights. Differences from the library loop: per-sample
weights; the loss is sum(w_i cw_{y_i} CE_i) / sum(w_i cw_{y_i}) with CE_i the label-smoothed cross-entropy; the model is selected on
accuracy of the validation clips named in B5 (stated per head type below). The library `scripts/benchmark.py` is not called.

### B3. Out-of-fold difficulty labels (training folds only)

For each outer fold and seed, with the training clips of that fold:
1. J = 5 groups from `StratifiedGroupKFold(5, shuffle=True, random_state=seed)` on the training clips, groups = source recording.
2. For each group j, train the E1 head (B2 architecture) on the other four groups for T = 30 fixed epochs (constant learning rate,
   no validation, no early stopping), and after every epoch e = 1..30 record, for the held-out group, the softmax and the logits.
   This gives for every training clip an out-of-fold trajectory.
3. Difficulty d_i = 1 - mean_e p_{y_i}^{(e)}(i) (data-map confidence). Variants: AUM_i = mean_e (z_y - max_{c != y} z_c); EL2N at epoch
   5 = || p^{(5)} - onehot(y) ||_2; forgetting count = number of epochs e >= 2 where the clip was correct at e-1 and wrong at e.
4. Tiers: easy if d < q50 of d over the training clips of the fold, hard if d >= q80, medium otherwise (q_e = 0.5, q_h = 0.8, the
   defaults of Part A3, not tuned).
5. The out-of-fold probabilities and logits at epoch 30 are the training-clip outputs of E1 ("oof"): p_1^oof, z_1^oof, margin
   m_1^oof. The same cross-fitting is run on the AST 256 embedding (E2 inputs, B3a) and the AST 384 embedding is not cross-fitted.
   B3a: the E2 cross-fit produces p_2^oof, z_2^oof, m_2^oof for the training clips with the same groups; its heads see the AST
   embedding only (no z_1).
6. The margin everywhere is the probability margin, top-1 minus top-2 softmax probability, the repo's gate (`analyze.margin`).

### B4. The cheap model, the gate and the escalated sets

- E1 (final): the B2 head trained on the fit part (90 percent by recording-grouped inner split) of the training clips, validated
  and early-stopped on macro F1 of the inner validation part (the protocol 3 recipe unchanged).
- Gate threshold for evaluation, "clean validation threshold": tau_1 = the threshold that escalates 30 percent of the inner
  validation clips by E1's margin (`analyze.thr_rate`, rate 0.30, the repo's b30). A test clip is flagged when m_1 < tau_1.
  The flagged share of each test fold is reported.
- Escalated training set S_2 = training clips whose m_1^oof is below the 30th percentile of m_1^oof over the training clips of
  the fold (tau_1^oof). It is calibrated on the out-of-fold margins because the final E1 is a different model.
- S_3 = the clips of S_2 whose m_2^oof (B3a) is below the median of m_2^oof over S_2. At test time E3 is evaluated on the test clips
  that are flagged by E1 and have m_2 < tau_2, with tau_2 = the median of m_2 over the inner validation clips flagged by E1 (E2
  uniform, emb-only head, B6 arm U-A).

### B5. What each arm trains and where it is selected

Training set of E2 in a fold: all training clips (fit part by the same inner split as B4). Selection of every E2/E3 head (checkpoint,
learning rate schedule, early stopping) uses accuracy on the inner validation clips flagged by E1 (m_1^oof below tau_1^oof), for
every arm including the uniform twin, so no arm has a selection advantage. The per-clip weights apply to the fit part only.

E2 inputs. Input B (primary, as D1): [AST256 embedding, z_1, m_1] with z_1 and m_1 out-of-fold for training and inner
validation clips and from the final E1 for test clips. Input A (sensitivity): AST256 embedding only. E3 input: [AST384 embedding,
z_1, m_1, z_2, m_2] with z_2, m_2 out-of-fold (B3a) for training clips and from the uniform emb-only E2 head for test clips.
All features are standardised with a scaler fit on the fit part.

### B6. Specialist arms (E2; each is trained per fold and seed; "twin" is the uniform arm with the same input and seed)

| Arm | Weights on the fit part | gamma | Input |
|---|---|---|---|
| U | 1 on every clip | 0 | B (twin of the B arms) |
| M0 | 1 on S_2, clips outside S_2 removed (epsilon 0) | 0 | B |
| R05 | 1 on S_2, 0.05 outside | 0 | B |
| R10 (primary specialist) | 1 on S_2, 0.10 outside | 0 | B |
| R20 | 1 on S_2, 0.20 outside | 0 | B |
| F | 1 on every clip | 2 | B |
| T | 1 on medium and hard tier clips, easy clips removed | 0 | B |
| U-A | 1 on every clip | 0 | A |
| M0-A | as M0 | 0 | A |
| R10-A | as R10 | 0 | A |

For E3 the arms are U, M0, R10 and F with S_3 in place of S_2 and input as in B5; E3 is secondary. The loss normalisation of F divides by
the sum of the weights, not by the sum of the focal factors (Part A6). The primary specialist is R10: epsilon 0.1 is the middle of the
stated range [0, 0.2] and was chosen now, before any test fold is scored. No arm is chosen by test results.

### B7. Metrics and tests

Contrast per outer fold f: accuracy of the specialist arm minus accuracy of its twin (U for B arms, U-A for A arms) on the test clips
flagged by E1 (m_1 < tau_1), seeds averaged inside the fold, in points. Macro F1 on the same clips, accuracy on the unflagged clips
and on all clips are reported beside it.
- Fold-level test: Nadeau-Bengio corrected two-sided t with K = 10, df 9, ratio n_test/n_train taken as the mean over folds
  (about 1/9); the exact form is Part A4. Effect sizes: the mean contrast with a 95 percent interval from the same corrected standard
  error and t(9), Cohen's d_z of the 10 fold contrasts, and a 95 percent percentile interval from 10,000 bootstrap resamples of folds.
- Count of folds with contrast >= +0.5 points and >= +1.0 point.
- Holm correction: the primary arm R10 (input B) is one test and is not corrected. All other E2 comparisons (M0, R05, R20, F, T,
  M0-A, R10-A) form one exploratory family with Holm correction; E3 has its own family. A significant exploratory arm creates a
  hypothesis for the next stage and does not change a verdict.

### B8. T1a definitions and the three verdict rules

T1a, strong model = the E2 head with emb-only input (cross-fit heads B3a on training folds, the uniform emb-only head U-A on test
folds), cheap model = E1.
- Spearman: rank correlation of the two models' margins (m_1 against m_2) over all clips of a fold.
- AUROC: difficulty score 1 - m_1 against the label "E2 prediction is wrong", over all clips of a fold.
- On training folds the clips are the out-of-fold training clips of the fold (E1 and E2 cross-fit outputs, pooled over the J
  groups). On test folds they are the test clips (final E1 and final U-A).
- Reported beside them (training folds only, because they need labels): AUROC for the E2 error of d, AUM (negated), EL2N and
  forgetting count, and the Spearman of d against the E2 data-map difficulty.
- Verdict, using the primary statistics above and the wording of Part A1: PASS when Spearman >= 0.3 and AUROC >= 0.6 in at least 7 of 10
  folds on the training folds and on the test folds; STOP when the mean fold Spearman is below 0.15 on both; otherwise
  INCONCLUSIVE (reported with the numbers, no go).
- T1 verdicts, specialist = R10 with input B on UrbanSound8K: proof = contrast >= +1.0 point in at least 7 of 10 folds and NB two-sided p
  < 0.05 with a positive mean (the T1 row of A1); STOP = mean contrast < 0.3 point (the T1 row and A5); D1 stage 1 GO = contrast >=
  +0.5 point in at least 7 of 10 folds with T1a mean test-fold Spearman above 0.3 (A5). A GO or STOP is declared from the primary arm only.
  Between the stop and the proof the verdict is INCONCLUSIVE. T2 to T6 start only on GO. "Holds on TAU-shift and under 0.5 on
  ESC-50 clean" (A1) needs TAU and ESC-50 and is not decided here; TAU is run only if T1a and T1 pass (B10).

### B9. Deviations from PLAN.md, fixed in advance

1. Clean AST embeddings at 400 frames are not in the final-round folder (corruptions only); the tiers are the stage 3b AST at 256 and 384
   frames, which are the E2 and E3 lengths of PLAN.md section 3.
2. The margin is the probability margin, the repo's gate, and not the logit margin of PLAN.md section 3 notation, because every gate
   result in the repo (-27 to -59) is on that margin. The logit margin AUROC is reported once for comparison.
3. Cross-fit heads train for a fixed 30 epochs without a validation set (B3); their size and the rest of the recipe equal the
   protocol 3 head.
4. The twin and all arms select on flagged inner-validation accuracy (B5), where the library selects on macro F1 of the whole inner
   validation part. E1 keeps the library rule.
5. E3 is secondary and has four arms.

### B10. T0 and TAU

T0 (UrbanSound8K) recomputes from the stored per-clip logits (`scores_cnn`, `scores_ast256`, `scores_ast384`, stage 2 `scores`,
`scores_ast_short`, stage 3b/3c and idea 5 stage 2 AST scores, all seeds, clean and field) and compares to stored files and to the
BACKLOG numbers:
- single-model front: mean macro F1 over (seed, fold) of every static system in the stage 3c hull set "new" (statics of stages 2, 3b,
  3c; no int8 or unpadded rows), costs read from stage 3c `analysis/urban/costs.json`; Pareto front on (cost, F1);
- the pre-registered cascade P3 for UrbanSound8K (mn10 w8a8 `esc` head, then the FP32 AST of idea 5 stage 2 `esc` head, rule avg, b50,
  threshold from inner validation) on clean and on the field mix: F1, escalation rate, cost;
- always-average of the same two tiers (cost = sum of both) and the hull rebuilt at the cascade's cost, with and without the
  always-average point (time-sharing of any two statics; `analyze.hull_pair` rule), cascade minus hull difference per fold, fold t p
  and NB p;
- cheapest-correct oracle: per clip the cheapest of the three tiers (mn10 w8a8 esc, AST w4g128@256 esc, AST fp32 esc) that is
  correct, giving oracle accuracy and mean cost, and the per-clip correctness table (`correctness_table.csv.gz`); the stored
  `oracle_k_f1` and `oracle_full_f1` of `selected.csv` are recomputed too;
- gate AUROC (margin of mn10 w8a8 esc against its own errors) and the b30 clean escalation rate;
- Match rule: every compared quantity is rounded to three decimals in the unit stated in BACKLOG (F1 as a fraction, differences in
  points, p values) and must be equal. A mismatch stops the pilot until explained (A1, T0 stop column).
TAU: T0, T1a and T1 on TAU 2020 Mobile are run only if T1a and T1 both pass on UrbanSound8K and the stored TAU embeddings (option B
folder) cover E1 and an AST tier with the official folds; otherwise the report says what extraction is needed.

## Amendment 1. TAU 2020 Mobile (written 24 Oct 2026 after the first 3 UrbanSound8K jobs finished, before any TAU head was trained)

Why. The owner's instruction reads "if T1a or T1 passes, also run the same on TAU 2020 Mobile if its stored embeddings are on
disk". B10 above said "both"; the owner's wording is "or" and replaces it. The TAU jobs may run while UrbanSound8K finishes; their
outputs are not opened until the UrbanSound8K verdicts of T1a and T1 are final, and they are reported only if T1a or T1 passed on
UrbanSound8K. What had been seen of UrbanSound8K at this point: 3 of 30 jobs (seed 0, folds 1 to 3), which show only that the
scripts run. No rule, arm or threshold of Part B changes because of them.

TAU design (everything else is Part B):
- Rows, folds, devices. The universe CSV `~/Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv` (23,035 rows; 14,400 of
  device A), its five location-grouped folds (`folder` column, `tau2020mobile.folds.json` of the option B folder). Device = last
  token of the clip name, location = token 2 of the name. Stored embeddings (clean, read only, option B folder
  `2026-10-23_2133_redesign-idea6-optionB_code-f320295`): E1 `emb_cnn/tau/mn10_w8a8__clean.npy` (960-d), E2 `emb_ast/tau/w4g128__clean.npy`
  (AST 4-bit group-wise at its native 1024 frames, 768-d). There is no second AST length, so E3 is not run on TAU.
- Training uses device-A rows of the four training folds only (protocol of option B). Inner splits and cross-fitting groups are
  by location. Out-of-fold difficulty, tiers, tau_1^oof, S_2, E1, tau_1 (30 percent of the clean device-A inner validation clips),
  arms, input B and A, selection set, seeds 0 to 2: as in Part B.
- Test conditions per fold: clean = the fold's device-A rows; shift = the fold's rows of the other eight devices (each 10 s segment
  counted once per device). Flagged = m_1 < tau_1 inside the condition. Contrasts are reported for flagged shift clips (primary,
  "TAU-shift") and flagged clean clips.
- K = 5, so "7 of 10 folds" of the rules is read as "4 of 5 folds" (70 percent rounded up) everywhere: T1a PASS needs Spearman >= 0.3 and
  AUROC >= 0.6 in at least 4 of 5 folds on training folds and on the shift test rows; T1 proof needs contrast >= +1.0 point in at least
  4 of 5 folds with NB two-sided p < 0.05 and a positive mean on flagged shift clips; STOP stays mean contrast < 0.3 point and
  T1a STOP stays mean fold Spearman < 0.15. The NB ratio is the mean over folds of n_test/n_train in each condition, df 4.
- T1a on TAU test rows: strong model = the UA head (emb-only) evaluated on the same rows, difficulty = 1 - m_1, as for UrbanSound8K.
- T0 for TAU (BACKLOG -52) is limited to the premise and the gate: static F1 of E1 and of E2 on clean and shift (the 0.6 and 7.3 point
  gaps of -52 use the best 8-bit CNN and best AST over all heads, so only the stored-score recomputation of those two is repeated),
  the gate AUROC for "cheap wrong" on clean (0.77) and shift (0.69) for the pre-registered pick of option B. The pick is read from
  option B's `analysis` files.
