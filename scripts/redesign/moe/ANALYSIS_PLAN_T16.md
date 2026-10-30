# Difficulty mixture of experts, T16: a router that reads the strong model's early hidden states

Written and committed on 25 Oct 2026 (branch `moe/t16`, at the T13 head) before any AST feature was extracted and before any router was trained or scored. Machine: M2 Pro, MPS for the
extraction (one process at a time, behind the lock directory `~/Documents/qmoe-work/.mps-lock`), CPU for everything else. No test label is read by the extraction, the head refits or the router
training; the scoring script is the first reader of test labels and is committed before it runs.

## 1. Question and family

Can the AST's own hidden states after block 4, 6, 8 or 10 tell which clips the small CNN gets wrong and the AST gets right, better than the CNN's margin, and does an early-exit system
(small CNN, then b blocks of the AST, then either stop or finish the AST) beat the time-sharing hull at equal cost?
Family: T1, T1b, T1c, T2b, T7, T9, T12, T13, T14, T15, T16, T17, T18, T3, T5 = **15** experiments. Holm runs inside T16 over the 3 task tests of the system criterion; each Holm p is also shown
multiplied by 15 (capped at 1) as an extra line, not used for the verdict.

## 2. Models and data (what is reused, what is new)

- E1 = MobileNet-v3 mn10 W8A8 backbone with the uniform head, the cheap model of T1c. Its embeddings (all cells), its head recipe, `tau1` and its out-of-fold logits on the training clips
  (`z1_oof`) come from the T1c run `2026-10-25_0052_moe-t1c_code-7d7429e` (read only). Splits (`tr`, `te`, `fit_p`, `iv_p`, `S2`) come from the same files.
- Comparator single models (stored embeddings, heads refitted with the T1c recipe and checked against the stored test logits): E2c (AST w4g128 at 400/512 frames), E2b (AST w4g128, 1024 frames),
  E2a (AST FP32, 1024 frames). TAU has E2b and E2a only.
- **S, the strong model of the early-exit path** (new): the FP32 AST at the frame count of the task, 400 (UrbanSound8K), 512 (ESC-50) or 1024 (TAU, then S is E2a itself), with a uniform
  head on the standard pooled embedding ((token 0 + token 1) / 2 after the final LayerNorm), trained as the T1c "U-A" heads. The early-exit path must finish the same graph that produced the
  early features, so it cannot end in the stored 1024-frame E2a on UrbanSound8K and ESC-50. S is also a comparator in the hull. Its relation to E2a (embeddings of the same clips) is reported.
- Extraction (GPU): MIT/ast-finetuned-audioset-10-10-0.4593 at revision f826b80, FP32, `QWave.ast_frames.shorten_ast`, the stage 2 audio reading, `NoiseMaker` and `mix` unchanged
  (`audio_shift.py`), background noise drawn from the other dataset's clips as in stage 2. After the embedding layer plus 4, 6, 8, 10 and 12 blocks (the last after the final LayerNorm) it
  stores two vectors per clip: token 0 ("CLS") and the mean over the patch tokens (tokens 2 onward), 768 each, float32, plus the standard pooled embedding after the last block.
  No tokens are cached. Before the full run, a check compares the 12-block pooled embedding against the stored FP32 embedding on 64 clips (TAU, same graph: relative L2 must be below 1e-4)
  and, with a 1024-frame model, the noisy cells bg 0 dB of UrbanSound8K and ESC-50 against the stored E2a (same noise plan: below 1e-3).
- Conditions extracted. UrbanSound8K (8,732 clips, 400 frames): clean, bg 10 dB, bg 0 dB for every clip, and the other seven field cells for the clips the test (seed 5001) or the
  validation (seed 5002) field assignment gives them. ESC-50 (2,000 clips, 512 frames): clean, bg 0, pink 0, white 0 for every clip, and the other nine field cells likewise.
  TAU (23,035 clips, 1024 frames): clean audio once (the shift is the device, see below). The field mix of a bed is assembled from these cells with `field_assignment`.
- Evaluation conditions, as in T1c: U-field, U-0dB (bg 0 dB), E-field, E-0dB (bg, pink, white 0 dB pooled), T-shift (test rows of devices other than A); descriptive: clean on all three,
  U-bg10, and E-bg0, E-pink0, E-white0 separately.

## 3. The router

Target (training clips only): y_r = 1 if E1 is wrong and S is right, from the out-of-fold logits of the training clips of the fold (E1: stored `z1_oof`; S: 5-fold grouped cross-fit with the
T1c recipe, 30 epochs, same `StratifiedGroupKFold(5, shuffle, seed)` and recording groups). Training clips are clean; the router is fitted on clean features and applied to each condition,
as the T2b gates were, which keeps the clip sets and the fitting recipe comparable with T2b. TAU trains on device A rows only (as T1c) and is tested on every test-fold row.

Input at depth b in {4, 6, 8, 10, final}: [CLS_b, mean_b (1,536 values), E1 logits, E1 margin], standardised on `fit_p`. Model: MLP, one hidden layer, ReLU, dropout 0.2, AdamW, BCE with a
positive-class weight (cost-sensitive). Hyper-parameters (hidden 32 or 128; weight decay 1e-4 or 1e-2; positive weight 1, sqrt(r), r with r = negatives over positives on `fit_p`) and the
epoch (patience 8, at most 60) are chosen by the AUROC on the grouped inner validation `iv_p` (clean). Trained on `fit_p` only. Three seeds per fold as in T1c, averaged inside the fold.

Controls on the identical training rows and clips:
(a) margin alone: score = minus the E1 margin (the probability margin of T1c).
(b) output-only MLP: same MLP and selection on [E1 logits, margin] with no hidden features, the honest output-only comparator.
(c) T2b's output-only gate, the stored T1c `A3h` weight on the non-E1 experts (score 1 minus the E1 weight) on the five T1c conditions, plus its hard `A2h` and `A1` operating points.
(d) the router at the final block (all 12 blocks), the information ceiling of the strong model's hidden state; it has no cost saving and is a reference.

## 4. Stage A, the AUROC test (runs first)

Per task, condition, depth, fold and seed: AUROC of the score for "E1 wrong and S right" on the test rows of the condition (labels computed from the test logits of E1 and S heads), seeds averaged
inside the fold. Gain = router AUROC minus margin AUROC, paired by fold. Primary condition per task: U-field, E-field, T-shift. Every other condition is secondary.
Depth used for the verdict: one per task, chosen on inner validation as the depth in {4, 6, 8, 10} with the largest mean gain over the margin on the held-out `iv_p` rows under the same condition
family as the primary condition (the field mix with the validation assignment; for TAU the shifted rows of the training folds, which the A-only heads never saw). Every depth is reported on every task and condition.
**AUROC pass**: at the chosen depth the gain is at least 0.03 in a majority of folds (6 of 10 UrbanSound8K, 3 of 5 ESC-50 and TAU) and the mean gain is at least 0.03, on at least two of the three tasks.
Nadeau-Bengio intervals for the gain (t = mean / sqrt((1/K + n_test/n_train) s^2), two-sided) are reported.
**Stop rule**: if at every depth in {4, 6, 8, 10} the mean AUROC gain, averaged over the three primary conditions, is below +0.01, the experiment stops, Stage B is run for completeness and labelled descriptive,
and the verdict is STOP. The stricter reading (no task's primary condition reaches +0.01 at any depth) is reported beside it.

## 5. Stage B, the early-exit system

Path: E1 on every clip; the AST embedding layer and b blocks on every clip; the router score s; if s >= theta, the remaining 12 - b blocks and the S head run on the stored block-b state (nothing is
recomputed) and S answers; otherwise E1 answers. Cost per clip (priced bit-ops, repo convention): E1 backbone and head and front end as in `costs.json`; AST front end (the extras of the "+esc" entries
over the stored backbone bit-ops, the same for FP32), patch embedding, b/12 of the encoder (linear plus attention, `transformer_cost` at the frame count of the task, blocks equal), router MLP
(MACs x 32 x 32); escalated clips add (12 - b)/12 of the encoder and the S head. Measured CPU latency (batch 1, 2 threads, FP32 kernels, 50 runs after 10 warm-ups) of the E1-free AST parts
(b blocks, the rest, the router) on one clip is reported where it can be run; E1's latency is not re-measured.

Operating point: theta from an escalation rate on the grid 5, 10, ..., 95, 99 percent (threshold = that quantile of the score on the held-out rows). The rate is chosen on the held-out rows under the
same condition family as the evaluated condition (clean `iv_p`; field with the validation assignment; bg or 0 dB cells on all `iv_p` rows; TAU shift on the shifted rows of the training folds) as the rate that
maximises (system accuracy minus hull accuracy at the system's cost). The whole curve is reported.
Comparators, chosen on those held-out rows and then frozen: the hull of the single models E1, E2c, E2b, E2a, S (time sharing of two models, exact in the weight; singles whose cost exceeds the budget
are excluded) at the system's mean cost: the pair and the weight are the held-out choice, the number reported is the same mix of the two models' test accuracies; the best single at equal or lower cost
is the held-out argmax. Because the condition family's held-out rows are the ones used to choose them, the comparator is not chosen on clean validation. Also reported, descriptive: T2b's stored arms A1, A2h, A3h on
the same rows with their own cost, the fixed S and the fixed E2a, and the test-chosen oracle hull (T13 convention, favours the hull).
**System pass**: on at least two of the three tasks, in the primary condition at the depth chosen in Stage A, system accuracy minus hull accuracy is at least 1.0 point, with the mean over folds
at least 1.0, a majority of folds at least +0.5, and Holm-adjusted Nadeau-Bengio p below 0.05 over the three task tests (family inside T16; x15 line shown).
**Verdict**: PASS needs both the AUROC pass and the system pass. AUROC pass without the system pass is PARTIAL (information exists, the cost does not pay). STOP as in section 4.
Anything else is NO PASS. The two criteria are evaluated at the same chosen depth, so no depth is picked after seeing the test.

## 6. Reporting and honesty clauses

Every depth, task and condition is reported (AUROC for the router, the output-only MLP, the margin and the stored T2b gate; system accuracy, cost and the contrasts at the chosen rate and over the
whole rate grid). Accuracy is the primary metric; macro-F1 of the primary system beside it. Costs and latency are labelled measured or priced. Known weaknesses stated in advance: (1) the router is
fitted on clean clips and applied to noisy ones, so its score is shifted by the noise; (2) the held-out rows used to choose rates and comparators are about 10 percent of the training clips; (3) the
hull is time sharing of five points, so a gain over it is a gain over a near-straight line; (4) S differs from the stored E2a on UrbanSound8K and ESC-50 (shorter input), so numbers are not
comparable with T1c's E2a rows except through the reported E2a comparator; (5) no independent reviewer pass is run inside this task.

## 7. Amendment 1 (25 Oct 2026, 14:08, before any router was trained or scored; FP32 features of UrbanSound8K were partly extracted, nothing read)

The priced FP32 AST is 20 times the cost of the 4-bit AST the hull already contains (UrbanSound8K: FP32 at 400 frames 4.5e13 bit-ops, w4g128 at 400 frames 2.1e12), so an FP32 early-exit
system sits far above the cheap end of the hull and the comparison is lopsided by construction. The owner's design allows the repo's 4-bit group-wise variant "only if cheap"; it is cheap here
(`QuantAST._layer` exposes the blocks, extraction is the same pass). Added, pre-registered, same criteria:

- **W4 variant.** The same extraction with `QuantAST` at `w4g128` (weights 4-bit, group 128, the configuration of E2b/E2c), features under `feat_w4g128/`. Its strong model is the stored one of the same graph:
  E2c (400/512 frames) on UrbanSound8K and ESC-50, E2b (1024 frames) on TAU, with the stored T1c heads and the stored T1c out-of-fold logits (`raw_oof`), so no new head. Target, router, controls, Stage A and Stage B are
  as in sections 3 to 5 with that model in place of S. A check compares the 12-block embedding of 64 clips with the stored w4g128 embedding (relative L2 near 0). Costs: fbank, patch embedding and head as
  for FP32; the encoder cost per block is (the stored `w4g128@L+esc` total minus fbank, patch embedding and head) / 12, so the single-model prices are the ones the hull uses.
- The FP32 variant stays the pre-registered primary (the owner's design). The verdict label is taken from FP32; a W4 result is reported next to it and called by name ("PASS-W4" and so on), never merged.
- Multiplicity: Holm over the **6** system tests (3 tasks x 2 variants); the AUROC pass is a count rule and is evaluated per variant.
- The hull in both variants contains E1, E2c, E2b, E2a and S (the FP32 short-input AST), so the W4 system is also tested against the FP32 model that costs more.
- Seeds, folds, depth selection on held-out rows, operating-point selection and every other rule are unchanged.

## 8. Amendment 2 (25 Oct 2026, before the full head and router runs; written after a one-fold look)

The code was tested end to end on UrbanSound8K fold 1, seed 0, conditions U-0dB, U-bg10 and U-clean (the debugging run; its outputs are kept in `debug_fold1/` and are not part of any reported number).
That run showed the 1,536-wide hidden-feature routers scoring below the margin on test AUROC (about 0.64 to 0.69 against 0.81 at 0 dB), and the hidden-feature routers not above the output-only MLP on clean
held-out rows (0.745 to 0.808 against 0.804). One reading is that 1,536 inputs overfit about 400 positive training clips. So that the question is not decided by capacity alone, a regularised router is
added: kinds `p4, p6, p8, p10, p12`, the same MLP, grid and selection on [64 PCA scores of the depth's 1,536 standardised hidden features (PCA fitted on `fit_p`), E1 logits, E1 margin].
These kinds are **exploratory**: motivated by a look at fold 1 data, they are not used for the depth choice, the AUROC pass, the stop rule or the verdict, which stay on `d4, d6, d8, d10` as registered in
sections 3 to 5. They are reported in full beside them.
