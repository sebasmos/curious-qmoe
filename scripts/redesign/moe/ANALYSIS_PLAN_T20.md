# T20 analysis plan: a stronger strong tier inside the current two-expert design

Written 2026-10-26 before any training, any extraction and any test-fold score of T20. Branch `moe/t20` (from `moe/t17`, 8f33aaf).
Source of the design: `mit-projects/36-qmoe/blindspot-reviews/reviews/review-fable5-1-max-effort.md`, section 5 (copied verbatim in
sections 1 and 2 below) and the rules of its section 2. Sections 3 to 9 fix every choice the source leaves open. Amendments are new
dated commits that say what changed and what had been seen.

## 1. Hypothesis and design (source section 5.1, verbatim)

| Item | Content |
|---|---|
| Thesis | The cascade's gain is bounded by how much better the strong tier is on exactly the clips the gate escalates. T17 part (c) found a fine-tuned AST (last two blocks plus adapter, all training clips, SpecAugment, mixup, distillation from the FP32 AST) beats the frozen head by 6.6 points on flagged UrbanSound8K clips (fold 1, 3 seeds). With that strong tier, the margin-routed cascade beats the hull, including the hull rebuilt with the fine-tuned model. |
| Experts | Cheap: the stage 3c pick per task (MN10-Q8 or MN20-Q8 with its head, real int8 via ONNX Runtime where available, scaler refit). Strong: the fine-tuned AST at the clips' real length, post-training quantized to w4g128 after fine-tuning (T12 measured 0.3 to 2.8 points PTQ loss; if it exceeds 2 points on inner validation, also report the FP32 fine-tuned tier at its cost), with its own head. |
| Fine-tuning recipe (from T17 part c, unchanged) | Cache the tokens entering block 10; train blocks 10 and 11 plus a LoRA adapter (rank 16) on all training clips of the fold (grouped inner validation for early stopping and the learning rate), SpecAugment and mixup on the cached fbank, KL distillation from the FP32 AST logits (temperature 2, alpha 0.5), batch 32, about 5 epochs (1.5 min per epoch cached). No difficulty weights (the specialist arm failed; the uniform twin is the object here). |
| Gate | The cheap tier's margin, budgets 5, 10, 20, 30, 50% on clean inner validation; average rule primary (the stage 3c pick), strong-alone secondary. |
| Tasks | UrbanSound8K (10 folds, 3 seeds; the cache exists), ESC-50 (50 classes; 5 folds, 3 seeds; 1.7 GiB cache), TAU (5 folds, 3 seeds; 40 GiB cache, run last and only if disk allows). Primary bed: field mix (ESC-50, UrbanSound8K) and device shift (TAU). |
| Comparators | The hull of every static including the fine-tuned AST (quantized and FP32) and the always-average of the cheap tier with the fine-tuned AST; the best single on inner validation of the condition family; random gate at the same count. |

## 2. Pre-registered rules (source section 5.2, verbatim)

| Rule | Statement |
|---|---|
| Precondition P1 | On the primary bed, the fine-tuned quantized AST static beats the frozen w4g128 static by at least 2.0 points on two of three tasks (fold level). If not, the tier is no stronger and T20 stops. |
| Primary H1 | One pick per task on clean inner validation; the cascade beats the equal-cost hull (rebuilt with the fine-tuned statics and their always-average) by at least 1.0 point on two of three tasks, majority of folds, Nadeau-Bengio, Holm over tasks; family-of-17 number beside it. |
| Secondary H2 | The cascade beats the fine-tuned AST alone by cost at equal accuracy within 1.0 point (non-inferiority, one-sided) at a cost at most 50% of the AST alone. |
| Secondary H3 (humility) | Escalation with the clean threshold rises at every noise step and on non-reference devices; exceptions listed. |
| Stop S1 | The pick is within 0.3 points of the hull on all tasks. |
| What a pass lets the paper claim | A two-expert quantized mixture, routed by the cheap expert's own confidence, where the strong expert is a fine-tuned low-bit transformer, beats every fixed model and uninformed time-sharing at equal cost under shift, with escalation that rises with the shift. |

## 3. Decisions on what the source leaves open (fixed now, before any T20 number exists)

### 3.1 Tasks, order, contingencies

- Order: UrbanSound8K (folds 1 to 10), then ESC-50 at 50 classes (folds 1 to 5), then TAU (folds 1 to 5, device A training, location-grouped folds as stored).
- P1 is evaluated as soon as UrbanSound8K and ESC-50 are complete, before any cascade, hull or H1 number is computed for any task.
  If both pass, P1 passes and TAU runs for H1. If both fail, P1 fails (two of three is out of reach) and T20 stops; TAU is not run.
  If exactly one passes, TAU runs and decides P1 (its P1 number is computed first, before any TAU cascade number).
- TAU contingency (pre-declared at the coordinator's request): if TAU cannot run (disk below the 25 GB floor or the machine time is
  exhausted), every "two of three tasks" clause (P1 and H1) becomes "both of the two tasks run", and Holm runs over the two tasks.
- The 40 GiB TAU token cache of the source is not built. The T17 part (c) trainer, whose recipe is used unchanged, does not train from
  cached tokens: it runs the frozen first ten blocks on a cached filter bank at every step so SpecAugment and mixup act on the input
  (`t17c_train.py`). TAU therefore needs only a filter-bank cache of its device-A clips (about 4 GB). The marker
  `~/Documents/qmoe-work/t20-tau-cache-skipped` is written now with this explanation.

### 3.2 Experts

- Cheap tier per task, fixed now:
  - UrbanSound8K: MN10-Q8 (EfficientAT mn10, simulated W8A8, exact length, the FP32 `esc` head), stored as T13's `mn10` for `urban`
    (`2026-10-23_1918_redesign-idea6-stage3c-unpadded_code-cbf6be6/scores_unpr/urban/mn10unp_urban_s{seed}/esc`). This is the stage 3c
    and stage 3b UrbanSound8K pick's cheap tier.
  - ESC-50 (50 classes): MN20-Q8 with the 4-bit head (`2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1/scores_cnn/esc50/mn20_esc50_s{seed}/4`),
    the stage 3c P3 pick's cheap tier.
  - TAU: MN20-Q8 with the 8-bit head (`2026-10-23_2133_redesign-idea6-optionB_code-f320295/scores/mn20_tau_s{seed}/8`), the option B
    pick's cheap tier (stage 3c has no TAU pick).
  - Real int8: not used. Stage 3c showed the real int8 tier needs heads retrained on its own embeddings and then costs about one point;
    the simulated W8A8 tier is what every stored comparator uses. Recorded as a deviation from "real int8 where available".
- Strong tier: the fine-tuned AST at the real length (UrbanSound8K 400 frames, ESC-50 512, TAU 1024), LoRA merged, every encoder Linear
  quantized to w4g128 with `QWave.backbone_quant` (8-bit dynamic activations, 16-bit attention probabilities, exactly the stored frozen
  w4g128 configuration), FP32 final LayerNorm and its own jointly trained linear head ("ft-Q4"). The same model unquantized is "ft-FP32".
- "With its own head" is read as the linear head trained jointly during fine-tuning and selected with the same checkpoint. Step 20.5's
  protocol 3 head is not refit on the fine-tuned embeddings: those embeddings of the training clips are in-sample for the fine-tuned
  blocks, so a head fit on them would see optimistic features. The cheap tier and every frozen static keep their stored protocol 3 heads.
- Frozen comparator of P1: the stored frozen w4g128 AST at the same real length with its stored protocol 3 `esc` head, same seed
  (UrbanSound8K and ESC-50: `emb_ast_short` scores of idea 6 stage 2; TAU: option B `ast_tau_s{seed}/esc/w4g128`).

### 3.3 Fine-tuning (recipe of T17 part c, uniform arm only)

- Code: `t20_train.py`, a generalization of `t17c_train.py` to any task, fold and length, with the `spec` arm removed. Unchanged: LoRA
  rank 16 (alpha 32) on the six Linears of blocks 10 and 11, the final LayerNorm and a new linear head trainable, first ten blocks frozen
  and run on the filter bank every step; SpecAugment (frequency mask up to 24 bins, time mask up to 60 frames), mixup with probability
  0.5 and Beta(0.4, 0.4); class-weighted cross-entropy mixed 0.5/0.5 with KL distillation at temperature 2 from the cross-fitted FP32 AST
  head logits (`E2a_oof` of T1c, cross-fitted inside the fold's training clips only, 1024-frame FP32 AST, as T17 part c); AdamW weight
  decay 0.01, head learning rate 1e-3, batch 32, 300 steps, 5% warm-up then linear decay, evaluation every 50 steps, gradient clip 1.0.
  The only generalization: the time mask's upper bound uses the task's frame count instead of the hard-coded 400.
- Steps: 300 as in T17 part (c) (its amendment 1), not "about 5 epochs" of the source table; the coordinator fixed "recipe unchanged
  from T17 part c". 300 steps are 1.35 epochs on UrbanSound8K, 7.5 on ESC-50 and 0.9 on TAU; early stopping picks the checkpoint.
- Inner split: the stored grouped inner split of the cheap tier's score files (`iv_idx_<fold>` of seed s; it depends on the seed and is
  identical across every stored expert). Fit clips = the fold's training clips (TAU: device A) minus that inner-validation set.
- Early stopping: the checkpoint with the highest macro F1 on all clean inner-validation clips of the fold (T17 part c used accuracy on
  the flagged inner-validation clips because its object was a specialist; the uniform tier's object is the whole distribution).
- Body learning rate: chosen per task from {1e-4, 3e-5} on fold 1, seed 0, by the best-checkpoint clean inner-validation macro F1
  (ties to 1e-4); the chosen run is kept as that task's fold-1 seed-0 model and the rate is fixed for the task.
- Seeds 0, 1, 2 on every task (fine-tuning seed s is paired with stored head seed s of every stored static and of the cheap tier).
- No test clip is read in training: the trainer asserts that no fit or inner-validation row is in the test fold.

### 3.4 Extraction

- `t20_extract.py`, one GPU process per (task, fold), all seeds: the frozen first ten blocks run once per batch in FP32 and once in
  w4g128 (fold- and seed-independent, since they are not trained), then each seed's tail in FP32 and in w4g128. Filter banks at
  evaluation are computed from the audio in float32 with stage 2's mixer (`audio_shift.py`, same background clip lists and directories
  as `idea6/optionC/extract_c.sh`), so the frozen and fine-tuned models see byte-identical inputs. The training cache is float16 as in
  T17 part (c) and is used only for training and for the early-stopping score.
- Rows and conditions: test clips of the fold, clean and under the test field assignment (seed 5001); inner-validation clips of each
  seed, clean and under the validation field assignment (seed 5002). TAU: every test row of the fold (device A and the other devices),
  the inner-validation rows of each seed, and a fixed sample of 600 rows of other devices from the training folds (seeded 20, stratified
  by device) for the shift family's inner validation.
- Checks before any test logit is kept: (a) frozen w4g128 through the split front and tail equals `QuantAST` on the whole model;
  (b) frozen w4g128 on 32 clean and 32 field-cell clips reproduces the stored `emb_ast_short` (TAU: option B) embeddings, relative L2
  below 1e-4; (c) the untrained tail (zero LoRA) reproduces the frozen FP32 model's pooled output.

### 3.5 Systems and costs

- Statics in the hull (each with its stored or own head): the cheap tier; MN10-Q8 and MN20-Q8 with their `esc` heads; frozen AST
  w4g128 and FP32 at the real length (UrbanSound8K and ESC-50); frozen AST w4g128 and FP32 at 1024 frames; ft-Q4; ft-FP32; and three
  always-average points (cheap tier with ft-Q4, with ft-FP32, with the frozen w4g128 at the real length; mean of probabilities on every
  clip, cost of both). A static missing for a condition is left out and listed.
- Costs: priced bit-ops from the stored cost tables (stage 3c `costs.json`, option B `costs.json`, T13 `cost_table`), backbone plus head,
  front end included as stored. ft-Q4 costs the frozen w4g128 at the same length (LoRA merged, same shapes); ft-FP32 the FP32 AST at the
  same length; its linear head is priced as the stored `esc` head (an upper bound). The cascade costs cheap + f x strong, f the realized
  escalation rate of the condition averaged over seeds and folds. The unpriced view (MACs) is reported as secondary.

### 3.6 Cascade, pick and comparators

- Gate: top-1 minus top-2 probability margin of the cheap tier; threshold per (seed, fold) from the cheap tier's clean inner-validation
  margins at budget b in {5, 10, 20, 30, 50}% (`analyze.thr_rate`). Decision on escalated clips: mean of the two tiers' probabilities
  (primary), the strong tier alone (secondary). Non-escalated clips keep the cheap tier's decision.
- One pick per task: the budget with the largest mean over (seed, fold) of [cascade clean inner-validation macro F1 minus the clean
  inner-validation hull of the statics of 3.5 at the cascade's inner-validation cost] (stage 3c's gain rule); ties to the lower budget.
  Primary strong tier ft-Q4, average rule. The secondary rows (strong alone; ft-FP32 as the strong tier) use the same budget rule.
- Hull: stage 2's `hull_pair` (best static or two-static time-sharing at the cascade's realized cost) on the mean test macro F1 of each
  static over (seed, fold) of the condition, then the fold-level hull F1 is the same lam-weighted pair of the statics' fold F1. Built on
  test scores, so it favours the hull. The old frozen hull (without the fine-tuned statics and their always-average points) is reported
  beside it.
- Random gate at the same count: per (seed, fold), the same number of test clips escalated uniformly at random, average rule, 20 draws.
- Best single: among single statics, the highest macro F1 on inner validation of the condition family (field family: inner-validation
  clips under the validation field assignment; TAU shift family: the 600 sampled rows of other devices). Reported, also restricted to
  cost at most the cascade's.

### 3.7 Statistics

- Macro F1 per (seed, fold), seeds averaged within folds; folds are the unit. Nadeau-Bengio corrected t with ratio n_test / n_train,
  two-sided; 95% Nadeau-Bengio interval and a fold bootstrap 95% interval (10,000 resamples, seed 0); exact sign-flip p with its floor
  (2/2^10 UrbanSound8K, 0.0625 with 5 folds). No pair-level p.
- P1 per task: mean over folds of (ft-Q4 minus frozen w4g128) macro F1 on the primary bed at least 2.0 points.
- H1 per task: d = cascade minus hull (fold level) on the primary bed; pass when mean d is at least 1.0, d is positive in a majority of
  folds (6 of 10, 3 of 5), and the Nadeau-Bengio p after Holm over the tasks run is below 0.05. H1 passes when two of three tasks pass
  (both of two under the contingency). Family-of-17 number: min(1, 17 x raw Nadeau-Bengio p) beside each Holm p.
- H2 per task on the primary bed: the cascade's cost at most 50% of ft-Q4's cost and the one-sided 95% Nadeau-Bengio lower bound of
  (cascade minus ft-Q4) above -1.0. Reported on clean audio as well.
- H3: the cheap tier's escalation rate at the pick's clean threshold, test clips, mean over seeds and folds: for bg, pink and white
  separately it must rise at each step clean, 20, 10, 5, 0 dB (UrbanSound8K and ESC-50); on TAU every non-A device's rate must exceed
  device A's. Exceptions listed.
- S1: the stop rule fires when the pick's mean d is below 0.3 points on every task run.
- PTQ check: ft-Q4 minus ft-FP32 on clean inner validation; if the loss exceeds 2 points on a task, the ft-FP32 cascade is reported for
  that task at its cost (it is reported for every task in any case, as a secondary row).

## 4. Machine rules

One MPS process at a time through `~/Documents/qmoe-work/t20_gpu_run.sh` (owner `claude-t20`), one fold (all seeds) per process
(about 30 minutes), swap checked before each process (wait above 9.5 GB used), CPU jobs at `nice -n 10` with two threads. Disk never
below 25 GB free. Results in `~/Documents/qmoe-results/<date>_moe-t20_code-<plan commit>/`.

## Amendment 1 (2026-10-26, about 15:00, before any T20 test score or fine-tuned logit was scored)

Seen at this point: only the fold-1 seed-0 learning-rate probe's inner-validation curve on UrbanSound8K (training log), and a mechanics
test of `t20_analyze.py` on synthetic "fine-tuned" scores copied from the stored frozen AST logits (frozen numbers, known from stage 3c
and T13). That test showed the frozen-tier cascade already sits above the hull on the ESC-50 field mix, so a pass of H1 alone would not
show what fine-tuning adds. Added, as secondary rows only (no rule changes): (1) the cascade with the frozen w4g128 tier at the real
length (TAU: 1024) as strong tier, budget by the same rule, against the rebuilt hull; (2) the fold-level paired contrast of the
fine-tuned cascade minus the frozen cascade, each at its own pick, with the Nadeau-Bengio interval. Also: `t17c_ast.py` sets
`HF_HUB_OFFLINE=1` (the pinned revision is cached; network time-outs only delayed loading).
