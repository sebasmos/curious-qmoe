# T12 full run: draft design, NOT registered, NOT recommended after the micro-pilot

Drafted 25 Oct 2026 after the micro-pilot came out clearly negative (spec minus twin -5.1 points on the shifted flagged clips). It is written so the owner can overrule the
recommendation without redesigning anything. It becomes a pre-registration only when committed before the first full-run job; until then every number below is a plan.

## Arms (identical to the micro-pilot, Amendment 1 recipe)

frozen (repo head on the frozen FP32@400/512/1024 embedding), twin (LoRA r16 on blocks 10 and 11 plus final LayerNorm and linear head, uniform sampling), spec (the same, flagged
training clips drawn with weight 1, others 0.10). Batch 32 (16 on TAU), 222 steps, lr 3e-5 body and 1e-3 head, four evaluations, best kept on flagged inner-validation accuracy.
k = 2 only. Secondary arms (exploratory, outside the confirmatory family, reported without a verdict): k = 4, spec with distillation to the twin's softmax on the unflagged clips.

## Folds, seeds, inputs

UrbanSound8K 10 folds, ESC-50 5 folds, TAU 5 folds (train on device A rows of the other folds, test rows of the held-out fold, shifted devices for T-shift). Seeds 0, 1, 2 (inner
split and initialisation), accuracies averaged over seeds inside a fold as in T1c. Flagged = E1 (mn10 W8A8) margin below the clean-validation 30 percent threshold on the row's own
condition (T1c definition). Inner split and out-of-fold flags: the stored files of the pilot and T1c where they exist, the T1c recipe for ESC-50.

## Test beds (the strong model clearly stronger on the flagged rows, T1c strength gate)

U-field and U-0dB (bg 0 dB), E-field and E-0dB (bg, pink, white at 0 dB pooled), T-shift. Clean conditions are descriptive controls.

## Confirmatory test, stage 1

Per bed, per fold d = spec minus twin accuracy (points) on the flagged test rows. PASS for a bed: mean d at least +1.0, d at least +1.0 in a majority of folds (6 of 10, 3 of 5),
Nadeau-Bengio corrected two-sided p (n_test/n_train 1/9 for UrbanSound8K, 1/4 otherwise), Holm over the five beds below 0.05. The design passes when at least one non-ESC bed
(U-field, U-0dB, T-shift) passes and no bed shows a significantly worse specialist (Holm p below 0.05 with negative mean). Extra line, not used for the verdict: Bonferroni over
the eight experiments on this question (T1, T1b, T1c, T2b, T7, T9, T13, T12).

## Stage 2, only if stage 1 passes

The system E1 -> specialist on flagged rows (margin gate at the clean-validation threshold, one operating point per bed picked on inner validation) against (a) the lower convex hull
of the fixed models on the cost-accuracy plane (E1, frozen AST, twin, each priced with the repo cost module, the AST quantized to w4g128 after training, QAT only if the
post-training loss exceeds 1.5 points), and (b) the always-average mixture of the experts the system can reach. Same Nadeau-Bengio and Holm. The T13 mixture result (MN10 + AST
w4g128) is shown as a third, descriptive yardstick. A pass needs at least +1.0 point at equal or lower cost on a non-ESC bed against both comparators.

## Family size

Stage 1: 5 tests (Holm). Stage 2: at most 5 beds x 2 comparators (the weaker contrast per bed, Holm over the beds). Project level: eight experiments.

## Compute budget and stopping rule

From the cost measurements and the pilot (two processes share the GPU at half speed each, so they do not add throughput): UrbanSound8K 60 runs at about 2 minutes (2.0 h), ESC-50
30 runs at about 2.5 minutes (1.3 h), TAU 30 runs at about 4 minutes (2.0 h, estimated from 0.625 s per step at batch 16), token caches about 1.6 h (TAU 40 GiB at 1,214 tokens,
about 0.5 to 1 h), post-training quantized scoring for stage 2 about 3 h. Total about 12 h of one GPU stream, cap 16 h. Disk about 66 GiB of caches (UrbanSound8K 3 x 5.9 GiB,
ESC-50 5 x 1.7 GiB, TAU 40 GiB) against 117 GB free.
Stopping rule, futility only (it cannot raise the false-positive rate): run UrbanSound8K folds 1 to 5 with all seeds first (about 1 h); if the mean spec-minus-twin contrast on
both U-field and U-0dB is below +0.3, stop and report a null. Stop also at the 16 h cap.
