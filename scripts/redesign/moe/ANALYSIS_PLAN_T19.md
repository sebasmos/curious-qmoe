# T19: precision cascade with low-bit activations inside one stored AST

Written and committed on 26 Oct 2026 on branch `moe/t19` (worktree `~/Documents/qmoe-work/t19`, from `moe/t13` at 94f6af6) before any code of
this experiment exists, before any extraction and before any score of any kind. Source of the design: the round plan
`mit-projects/36-qmoe/blindspot-reviews/reviews/review-fable5-1-max-effort.md`, section 4 (copied verbatim in Part A below), with the rules of its
section 2. Part B fixes every choice the round plan leaves open, and lists the deviations that had to be decided before any data. Amendments are
new dated commits that say what changed and what had already been seen.

Family: T19 is counted in the campaign family of 17 (T1, T1b, T1c, T2b, T7, T9, T12, T13, T14, T15, T16, T17, T18, T3, T5, T19, T20). Holm runs over
the three tasks of each test inside T19; the README reports the family-of-17 number (Holm p times 17, capped at 1) beside it.

# Part A. Round plan section 4.1 to 4.3, verbatim

## 4. T19: precision cascade with low-bit activations (the original thesis, cause removed)

### 4.1 Hypothesis and design

| Item | Content |
|---|---|
| Thesis | One stored 8-bit AST serves two tiers. The cheap tier reads its 4-bit weight view with 4-bit activations (A4) and answers the clips whose margin is above a threshold fixed on clean validation. The 8-bit tier (W8A8) runs only on the rest. Routing by the cheap tier's own margin beats every fixed precision and random time-sharing at equal cost, and escalation rises with acoustic shift. |
| Why it can work now | The cheap pass costs about 27% of the 8-bit pass instead of 56% (section 4.2), and for the same network the margin measures quantization sensitivity (section 1, fact 5). The gate quality for precision pairs was already measured at AUROC 0.74 to 0.99. |
| Experts (primary) | Parent: `w8g128` (8-bit weights, group 128) at the clips' real length (512 frames ESC-50, 400 UrbanSound8K, 1024 TAU) with A8 activations and 8-bit attention probabilities ("W8A8"). Child: the nested 4-bit view `n4g128` (top bits of the stored parent, k+1 planes read as the audit requires) with A4 activations and 8-bit probabilities ("N4A4"). One stored copy. Head: the parent's FP32 head applied to both views, plus a head trained on the N4A4 embeddings (option C showed 3-bit tiers need their own heads: state which is used; the primary uses the view's own head, trained on clean training clips). |
| Activation quantization (new code) | Per-token dynamic asymmetric uniform quantization of every Linear input and of Q, K and V to b_a bits, with percentile clipping (99.9th) and a SmoothQuant-style migration of activation outliers into the weights (per-channel scale s = max|X|^alpha / max|W|^(1-alpha), alpha 0.5, calibrated on 256 clean training clips of the fold, never test clips). Attention probabilities at 8 bits (the 16-bit setting was a 0.6 to 1.4 point luxury; price it at 8). LayerNorm and softmax stay FP32 (standard, priced as elementwise, state it). Settings to implement: A8 (reference, must reproduce the stored embeddings to 1e-5), A6, A4. |
| Static front for the hull | Every static at the same length: W8A8, W8A6, W8A4, W4A8 (= stored w4g128), W4A6, N4A4, W4A4 (group-wise, not nested, as a control), W6A6, W3A8 (stored), and two static mixed-precision models (A4 on the least sensitive half of the blocks by a per-block sensitivity scan on inner validation, A8 elsewhere; and the 4-layer variant), plus the always-average of N4A4 and W8A8 (the average rule at escalation 1.0) and the small CNN tiers (MN10-Q8, MN20-Q8) as cheaper statics. Costs priced per section 2. |
| Gate | Top-1 minus top-2 margin of the cheap tier's head. Budgets 5, 10, 20, 30% chosen on clean inner validation of the fold (no "match" rule, per the T7 lesson). Decision on escalated clips: the 8-bit tier alone (primary, the original design) and the average of both tiers (secondary). |
| Curiosity signal (secondary, the quantized replacement for the old MC-dropout router) | Two cheap passes with different stochastic rounding of activations; the Jensen-Shannon disagreement is the "epistemic uncertainty due to quantization". Cost counted as two cheap passes. Compared with the margin for predicting "4-bit wrong, 8-bit right". |
| Tasks and conditions | ESC-50 (50 classes, primary; the 5 groups secondary), UrbanSound8K, TAU (clean device A, shift on other devices). Conditions: clean, background and pink and white noise at 20, 10, 5, 0 dB, the field mix, the three non-additive corruptions where the plan allows (ESC-50 and UrbanSound8K). Primary test bed: the field mix on ESC-50 (50 classes) and UrbanSound8K, and the device shift on TAU. |

### 4.2 Cost derivation (to be verified by the agent with `QWave/costs.py` before scoring)

| Term | MACs | W8A8 bits | N4A4 bits | W4A6 bits |
|---|---|---|---|---|
| Linear layers | 103.1 G | 64 (6.60 T) | 16 (1.65 T) | 24 (2.47 T) |
| QK^T | 13.6 G | 64 (0.87 T) | 16 (0.22 T) | 36 (0.49 T) |
| PV (probabilities at 8 bits here; the stored runs used 16) | 13.6 G | 64 (0.87 T) | 32 (0.44 T) | 48 (0.65 T) |
| Patch conv (quantize to 8-bit: 0.015 T; FP32 as stored: 0.24 T) | 0.24 G | 0.24 T | 0.24 T | 0.24 T |
| Total at 1024 frames (16-bit PV in the stored W8A8 gives 9.45 T) | | about 8.6 T with 8-bit PV | about 2.5 T | about 3.9 T |
| Ratio r = cheap over 8-bit | | 1 | about 0.27 to 0.29 | about 0.41 to 0.45 |
| Break-even factor 1/(1 minus r) | | | 1.37 to 1.41 | 1.7 to 1.8 |

The cascade at escalation rate f beats time-sharing between the cheap and the 8-bit static only if the net fix rate g among escalated clips exceeds (acc_8 minus acc_cheap) divided by (1 minus r). With r = 0.27 and a 3-point gap, g must exceed 4.1 points among escalated clips. The gate's measured precision at low budgets makes 10 to 15 plausible. The hull also contains the intermediate statics (W4A6, mixed precision), which is where the earlier precision cascades died; that is the risk and it is why they are all in the front.

### 4.3 Pre-registered rules

| Rule | Statement |
|---|---|
| Precondition P1 (accuracy of A4) | On clean inner validation of the first fold of ESC-50 (50 classes) and UrbanSound8K, after calibration, N4A4 is within 8 points of W8A8 and A8 reproduces the stored embeddings to 1e-5. If N4A4 loses more than 8 points, the pre-declared fallback is W4A6 as the cheap tier (r about 0.41) and the rules below apply unchanged. If W4A6 also loses more than 8 points, T19 stops ("activation quantization of this AST is not usable below 8 bits"). |
| Precondition P2 (the gate) | On inner validation, the cheap tier's margin has AUROC at least 0.70 for "cheap wrong and 8-bit right" (the idea 5 range was 0.74 to 0.93). Below 0.70, stop. |
| Primary H1 | One pick per task (budget and decision rule) on clean inner validation. On the primary test bed (field mix for ESC-50 50 classes and UrbanSound8K, device shift for TAU), the cascade beats the equal-cost time-sharing hull of every static and the always-average point by at least 1.0 macro F1 point on two of the three tasks, in a majority of folds, Nadeau-Bengio p below 0.05 after Holm over the three tasks. Report the family-of-17 number beside it. |
| Secondary H2 | The same on clean audio. |
| Secondary H3 (humility) | With the clean threshold held fixed, the escalation rate rises at every noise step on both datasets and on every non-reference TAU device; exceptions listed. |
| Secondary H4 (curiosity) | The rounding-disagreement signal's AUROC for "cheap wrong and 8-bit right" is at least 0.02 above the margin's on two tasks; otherwise the margin stays the signal and the paper says so. |
| Stop S1 | The pick is within 0.3 points of the hull on all three tasks. |
| Stop S2 | A static mixed-precision model (section 4.1) is within 0.3 points of the cascade at equal or lower cost on all three tasks: then the paper's claim becomes "input-adaptive precision does not beat static mixed precision here" and T19 is closed. |
| What a pass lets the paper claim | Exactly the original claim, with the mechanism that works: uncertainty-aware precision routing in one stored quantized network beats any fixed precision and uninformed time-sharing at equal cost, and the routing signal (margin, or rounding disagreement) rises with acoustic shift. Latency stays a bit-op proxy on this hardware (no int4 activation kernels on the M2 Pro). |

# Part B. Choices left open, fixed here before any data

## B1. Deviations decided before any data (with the reason)

| # | Plan text | What T19 does | Why |
|---|---|---|---|
| D1 | Calibration "on 256 clean training clips of the fold" (4.1) and "per fold" (19.4) | One calibration per task, on 256 clean clips of the other datasets: ESC-50 task from 256 UrbanSound8K clips, UrbanSound8K task from 256 ESC-50 clips, TAU from 128 ESC-50 plus 128 UrbanSound8K clips, drawn with `numpy.random.default_rng(19)`. Each of the 256 calibration inputs is filled to the task's input length (5.12 s, 4.0 s, 10.24 s) by concatenating randomly drawn clean clips of the source datasets end to end and cutting at the length (TAU: 128 inputs from ESC-50 clips and 128 from UrbanSound8K clips), so no input is mostly padding. Stored as `calib/<task>.pt` and `calib/<task>.json` (19.4 asks for `calib/<task>_f<k>.json`; one file per task replaces it) | A per-fold calibration changes the backbone in every fold, so every clip would need one extraction per fold (5 times on ESC-50, 10 times on UrbanSound8K, 5 times on TAU): about 80 GPU hours on a shared GPU. Clips of another dataset are never clips of the task, so no test clip of any fold is in any calibration list (asserted by name). This is stricter than the plan on leakage and looser on domain match. |
| D2 | Sensitivity scan "on inner validation" (4.1, 19.7) | The per-block scan runs on the same calibration clips, label free: for block i, the relative squared error of the pooled embedding against the N4A8 pass (n4g128, A8 everywhere) when only block i runs at A4, mean over clips | A scan on fold-1 inner validation would touch clips that are test clips of other folds, and a scan per fold would need one mixed model per fold (the D1 problem again). |
| D3 | "Per-token dynamic asymmetric uniform quantization of every Linear input and of Q, K and V" | Linear inputs: per token, asymmetric (scale (max minus min) / (2^b minus 1) over the token's clipped range, which is widened to contain 0; integer zero point). Q and K: per token and head, symmetric. V: per head channel over the tokens, symmetric (the scale must be constant along the summed axis for an integer P V). P: per query row, unsigned 8 bits, s = max_j P_ij / 255 | An asymmetric Q and K needs a zero-point correction on every score element (two or three extra multiply-adds per element of Q K^T), which under the section 2 pricing costs as much as the A4 Q K^T product itself. A zero point on a Linear input costs one correction per token and output row (priced, B4). |

## B2. Activation quantization (the new code, step 19.2)

1. SmoothQuant migration on the four Linear inputs of each block (query-key-value, attention output, fc1, fc2): s_j = max|X_j|^0.5 / max_i |W_ij|^0.5,
   X_j from the calibration clips of the task, the weight columns multiplied by s_j before weight quantization and the activations divided by s_j.
   For the query-key-value and fc1 inputs the division folds into the LayerNorm, for the attention output into the value projection, for fc2 it is
   one elementwise multiply after GELU (elementwise, unpriced like LayerNorm and GELU). The same smoothed master weights serve every new
   configuration of a task, so the 8-bit parent and its nested 4-bit view are one stored copy.
2. Percentile clipping, static bounds per site from the calibration clips (values after smoothing): Linear inputs clamp to [P_(100-p), P_p] of the
   values, Q, K and V to plus or minus P_p of the absolute values, then the per-token (or per-channel) dynamic range is taken inside the clamp.
   p is 99.9 by default (the plan); per task and activation bit-width, p is chosen from {99.9, 99.99, none} by the lowest mean squared error of the
   pooled embedding against the FP32 model on the calibration clips (label free, no clip of the task): A4 judged on N4A4, A6 on W4A6, A8 on W8A8.
   The choice is written to `calib/<task>.json` and reported.
3. LayerNorm, softmax (exponent and normaliser), GELU, residual additions, biases, the patch convolution (FP32 as stored), the [CLS] and [DIST]
   tokens and position embeddings stay FP32.
4. A8 reference: with smoothing off, clipping off, symmetric Linear inputs and 16-bit P (the stored stage 2 rule), the new code must reproduce the
   stored embeddings to 1e-5 (maximum absolute difference) on 20 clips: `w4g128` and `w8g64` at 512 frames on ESC-50 from the idea 6 stage 2
   folder. A unit test checks the same against the legacy path on random inputs. If it fails, nothing is extracted.
5. Stochastic rounding (H4 only): floor(x / s + u) with u uniform on [0, 1), on every activation input (Linear inputs, Q, K, V, P), two passes
   with generator seeds 1 and 2; the deterministic pass is the scored cheap tier.

## B3. Configurations (names used in every file)

All new configurations: 512 frames on ESC-50, 400 on UrbanSound8K, 1024 on TAU; smoothed master of the task; 8-bit P; group 128 for every
weight grid. `w`: direct round to nearest at that bit-width; `n`: nested child of the 8-bit parent (k + 1 planes read, as the audit requires).

| Name | Weights | Activations | Role |
|---|---|---|---|
| `w8a8` | w8g128 (the stored parent) | A8 | expensive tier (W8A8) |
| `n4a4` | n4g128 (nested view of `w8a8`) | A4 | cheap tier, primary (N4A4) |
| `n4a6` | n4g128 (nested view) | A6 | cheap tier fallback if P1 rejects A4 (one stored copy kept); also a static |
| `w4a6` | w4g128 | A6 | static (W4A6 of the plan) |
| `w4a4` | w4g128 | A4 | static, group-wise control (W4A4) |
| `w8a6`, `w8a4` | w8g128 | A6, A4 | statics |
| `w6a6` | w6g128 | A6 | static |
| `w4a8` | w4g128 | A8 | static (the P8 smoothed twin of the stored w4g128) |
| `n4a8` | n4g128 | A8 | static, and the reference of the sensitivity scan |
| `mp6` | n4g128 | A4 on the 6 least sensitive blocks, A8 on the rest | static mixed precision ("half of the blocks") |
| `mp8` | n4g128 | A4 on the 8 least sensitive blocks, A8 on the 4 most sensitive | static mixed precision ("the 4-layer variant": 4 blocks kept at A8) |
| `n4a4sr1`, `n4a4sr2` | as `n4a4` | A4 with stochastic rounding, seeds 1, 2 | H4 signal only, not a static |

Stored statics added to the hull unchanged (legacy rule: no smoothing, symmetric A8, 16-bit P), each with an own head trained here by the same
recipe (B5) and with its stored FP32-trained head: ESC-50 and UrbanSound8K at 512 and 400 frames from idea 6 stage 2 (`fp32`, `w8g64`, `w4g128`
(the plan's stored W4A8), `n4`, `w4`, `w3g64` (the plan's stored W3A8)); TAU at 1024 frames from option B (`fp32`, `w8g64`, `w4g128`, `n4`).
The small CNN tiers MN10-Q8 and MN20-Q8 with their stored `esc` heads (the T13 loaders). The always-average of the cheap tier and `w8a8`.
A stored static enters a test bed only where its scores exist for that bed (UrbanSound8K short AST: no 20 dB cells; the field mix uses 10, 5 and
0 dB only, so it is complete).

Conditions extracted. Every static: clean and the field mix (ESC-50, UrbanSound8K; the field rows only, stage 2 rule), TAU: every device (clean is
device A, shift the other eight devices). Cheap tier and `w8a8`: also the 12 additive noise cells (bg, pink, white at 20, 10, 5, 0 dB; H3), then
the 9 non-additive corruptions of the final round if GPU time allows (descriptive). SR passes: clean and the primary bed. Same audio for every
configuration (stage 2 mixer, the stored `bg_plan.json` of idea 6 stage 2 for the background clips; the corruption code of the final round).

## B4. Costs (step 19.3)

Primary view: priced bit-ops per clip = fbank front end (the stored stage 3c / option B value of the task) + AST backbone + head. Backbone:
Linear MACs x weight bits x activation bits (per block for `mp6` and `mp8`) + Q K^T MACs x a x a + P V MACs x 8 x a + patch convolution at 32 x 32
+ one 32 x 32 rescale per token, output row and weight group (group 128) + for asymmetric Linear inputs one 32 x 32 zero-point correction per
token and output row. Legacy stored configurations keep their stored rule (16-bit P, A8, no zero point). Head: the FP32 head's MACs at 32 x 32.
Secondary view: the same without the rescale and zero-point terms (unpriced). A cascade costs the cheap pass plus its head on every clip plus the
rate times the `w8a8` pass and its head; the always-average costs both passes; the H4 signal costs two cheap passes. The code must reproduce the
audit's totals at 1024 frames (W8A8 with 16-bit P 9.45 T, W4A8 6.15 T, unpriced, no front end and no head) before scoring. No int4 activation
kernel exists on this Mac, so every A4 and A6 cost is a bit-op proxy; no latency is claimed.

## B5. Heads (step 19.8)

Protocol 3 recipe of `scripts/benchmark.py` (`esc` kind: FP32 MLP 768-640-320-K, the config's optimiser, loss, schedule, early stopping on inner
validation macro F1, grouped inner split by recording, standard scaler fitted on the fit rows), trained on the clean training clips of each
configuration's own embeddings; ESC-50 50 classes (primary) with seeds 0 to 4 and the 5 groups (secondary, descriptive) with seeds 0 to 4,
UrbanSound8K seeds 0 to 2, TAU device A seeds 0 to 2 (location as the group, as option B). The loop is a copy of benchmark.py's for one kind,
calling the same functions in the same order, and is checked equal to a benchmark.py run (same inner-validation F1 and logits) on one
configuration and seed before use. A head scores every condition file of its configuration: test rows by the head of the fold that holds them
out, inner-validation rows by each fold's head. Secondary head: the `w8a8` head of the fold applied to every view (the parent's head transferred).

## B6. Preconditions (step 19.5, fold 1 only)

Extraction for P1: clean, the four configurations `w8a8`, `n4a4`, `w4a6`, `w4a4` plus `n4a6`, every clip except fold 1's test clips, ESC-50 and
UrbanSound8K. Heads only for fold 1 (no test fold is scored).
- P1: mean over seeds of the fold-1 clean inner-validation macro F1. `n4a4` passes if it is within 8.0 points of `w8a8` on ESC-50 (50 classes) and
  on UrbanSound8K. Else the cheap tier is `n4a6` (the nested A6 view; the plan's W4A6 fallback, nested to keep one stored copy) if it is within 8.0
  points on both; else T19 stops. The A8 reproduction check (B2.4) is part of P1.
- P2: on the same rows, AUROC of the chosen cheap tier's margin (low margin = positive) for the target "cheap wrong and `w8a8` right", per seed,
  mean over seeds; P2 passes if it is at least 0.70 on both ESC-50 and UrbanSound8K. Else T19 stops.
- The decision is committed as Amendment 1 before any further extraction.

## B7. Cascade, pick and comparators (steps 19.9, 19.10)

Gate: top-1 minus top-2 softmax margin of the cheap tier's own head. Budgets 5, 10, 20, 30%: for each fold and seed, the threshold is the margin
quantile at the budget on that fold's clean inner-validation rows; clips with margin below it escalate. Decision on escalated clips: `w8a8` alone
(rule B, primary) or the mean of both tiers' probabilities (rule avg, secondary). Thresholds never see test rows (assert on indices).

The pick, one per task: the (budget, rule) with the largest mean over folds and seeds of (cascade minus clean inner-validation hull at the cascade's
inner-validation cost), clean inner validation only; ties go to the cheaper point. The secondary head variant (parent head) is reported, not picked.

Hull: for each task and test bed, the time-sharing hull of every static of B3 (both head kinds where available) and the always-average point, on the
mean macro F1 over every (seed, fold) of the bed (stage 2 `hull_pair`), at the cascade's mean realized priced cost; the hull's fold values are the
interpolation of its two statics' fold values (T13 rule). The hull is built on test F1, which favours the hull.
Also reported: random time-sharing (exact expected F1 of `w8a8` on a random share of clips equal to the cascade's rate, else the cheap tier), a
random gate at the same escalation count (mean of 100 draws), the best single static at no higher cost chosen on inner validation of the same
condition family (field rows of the inner validation for the field mix; for TAU shift, the other-device rows of the inner-validation locations), and
the oracle at the same count.

## B8. Tests and the exact reading of each rule

Unit: the outer fold, seeds averaged within a fold. Macro F1 in points. Nadeau-Bengio corrected paired t, two-sided, ratio n_test / n_train
(mean over folds), sign-flip floor stated (0.0625 at 5 folds, 0.002 at 10), bootstrap 95% interval over folds (10,000 resamples, seed 0) beside the
NB interval. "Majority of folds" = at least 3 of 5 or 6 of 10 folds with a fold difference of at least the stated margin (T13 rule).
- H1: per task, the pick on the primary bed (field mix for ESC-50 50 classes and UrbanSound8K, other-device rows for TAU) minus the hull: mean at
  least +1.0, at least +1.0 in a majority of folds, NB p after Holm over the three tasks below 0.05. H1 passes if at least two tasks pass.
- H2: the same on clean (device A for TAU).
- H3: with the clean threshold of the pick's budget fixed, the escalation rate (fold level) on clean, 20, 10, 5, 0 dB for bg, pink and white on
  ESC-50 and UrbanSound8K, and on each of the eight TAU non-reference devices against device A. A step rises if its mean increase is positive and
  positive in a majority of folds; NB p reported. H3 holds if every step rises; else it holds with the listed exceptions.
- H4: per task, on the primary bed's test rows, AUROC (per fold, seeds averaged) of the Jensen-Shannon divergence between the two SR passes'
  head probabilities, and of the deterministic cheap margin, for "cheap wrong and `w8a8` right". H4 passes if the mean difference (JS minus margin)
  is at least +0.02 and positive in a majority of folds on at least two tasks.
- S1 fires if the pick minus the hull on the primary bed is at most +0.3 on all three tasks.
- S2 fires if on all three tasks a mixed-precision static (`mp6` or `mp8`, own head) with priced cost at or below the pick's mean cost has mean
  primary-bed F1 at least the pick's minus 0.3.
- Family of 17: Holm p times 17, capped at 1, reported beside each H1 and H2 Holm p.

## B9. Order and checks

19.2 code and unit tests; 19.3 cost table from code; 19.4 calibration; B2.4 reproduction check; 19.5 P1 and P2 and Amendment 1; 19.6 full
extraction (ESC-50, UrbanSound8K, then TAU), checks: row counts equal the CSVs, names checksums equal across configurations, no NaN in the rows
extracted; 19.7 scan is done in 19.4 (D2), the two mixed models extracted with the rest; 19.8 heads; 19.9 and 19.10 scoring and analysis;
19.11 README. GPU: one MPS process at a time through `~/Documents/qmoe-work/t19_gpu_run.sh`, each at most about 90 minutes. If time runs out
before TAU completes, H1 is decided on the tasks available and the README says so; nothing is reselected after seeing test scores.


# Amendment 1 (26 Oct 2026, about 19:45 Irish time): preconditions, calibration outcome, the cheap tier

Seen before this amendment: the calibration outputs below (label free, other-dataset clips only); the A8 reproduction check; the fold-1 clean
inner-validation macro F1 of the five P1 configurations (heads trained on fold 1 only with `--no-test`). No test row of any fold has been scored
by any T19 head, except the head-recipe reproduction check on the stored legacy FP32 embeddings (ESC-50, fold 1, seed 0), which B5 requires and
which equals the stored stage 2 numbers to every digit. Nothing below changes a rule; it records the outcomes the plan said would be fixed here.

1. A8 reproduction (B2.4): passes, maximum absolute difference 0.0 against the stored `w4g128` and `w8g64` embeddings (`logs/repro_a8.json`).
   A first run failed (0.42) because the T19 path quantized Q, K and the Linear inputs with torch ops while the stored run used the fused Metal
   kernels: on MPS a handful of rounding-boundary differences in layer 0 grow to a 4% relative difference of the pooled embedding (the legacy
   code run with torch ops differs from itself by up to 2.8). The T19 path now calls the legacy kernels wherever its arithmetic is the legacy one
   (Q and K in every non-stochastic setting, symmetric Linear inputs); asymmetric Linear inputs use torch ops, as B2 says. Commit 8aa8b78.
2. Clip choice (B2.2), relative squared error of the pooled embedding against FP32 on the calibration inputs (none / 99.99 / 99.9):
   ESC-50 A4 0.520 / 0.733 / 0.963, A6 0.197 / 0.504 / 0.855, A8 0.037 / 0.390 / 0.770;
   UrbanSound8K A4 0.443, A6 0.166, A8 0.024 (no clipping); TAU A4 0.549, A6 0.217, A8 0.055 (no clipping).
   Static percentile clipping is worse at every width on every task, so the rule picks no clipping everywhere (`calib/<task>.json`).
3. Mixed models (D2), A4 blocks (0-based) of `mp6` and `mp8`: ESC-50 [1, 2, 3, 5, 7, 8] and [1, 2, 3, 4, 5, 6, 7, 8];
   UrbanSound8K [1, 2, 3, 5, 7, 8] and [1, 2, 3, 4, 5, 7, 8, 9]; TAU [1, 2, 3, 4, 5, 8] and [1, 2, 3, 4, 5, 6, 7, 8].
4. P1 (B6), fold-1 clean inner-validation macro F1, mean over seeds: ESC-50 `w8a8` 96.81, `n4a4` 95.76, `n4a6` 97.01,
   `w4a6` 96.60, `w4a4` 95.49; UrbanSound8K `w8a8` 88.71, `n4a4` 88.61, `n4a6` 85.98, `w4a6` 87.07,
   `w4a4` 87.29. `n4a4` is within 8 points on both: P1 passes, the cheap tier is `n4a4` (N4A4), and the H4 passes are `n4a4sr1` and `n4a4sr2`.
5. P2 (B6): AUROC of the `n4a4` margin for "cheap wrong and `w8a8` right" on the same rows: ESC-50 0.946, UrbanSound8K 0.891.
   Passes (bar 0.70).
6. Scheduling inside B3 (no rule changes): extraction order is every static on clean and the field mix, the SR passes, the cheap tier on the 12
   noise cells, then TAU, then `w8a8` on the noise cells, then the cheap tier on the 9 corruptions. Items after TAU are descriptive and are dropped
   first if GPU time runs out; the README says which ran.
7. The H4 SR files are scored by the cheap tier's own deterministic heads (`t19_heads.py --transfer n4a4`).
