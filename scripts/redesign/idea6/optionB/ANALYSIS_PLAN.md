# Idea 6 option B: analysis plan

Written and committed on 23 Oct 2026 before any embedding of TAU 2020 Mobile existed, before any head was trained
and before any test clip was scored. The dataset was downloading when this was written (its meta.csv was read to
build the folds, nothing else). Results folder: `~/Documents/qmoe-results/2026-10-23_2133_redesign-idea6-optionB_code-f320295/`
(this file is copied there). Stage 3b's plan (`scripts/redesign/idea6/stage3b/ANALYSIS_PLAN.md`) and Stage 3c's
(`scripts/redesign/idea6/stage3c/ANALYSIS_PLAN.md`) hold for everything not restated here: the gate, the hull, the
statistics, the head recipe, the costs.

Why. Stages 3b and 3c found that the margin-gated cascade beats the time-sharing hull of two fixed models only
where the cheap tier has headroom below the AST; on ESC-50 and UrbanSound8K the 8-bit EfficientAT mn10 and mn20 sit
within two points of the AST. This stage adds one public task where a small CNN is expected to trail the
transformer by several points and asks whether the cascade passes the cost bar there, on clean audio and under the
dataset's own acoustic shift.

## Data and protocol

- TAU Urban Acoustic Scenes 2020 Mobile, development set (Zenodo 3819968, version 2; the README records the choice,
  the licence, sizes and MD5s). 23,035 segments of 10 s: 14,400 of device A, 1,078 B, 1,077 C, 1,080 each of the
  simulated S1 to S6. Every B, C and S segment is a parallel version of a device-A segment at the same location
  (checked from the names: every one has its A twin). 10 scenes, 514 locations, each location in one scene and one
  city.
- Folds: five location-grouped folds built once over the device-A segments (`build_csv.py`: StratifiedGroupKFold
  on the scene label with the location id as group, shuffle, random_state 0); every other device's segment takes the
  fold of its location. Fold sizes (device A): 2,914 / 2,903 / 2,878 / 2,961 / 2,744; locations 100 to 104 per
  fold; `tau2020mobile.folds.json` has every count. The official DCASE split is recorded in the CSV (`official`
  column) and is not used: it is one split, and three seeds on one split give a t-test with two degrees of freedom.
- Training: every head is trained on device-A rows only (the training CSV is the device-A block of the universe
  CSV, 14,400 rows, same order), leave-one-fold-out over the five folds, protocol 3 recipe (scripts/benchmark.py,
  urban8k config with n_splits 5, inner split 10% grouped by location id = token 2 of the name, seeds 0 to 2).
  Head kinds esc (FP32), 8, 4, bitnet on the CNNs; esc and 4 on the AST. Heads are trained on the clean FP32
  embeddings and applied to the quantized backbones' embeddings, as in Stages 3b and 3c (the scaler is the FP32 fit
  rows' one); the real-int8 tier is the exception (S4).
- Conditions, per fold: `clean` = the fold's device-A rows; `shift` = every B, C, S1 to S6 row whose location is in
  the fold (about 1,700 rows per fold, pooled over the eight devices, each 10 s segment counted once per device
  version); `dev:<d>` = one device's rows (descriptive). No audio is modified. Stage 2's synthetic field mixture is
  not part of the primary families (E4 below).
- Backbones and tiers. Cheap: EfficientAT mn10 (960-d) and mn20 (1920-d), FP32 and W8A8, the 10 s clip at its
  own length (1001 mel frames; nothing padded or cut), extracted with Stage 3c's extractor. Expensive: the AST at
  its native 1024 frames (998 fbank frames of the 10 s clip, padded to 1024 by the feature extractor as in
  pretraining), configurations fp32, w8g64, w4g128, n4, extracted with Stage 2's extractor at `--max-length 1024`.
- Costs: priced bit-ops (Stage 3b's primary view: one 32 x 32 rescale per output element of every quantized
  layer). CNN: front end at 32 x 32 plus the backbone's `backbone_bitops_rescale1024` from the sidecar plus the
  head. AST: the fbank front end at 32 x 32 plus idea 5b's `bitops(cfg, 1024)` at 1214 tokens plus the head. Head
  costs from `costs_idea6.head_costs` at 10 classes. Every clip has the same length, so a system's cost is one
  number; a cascade's cost is the cheap cost plus its escalation rate times the AST tier's cost, per (seed, fold,
  condition).
- Static systems (24): `cnn:<mn10|mn20>:<fp32|w8a8>:<esc|8|4|bitnet>` (16) and `ast:<fp32|w8g64|w4g128|n4>+<esc|4>`
  (8). Every hull is built from all 24 on the condition at hand.
- Cascade family (320 designs): cheap tier in {mn10 W8A8, mn20 W8A8} x cheap head in {esc, 8, 4, bitnet} x AST
  tier in {fp32, w8g64, w4g128, n4} (its FP32 head) x rule in {ast, avg} x budget in {b05, b10, b20, b30, b50}.
  Gate: escalate when the cheap head's top-1 minus top-2 softmax probability is below a threshold set per (seed,
  fold) once on the fold's clean inner-validation rows at the budget's rate; the threshold is applied unchanged on
  every condition.
- Statistics: d per (seed, fold), seeds averaged within each fold, two-sided one-sample t-test over the five fold
  means decides; the Nadeau-Bengio corrected t (ratio = mean over folds of the condition's test rows over the
  training rows) and the exact sign-flip p (floor 0.0625) are reported next to it. A test passes when mean d > 0
  and its Holm-adjusted fold-level p < 0.05. Non-inferiority within 1 point: one-sided fold-level t-test of H0
  mean d <= -0.01.

## Selection (once)

On clean inner-validation rows only: each design's macro F1 minus the expected macro F1 of the hull of all 24
clean statics at the design's mean inner-validation cost (two-system time-sharing, 200 simulated draws, seed
12346), averaged over (seed, fold); the largest mean gain is the pick, ties to the lower cost. One pick for the
whole stage; it is tested in H1 and H2 and used in every secondary.

## Primary (family of two, Holm within)

- H1, clean: the pick on the clean test rows against the hull of every static at the pick's mean clean cost.
  PASS or FAIL.
- H2, shift: the same pick, same thresholds, on the shift rows against the hull of every static on the shift rows
  at the pick's mean shift cost. PASS or FAIL.

## Secondary (Holm within each family, raw p alongside)

- S1, headroom: on clean and on shift, the best cheap W8A8 static (over backbone and head, chosen on mean F1 of
  that condition) against the best AST static (over configuration and head): fold-level difference with its 95%
  CI, and the full static front (E1). This is the premise of option B and is reported first, with no verdict.
- S2, humility of the pick's cheap tier: escalation rate with the clean thresholds on clean, on shift and per
  device, at the pick's budget and at b05, b10, b20, b30; "rises" when a device's rate exceeds the clean rate
  (count out of eight devices per budget). Gate AUROC of 1 - margin for "cheap wrong" on clean, shift and per
  device, and for "cheap wrong, AST tier right" on clean and shift. The same table for every cheap W8A8 tier at the
  pick's head kind (descriptive).
- S3, baselines for the pick at the same count, on clean and on shift (families of two conditions each): the random
  gate at the same rate (exact expected F1); the shift-detector gate: a logistic regression (C = 1, standardized
  inputs) on idea 7's ten per-clip audio statistics, fit per (seed, fold) on the fold's inner-validation device-A
  rows and every other device's rows of those inner-validation locations (target: not device A), ranking the
  condition's rows and escalating the top n where n is the pick's escalation count on that condition; the best
  static at no higher cost. The oracle at the same count is reported.
- S4, real int8 for the pick's cheap backbone (ONNX Runtime static QDQ as Stage 3c: int8 symmetric per-channel
  weights, uint8 per-tensor activations, Percentile calibration as chosen label-free in Stage 3c, one model per fold
  calibrated on 128 clean device-A training clips of the fold drawn with numpy seed 0 from the clips in no seed's
  inner-validation split; node types recorded). Scoring per the Stage 3c audit: the pick's head weights unchanged,
  its StandardScaler refit on fold f's int8 embeddings of the fold's fit rows (training rows minus inner
  validation). (a) Non-inferiority within 1 point of the int8 static against the W8A8 static with the same head,
  on clean and on shift (family of two). (b) The pick with the int8 tier (thresholds re-set on the int8 model's own
  clean inner validation at the budget's rate) against the hull with the int8 statics added at the W8A8 price, on
  clean and shift (raw p, descriptive). The relative embedding error of int8 against FP32 and W8A8 is reported.
- S5, clean inner-validation sanity: the pick's gain over the inner-validation hull and the runner-up's.

## Exploratory (no verdict)

- E1: static F1 and priced cost of every static on clean, shift and per device; W8A8 minus FP32 per CNN and head.
- E2: every design of the family on clean and on shift against the hull, Holm over all designs of both conditions.
- E3: the pick's per-device F1 and escalation.
- E4, only if time allows after everything above is written: Stage 2's synthetic field mixture on the device-A
  rows (background clips from ESC-50 as in Stage 2, seeds 5001 and 5002), the pick against the field hull, raw p,
  not a primary test.

## Rules fixed now

- Nothing above changes after a test clip is scored; any later analysis is labelled post hoc.
- Every system must be complete (every seed, head and condition) before the selection runs. A system that cannot
  be completed leaves every hull and the family and the README says so.
- A failed extraction or run is rerun from its checkpoint with the same code; a code change is a deviation and is
  reported.
- Extraction on MPS (at most two processes); ONNX Runtime, heads, scoring and analysis on the CPU at nice 10 with
  2 threads per job.
- Real data only; every number in the README comes from a file in the results folder.
