# Idea 6 Stage 2: analysis plan

Written and committed on 22 Oct 2026 before any noisy test fold was scored. At commit time the clean
EfficientNet embeddings of both datasets were being extracted, the ESC-50 heads were training, and no cheap-tier
head had been applied to any noisy embedding. Results folder:
`~/Documents/qmoe-results/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/` (this file is copied there).

Thesis: a cheap EfficientNet-B3 tier whose softmax margin sends unsure clips to a quantized AST beats every static
system, and every random time-sharing of two static systems, at the same mean cost, on audio it was not tuned on.

## Systems

Cheap tier. EfficientNet-B3 (ImageNet weights) on the mel image of `QWave/effnet_quant.py` (16 kHz, STFT 2048 /
512, 128 HTK mel bands, magma, 300 x 300), pooled 1536-d features, two backbone configurations: `fp32` and `w8a8`
(per-channel 8-bit weights, 8-bit dynamic activations, commit 9f346ba). Heads trained with protocol 3
(`scripts/benchmark.py`, inner split grouped by recording) on the clean FP32 rebuilt embeddings: kinds `esc`
(FP32), `8`, `4`, `bitnet`; seeds 0 to 4 on ESC-50 (50 classes and 5 super-classes), 0 to 2 on UrbanSound8K. The
same head is applied to both backbones (scaler fitted on the fold's clean FP32 fit portion). 8 cheap systems.

AST tier. Idea 5 Stage 2's embeddings, heads (`esc` FP32 and `4`) and per-clip logits, read only, same seeds. The
expensive tier of a cascade is one of `w4g128`, `n4`, `w4`, `w5`, `w6`, `w8g64`, `fp32` with the FP32 AST head.
Every AST configuration scored in full in a condition, with either head, is a static system for the hull.

Both tiers read identical audio: idea 5 Stage 2's `audio_shift.py` (copied unchanged), same background clips,
offsets and SNRs, same field-mixture assignment (test clips seed 5001, inner-validation clips seed 5002).

## Costs

Cost per clip in bit-ops (MACs x weight bits x activation bits). Primary view `priced`:

- AST: idea 5b's view (ii), `bitops(cfg, 1024)` of `idea5b_common.py` (copied unchanged from fdb2143): Stage 2's
  bit-ops plus one 32 x 32 rescale per token, output row and weight group (per channel: one per row), plus the
  Kaldi fbank front end at 32 x 32 (`ast_fbank_frontend_macs`, mean over the dataset's clip lengths).
- EfficientNet: `fp32` MACs x 32 x 32; `w8a8` MACs x 8 x 8 plus one 32 x 32 rescale per output element of every
  convolution; plus the mel image front end at 32 x 32 (`mel_image_frontend_macs`, mean over clip lengths).
- Heads: `QWave.costs.cost` on the network `scripts/benchmark.py` builds (1536 inputs for EfficientNet heads, 768
  for AST heads), as in Stage 1.
- Cascade on one (seed, fold): cheap system + test escalation rate x (AST tier + its head).

Exploratory view `unpriced`: the same without rescales (Stage 1 and idea 5 Stage 2 counting). Weight storage of
both tiers is reported, not tested.

## Design family and selection (no test fold is read)

Design = (cheap backbone in {fp32, w8a8}) x (cheap head in {esc, 8, 4, bitnet}) x (AST tier in {w4g128, n4, w4,
w5, w6, w8g64, fp32}, FP32 AST head) x (rule in {ast, avg}) x (budget in {b05, b10, b20, b30, b50}): 560 designs
per task. No `match` rule (idea 7 found first-crossing thresholds noisy and optimistic).

Gate: escalate a clip when the cheap head's top-1 minus top-2 softmax probability (raw, no temperature) is below a
threshold. Threshold per (seed, fold), set once on the fold's CLEAN inner-validation split: the reachable
escalation rate closest to the budget (ties: fewer escalations), as Stage 1's `thr_rate`. It is applied unchanged
to every condition. Rule `ast`: escalated clips take the AST head's prediction; `avg`: the argmax of the mean of
both tiers' probabilities.

Selection, per task and cost view: on the clean inner-validation rows only, each design's macro F1 minus the
expected macro F1 of the time-sharing hull of the clean static systems (inner-validation F1) at the design's mean
inner-validation cost, averaged over (seed, fold). The design with the largest mean gain is picked; ties go to the
lower cost. This is Stage 1's procedure with the family above.

## Hull

At a cost c: the best static system costing at most c, or a random mix of one costing at most c with one costing
more at the share that makes the expected cost c, chosen on mean F1 (linear interpolation) among the static
systems of that condition. On test rows the pair is chosen on mean test F1 (this favours the baseline). Its F1 on
each (seed, fold) is the exact expected macro F1 of the random mix (`mix_expected_f1` of idea 5b), or the mean of
200 simulated draws when a split lacks a class (inner validation).

## Hypotheses

Statistics: d = cascade F1 minus comparator F1 per (seed, fold); fold level = d averaged over seeds within each
fold (test clips are shared across seeds); two-sided one-sample t-test over the 5 (ESC-50) or 10 (UrbanSound8K)
fold means. A hypothesis passes when mean d > 0 and its Holm-adjusted p < 0.05. Pair-level p values are reported
alongside and decide nothing.

H1, primary (one test per task, Holm over the 3 tasks). The design picked under the `priced` view, with its
clean thresholds, on the field mixture (test rows take the test assignment's condition in both tiers), against the
hull of every static system on the field mixture at the cascade's mean field-mixture test cost. GO when H1 passes
on all three tasks; otherwise NO GO, with the per-task outcome stated.

H2, secondary: clean replication. The same design and thresholds on the clean test folds against the hull of every
clean static system. Holm over 3.

H3 and H4, secondary cost views. The selection redone with the family restricted to the FP32 cheap backbone (H3)
and to the W8A8 cheap backbone (H4), `priced` costs, tested as H1 on the field mixture. Holm over 3 within each.
If the primary pick already uses one backbone, the matching view repeats it and says so.

## Baselines (for the H1 and H2 designs, every task; Holm over 3 within each comparison)

- Random gate at the same rate: each test clip escalated independently with probability equal to the cascade's
  realized test escalation rate on that (seed, fold); exact expected macro F1; same expected cost.
- Audio-statistic gate: idea 7's `noise_lr` detector, reused so both ideas face the same detector. Logistic
  regression (C = 1, standardized inputs) on idea 7's ten per-clip audio statistics (read from
  `2026-10-22_1105_redesign-idea7-stage1_code-e2af723/audio_stats/`, made with the same corruption code), fit per
  (seed, fold) on the fold's inner-validation clips under the field validation assignment to predict "clip is
  corrupted"; it escalates the k test clips it scores highest, k = the cascade's realized escalation count.
- Oracle at the same count k (fixable clips first, harmful last), and the full oracle (exactly the clips the
  cheap head gets wrong and the escalated decision gets right).
- The best static system at no higher mean cost.

## Humility outputs

- Escalation rate against condition (clean; background, pink, white at 20, 10, 5, 0 dB; field) with the clean
  thresholds, for each task's H1 design and for every cheap system at b10, b20 and b30. It depends on the cheap
  tier only, so it covers every extracted condition, including UrbanSound8K pink, white and 20 dB. Reported with
  the number of steps down in SNR where the rate does not rise.
- Gate AUROC per condition: 1 - margin for "cheap wrong" (every condition) and for "cheap wrong, AST tier right"
  (conditions with the AST tier), per seed pooled over folds, mean over seeds; the same for idea 7's detector
  where its statistics exist.
- Calibration per condition: top-label ECE (15 bins) and mean confidence minus accuracy of every cheap system,
  before and after temperature scaling with T fit per (seed, fold) by NLL on that system's clean inner-validation
  logits; the H1 cascade's output probabilities raw.

## Exploratory (stated families, no verdict)

- E1: the H1 design under every condition where both tiers exist (ESC-50 both label sets: clean, background,
  pink and white at 10, 5, 0 dB, background 20 dB, pink and white 20 dB if Stage 2's scores are complete at
  analysis time, field; UrbanSound8K: clean, background 10 and 0 dB, field) against the hull, the random gate and
  the audio gate. Holm within each comparison over all (task, condition) cells.
- E2: all 560 designs per task on the field mixture against the hull (Holm over 1,680), to show how much the pick
  matters.
- E3: the `unpriced` view, with its own selection, tested as H1.
- E4: W8A8 against FP32 cheap backbone, every head and condition (static F1).

Extractor check, reported and not tested: clean F1 of the rebuilt heads against Stage 1's CSV heads per task and
kind (Stage 1: FP32 head 0.714, 0.739, 0.773), and the W8A8 embedding error.

## Rules fixed now

- Nothing above changes after a noisy test fold is scored; any later analysis is labelled post hoc.
- A condition enters a cascade comparison only when every needed score file is complete for every seed and head.
- If an extraction or run fails, it is rerun from its checkpoint with the same code; a code change is a deviation
  and is reported.

## Amendment 1 (after the Stage 1 audit, before any noisy test fold was scored)

Written on 22 Oct 2026 after the coordinator forwarded the idea 6 Stage 1 audit
(`2026-10-22_1101_redesign-idea6-stage1-audit_code-688a498`). At this point no head had been applied to any noisy
embedding and no test fold of any condition had been compared with anything. Where this section and the text above
differ, this section holds. Changes, each a deviation from commit 14c8504:

1. Static AST on unpadded input. The AST is also run at 512 input frames on ESC-50 (clips are 5 s, 498 fbank
   frames) and 400 on UrbanSound8K (at most 4 s, 398 frames), by cutting the position table to the first time
   positions (`QWave/ast_frames.py`; same patches and positions as the 1024-frame model, without the padding
   patches). Configurations `fp32`, `w8g64`, `w4g128`, `n4`, `w3g64` (named `<config>@512` or `@400`), extracted with
   idea 5 Stage 2's quantizer and audio code on: ESC-50 clean and every noisy condition (20 dB last); UrbanSound8K
   clean, background 10 and 0 dB and the field cells. Heads: `esc` (FP32) and `4`, trained per seed with the same
   protocol 3 recipe on the clean FP32 short embeddings (a backbone with a different input is a different
   backbone, so it gets its own head, as idea 5 trained its heads on the clean FP32 1024-frame embeddings).
   - The short configurations are static systems in every hull (with both heads) and expensive tiers of the design
     family (FP32 head). The family becomes 2 x 4 x 12 x 2 x 5 = 960 designs per task (E2: 2,880).
   - Costs: the priced view counts rescales with the short token count (602 or 470 tokens), the formula of idea 5b's
     view (ii) with tokens as a parameter; the unpriced count must equal the extraction sidecars. The fbank front end
     is the same as at 1024 frames (the padding frames are normalized, not computed).
   - New exploratory output E5: static F1 of each short configuration against the same configuration at 1024 frames,
     per condition (how much the shorter input costs).
   - If the short extraction is not complete for a condition at analysis time, that condition's hull and family use
     what is complete for every seed and head, and the README says which systems were missing. The field mixture
     enters H1 only with every short configuration present.
2. Statistics. The decision stays the fold-level t-test with Holm over 3. Reported next to it for every hypothesis:
   the Nadeau-Bengio corrected t-test on the same fold means (variance times 1/k + n_test/n_train, k - 1 degrees of
   freedom), Holm over 3, and the exact two-sided sign-flip p over the fold means with its floor (2 / 2^k: 0.0625
   with 5 folds, 0.0020 with 10). Any hypothesis where these disagree with the decision is flagged in the README.
   Pair-level p values are neither computed nor reported.
3. Latency. CPU wall clock at batch size 1, 2 threads, one real clip per dataset, 20 warm-up and 100 timed passes,
   median and IQR: the mel image front end, EfficientNet-B3 FP32 (W8A8 runs in fake quantization and is reported
   as a simulation, not a speed), the cheap head, the fbank, the FP32 AST at 1024 and at the short length, the AST
   head; then the escalated path (both tiers in sequence) and the mean per clip at each H1 design's escalation rate.
   No low-bit kernels exist here, so every quantized configuration is timed at FP32. Measured after the GPU jobs end,
   with the load average recorded.
4. Stage 1 history. The README describes Stage 1's selection rule as post hoc (written after its test comparisons
   were read, audit check 4).

Unchanged: the primary hypothesis, the field mixture, selection on clean inner validation under the priced view,
budgets only (no `match`), the hull, idea 7's `noise_lr` audio gate, the random gate, the oracle, the humility
outputs, H2 to H4, E1 to E4.

## Amendment 2 (owner's request via the coordinator, before any noisy test fold was scored)

Written on 22 Oct 2026, still before any head was applied to a noisy embedding. Where this differs from the text
above, this holds.

1. `w4` (per-channel 4-bit, direct round to nearest, not nested) joins the short AST: `w4@512` on ESC-50, `w4@400` on
   UrbanSound8K, clean and every condition of amendment 1, extracted after the running short-AST job with the same
   code. It is scored with the short-AST heads (trained on the clean FP32 short embeddings), is a static system in
   every hull and an expensive tier. The family becomes 2 x 4 x 13 x 2 x 5 = 1,040 designs per task (E2: 3,120).
2. Real-shift humility check, exploratory only, outside the verdict (E6 and E7), on stored clean embeddings.
   - Class mapping, verified against the official metadata (`ESC-50-master/meta/esc50.csv`,
     `UrbanSound8K/metadata/UrbanSound8K.csv`): ESC-50 targets 0 dog, 42 siren, 43 car_horn, 44 engine against
     UrbanSound8K classIDs 3 dog_bark, 8 siren, 1 car_horn, 5 engine_idling; common labels dog, siren, car_horn,
     engine (0 to 3).
   - E6, cross-dataset. For each direction (ESC-50 to UrbanSound8K and back), heads are trained on the source
     dataset's shared-class clips (ESC-50: 160 clips, 5 official folds; UrbanSound8K: 3,358 clips, 10 folds),
     protocol 3, FP32 head (`esc`), seeds 0 to 2: one on the rebuilt FP32 EfficientNet embeddings (applied also to
     the W8A8 ones), one on the clean FP32 1024-frame AST embeddings (applied to `w4g128` and `n4` at 1024 frames;
     the short ASTs differ in length between the datasets, so they do not transfer and are left out). Each
     (seed, fold) head scores its own test fold (in-domain) and every shared-class clip of the target dataset.
     Thresholds come from the source fold's clean inner validation at b05 to b50, rule `ast`. Reported per cell
     (2 directions x 2 cheap backbones x 2 AST tiers x 5 budgets = 40): cheap-tier accuracy, cascade accuracy and
     escalation rate in-domain and cross-domain, and gate AUROC (1 - margin for "cheap wrong" and for "cheap
     wrong, AST right"). One test family: cross-domain minus in-domain escalation rate, fold-level t-test over the
     source folds, 2 directions x 2 backbones x 5 budgets = 20 tests, Holm within.
   - E7, out-of-distribution. The H1 cheap system of each task (its backbone and head, every (seed, fold)) scores the
     other dataset's clips of classes its training set lacks: UrbanSound8K air_conditioner, children_playing,
     drilling, gun_shot, jackhammer and street_music for the two ESC-50 tasks; the ESC-50 clips of the 46 classes
     outside the shared four for UrbanSound8K. Reported: escalation rate on those clips with the clean thresholds
     (the H1 budget and b10, b20, b30) next to the in-domain test fold's rate, and the AUROC of 1 - margin for
     "out of distribution" against the in-domain test clips, next to the same AUROC of idea 7's `noise_lr` score
     (fit as in the main analysis, applied to each clip's clean audio statistics). Two test families, each 3 tasks,
     Holm within: OOD minus in-domain escalation rate at the H1 budget, and margin AUROC minus `noise_lr` AUROC;
     fold-level t-tests over the in-domain folds.
   - `scripts/benchmark.py` refuses a task with fewer classes than folds (a check meant for stratified k-fold, run
     even with predefined folds). It is changed to skip that check when `fold_column` is set; every earlier
     configuration has at least as many classes as folds, so nothing else changes. Applied after the main heads
     finished training.

Note on the ESC-50 5-class task (found while checking the class map, not a change to the plan): the CSV's `class_id`
comes from the map in `data/esc/esc-50-data_processing.py`, which groups targets 0 to 4 and 25 to 29 as Animals,
5 to 9 and 30 to 34 as Natural soundscapes, and so on. It agrees with the official ESC-50 grouping (targets 0 to 9
Animals, 10 to 19 Natural, 20 to 29 Human, 30 to 39 Interior, 40 to 49 Exterior) for 10 of the 50 classes. The
task here stays as the CSV defines it, five groups of ten classes, and the README says so.

## Amendment 3 (coordinator's request, after noisy scoring had started on the ESC-50 50-class task)

Written on 22 Oct 2026 at about 12:10. Noisy test folds of the ESC-50 50-class cheap heads had been scored since
12:03 (scores written, nothing compared or summarized). This task is therefore not pre-registered: it repeats a
fixed procedure on new labels and confirms the procedure only.

- Name in the README: the pre-registered `esc5` task is "ESC-50, five fixed groups of ten classes (not the official
  categories)". It stays as pre-registered.
- New secondary task `esc5off`: ESC-50 with the official five categories, label = target // 10 (0 Animals, 1
  Natural soundscapes and water sounds, 2 Human non-speech, 3 Interior/domestic, 4 Exterior/urban), derived from
  the `label` (target) column. The dataset CSVs and the processing script are not edited: derived training CSVs in
  the results folder carry the new label in `class_id` (benchmark.py's label column), every other column unchanged.
- Run exactly as `esc5`: the same tiers, budgets, rules, selection, hull, baselines, statistics and humility outputs,
  with every head retrained on the new labels with the protocol 3 recipe on the stored clean embeddings, on the CPU:
  cheap heads `esc`, `8`, `4`, `bitnet` on the rebuilt FP32 EfficientNet embeddings; AST heads `esc` and `4` on the
  clean FP32 1024-frame AST embeddings (idea 5 Stage 2's CSV) and on the clean FP32 short embeddings; seeds 0 to 4.
  The 1024-frame AST heads are scored here on idea 5 Stage 2's stored embedding files (read only).
- Reported next to the primary results with raw p values; it is not part of the Holm family of H1 to H4 and does
  not enter the verdict. E6 and E7 are not repeated for it.
