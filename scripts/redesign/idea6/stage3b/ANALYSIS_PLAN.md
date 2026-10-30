# Idea 6 Stage 3b: analysis plan

Written and committed on 22 Oct 2026, before any head of a new system is trained and before any test fold of a new
system is scored. At commit time the new embeddings were being extracted (AST at 256 and 384 frames, EfficientAT
mn04 and mn10); no head existed for them, so no test fold of a new system had produced a single prediction. Results
folder: `~/Documents/qmoe-results/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db/` (this file is copied there).

Why. Idea 6 Stage 2 (`2026-10-22_1120_redesign-idea6-stage2_code-9f346ba`) passed its pre-registered H1 on all three
tasks: a W8A8 EfficientNet-B3 cheap tier (about 0.15 T priced bit-ops per clip) that sends unsure clips to a 4-bit
AST at 512 or 400 frames (about 2.45 or 1.80 T) beats the time-sharing hull of every static system on the field
mixture by 4.4 to 6.0 points. Its audit (`2026-10-22_1623_redesign-idea6-stage2-audit_code-c892c2f`) found no static
system between 0.15 T and 1.8 T, so the hull there is a straight line. This stage fills that range with real static
systems and asks two questions.

## New systems

Audio for every new system: Stage 2's `audio_shift.py` (idea 5 Stage 2's file, unchanged), the same CSVs, background
clips, offsets, SNRs and field assignments (test clips seed 5001, inner-validation clips seed 5002).

AST statics (Stage 2's `extract_ast_short.py`, `QWave/ast_frames.py` and `QWave/backbone_quant.py`; only the
`--frame-offset` option is new and defaults to 0). Configurations `n4` (nested 4-bit, per channel) and `w4g128`
(4-bit, groups of 128):
- `<cfg>@256` and `<cfg>@384`: the first 256 or 384 fbank frames of each clip (2.56 s and 3.84 s; patches cover
  frames 0 to 255 and 0 to 375), 302 and 446 tokens.
- `<cfg>@256x2`, ESC-50 tasks only: the mean of two pooled embeddings, frames 0 to 255 and 256 to 511 (the second
  window padded as the feature extractor pads). UrbanSound8K clips are at most 398 frames, so two windows there cost
  more than `n4@400`, which already sees the whole clip; it is not run.
- Conditions: ESC-50 clean and all 12 noisy conditions (background, pink, white at 20, 10, 5, 0 dB); UrbanSound8K
  clean, background 10 and 0 dB in full and the other field cells for the clips the field assignments need (Stage 2's
  short-AST set; the AST tier exists only there).
- Heads, per length: `esc` (FP32) and `4`, protocol 3 recipe (`scripts/benchmark.py`, inner split grouped by
  recording), trained on the clean FP32 embeddings at that length (FP32 at 256 and 384 frames and the FP32 two-window
  mean are extracted on clean audio for this only) and applied to `n4` and `w4g128`, as Stage 2 did at 512 and 400
  frames. Seeds 0 to 4 (ESC-50 tasks), 0 to 2 (UrbanSound8K). Tags `ast256_<task>_s<seed>`, `ast384_...`,
  `astm2_...`.

Small audio CNN (`QWave/efficientat_mn.py`): EfficientAT MobileNetV3 `mn04` (width 0.4) and `mn10` (width 1.0),
AudioSet weights of release v0.0.1 (`mn04_as_mAP_432.pt`, `mn10_as_mAP_471.pt`; no checksum is published with the
release, so the SHA-256 of the downloaded files is recorded, and the files' sizes equal the release listing),
loaded with `torch.load(weights_only=True)` and strict key matching. The rebuild was checked against EfficientAT's
own code (commit a425fdc) on the same weights: logits, embeddings and log mel equal to 0.0 on random input and six
ESC-50 clips, and the AudioSet top labels are sensible (dog: Bark, Dog; rooster: Chicken, rooster).
- Front end: the model's documented one (32 kHz, pre-emphasis, STFT 1024 / window 800 / hop 320, 128 Kaldi mel bands
  to 15 kHz, log, (x + 4.5) / 5), on the 16 kHz audio both other tiers read, resampled to 32 kHz (torchaudio
  defaults), so nothing above 8 kHz reaches it. Clips run at their own length.
- Embedding: the global average pool of the last 1 x 1 convolution (384-d mn04, 960-d mn10).
- Backbones `fp32` and `w8a8`: every Conv2d and squeeze-excitation Linear with symmetric per-output-channel 8-bit
  weights; inputs 8-bit dynamic with scales an integer kernel can factor out (per pixel for 1 x 1 convolutions, per
  channel for depthwise, per clip for the stem, per row for Linear), as Stage 2's EfficientNet.
- Conditions: clean and all 12 noisy conditions on both datasets (it is a cheap tier, so its humility curve needs
  every condition).
- Heads: `esc`, `8`, `4`, `bitnet`, protocol 3 recipe on the clean FP32 embeddings of that model, applied to both
  backbones (Stage 2's convention). Tags `mn04_<task>_s<seed>`, `mn10_<task>_s<seed>`.

Static systems added to every hull: `cnn:<mn04|mn10>:<fp32|w8a8>:<head>` (16) and `ast:<cfg>@<len>+<head>` (8, plus 4
two-window systems on ESC-50). Stage 2's statics stay: 8 EfficientNet systems, every 1024-frame AST configuration and
every 512- or 400-frame configuration, each with both heads.

## Tasks

ESC-50 50 classes (`esc50`), ESC-50 five fixed groups of ten classes from the repo's `class_id` (`esc5`; these are not
the official categories), UrbanSound8K (`urban`): the family of every primary test. ESC-50 with the official
categories, target // 10 (`esc5off`), run with the same procedure, every head retrained on the derived labels,
reported with raw p values outside the Holm families (as Stage 2's amendment 3).

## Costs

Priced bit-ops (Stage 2's `priced` view) are the primary and only tested cost view:
- AST at L frames: Stage 2's count with the token count of L (`ast_bitops_tok`, one 32 x 32 rescale per token,
  output row and weight group) plus the Kaldi fbank front end (Stage 2's mean over clip lengths) plus the head.
  The two-window mean: two backbone passes, one fbank, one head. About 1.07 T (`n4@256`), 1.69 T (`n4@384`), 2.14 T
  (`n4@256x2`), 1.25, 1.96 and 2.50 T for `w4g128`, before the head.
- CNN: front end (resampler taps, pre-emphasis, window, 1024-point FFT, power, mel weights, log, normalization) at
  32 x 32, mean over the dataset's clip lengths; backbone MACs from forward hooks at each clip's frame count, mean
  over clips; `fp32` at 32 x 32, `w8a8` at 8 x 8 plus one 32 x 32 rescale per output element of every quantized
  layer; plus the head (`QWave.costs.cost` on the network `scripts/benchmark.py` builds for 384 or 960 inputs).
- Cascade on one (seed, fold): cheap system + test escalation rate x (AST tier + its FP32 head).
Elementwise work is left out on every system, as in Stage 2.

## Rules carried over from Stage 2

Gate: escalate when the cheap head's top-1 minus top-2 softmax probability (raw) is below a threshold set per (seed,
fold) once on the fold's clean inner-validation split at the budget's rate (`thr_rate`), applied unchanged to every
condition. Rules `ast` (the AST head decides escalated clips) and `avg` (argmax of the mean of both tiers'
probabilities). Budgets b05, b10, b20, b30, b50 only (no `match`).

Hull at cost c: the best static system costing at most c, or a random mix of one costing at most c with one costing
more at the share that makes the expected cost c, chosen on mean F1 among the static systems of that condition. On
test rows the pair is chosen on mean test F1 (favours the baseline) and its F1 per (seed, fold) is the exact expected
macro F1 of the mix; on inner validation, 200 simulated draws (numpy seed 12346) when needed, as Stage 2.

Statistics: d = cascade F1 minus comparator F1 per (seed, fold); seeds averaged within each fold; two-sided one-sample
t-test over the 5 (ESC-50) or 10 (UrbanSound8K) fold means decides; Nadeau-Bengio corrected t (variance x (1/k +
n_test/n_train)) and the exact sign-flip p with its floor are reported next to it. No pair-level p values. A test
passes when mean d > 0 and its Holm-adjusted fold-level p < 0.05.

## P1: does the Stage 2 result survive a complete front? (primary, Holm over the 3 tasks)

Stage 2's H1 design of each task, as picked and unchanged, with its clean thresholds (ESC-50/50: W8A8 EfficientNet +
FP32 head -> n4@512, ast, b50; five groups: W8A8 + 4-bit head -> n4@512, ast, b50; UrbanSound8K: W8A8 + 8-bit head
-> n4@400, avg, b30), on the field mixture, against the hull of every static system (Stage 2's and the new ones) on
the field mixture at the cascade's mean field-mixture cost. Per task: PASS or FAIL. The cascade's per-(seed, fold) F1
and cost are recomputed from the score files and must equal Stage 2's stored rows; the Stage 2 hull is recomputed as
a check (it must give Stage 2's H1 numbers).

## P2: is a better cheap tier available? (primary, Holm over the 3 tasks)

Design family per task: cheap backbone in {EfficientNet `w8a8`, mn04 `w8a8`, mn10 `w8a8`} x cheap head in {esc, 8,
4, bitnet} x AST tier (FP32 AST head) in {Stage 2's 13: w4g128, n4, w4, w5, w6, w8g64, fp32 at 1024 frames; fp32,
w8g64, w4g128, n4, w3g64, w4 at 512 or 400; and the new n4 and w4g128 at 256 and 384, plus the two-window mean on
ESC-50} x rule in {ast, avg} x budget in {b05, b10, b20, b30, b50}: 3 x 4 x 19 x 2 x 5 = 2,280 designs on the ESC-50
tasks, 3 x 4 x 17 x 2 x 5 = 2,040 on UrbanSound8K.

Selection, per task, on clean inner-validation rows only (Stage 2's procedure): each design's macro F1 minus the
expected macro F1 of the hull of every clean static system (inner-validation F1, Stage 2's and the new ones) at the
design's mean inner-validation cost, averaged over (seed, fold); the largest mean gain is picked, ties to the lower
cost. The pick is tested once on the field mixture against the hull of every static system at its mean
field-mixture cost, same statistics. Per task: PASS or FAIL.

Full family stated: two primary questions, each a family of 3 tests (one per task) with Holm within. Holm over all 6
is reported as a stricter check and decides nothing.

## Secondary (stated families, Holm within each, raw p alongside)

- S1, clean audio: P1 designs and P2 designs, clean test folds, against the hull of every clean static system (two
  families of 3).
- S2, humility of the P2 cheap tier (descriptive): escalation rate with the clean thresholds at the P2 budget and at
  b10, b20, b30 on every extracted condition (clean; background, pink, white at 20, 10, 5, 0 dB; field), with the
  number of steps down in SNR where the rate does not rise; gate AUROC of 1 - margin for "cheap wrong" on every
  condition and for "cheap wrong, AST tier right" where the AST tier exists (per seed pooled over folds, mean over
  seeds). The same table for Stage 2's EfficientNet cheap system, for comparison.
- S3, does the cascade now match the static short AST at lower cost? Comparator: static `n4` at Stage 2's short
  length (512 frames on ESC-50, 400 on UrbanSound8K) with the FP32 head. For the P2 design (family of 3, Holm) and
  the P1 design (family of 3, Holm), field mixture and clean: non-inferiority within 1 point, one-sided fold-level
  t-test of H0: mean d <= -0.01 (t = (mean + 0.01) / (sd / sqrt(k)), p = P(T_{k-1} >= t)). "Matches at lower cost"
  needs Holm p < 0.05 and a mean cascade cost below the static's.
- S4, baselines for the P2 design on the field mixture (families of 3): random gate at the same rate (exact expected
  F1), idea 7's `noise_lr` audio gate at the same count (as Stage 2), and the best static system at no higher cost;
  the oracle at the same count is reported.

## Exploratory (no verdict)

- E1: static F1 (mean over seeds and folds) and priced cost of every static system on every condition: the new
  front.
- E2: every P2 design on the field mixture against the new hull, Holm over all designs of the three tasks.
- E3: W8A8 minus FP32 for each CNN and head, static F1 per condition; the CNNs' W8A8 embedding error.
- E4: P1 against Stage 2's hull (a reproduction check of Stage 2's H1).

## Rules fixed now

- Nothing above changes after a test fold of a new system is scored; any later analysis is labelled post hoc.
- Every new system must be complete (every seed, head and needed condition) before P1 and P2 are computed. The one
  exception is the two-window mean: if its extraction or heads do not finish, it is left out of every hull and family
  and the README says so.
- If an extraction or a run fails, it is rerun from its checkpoint with the same code; a code change is a deviation
  and is reported.
- CPU work runs at nice 10 with 2 threads per job (a latency benchmark shares the machine); extraction runs on MPS.

## Amendment 1 (23 Oct 2026, 08:30, before any UrbanSound8K CNN head was trained or any CNN test fold scored)

Where this differs from the text above, this holds.

1. UrbanSound8K CNN input. The plan ran every clip at its own length. On MPS that compiles one graph per input shape,
   and UrbanSound8K has 354 distinct frame counts (7,326 of 8,732 clips are 4 s). The first clean extraction process
   had grown to 40 GB (33 GB compressed or swapped) at 5,120 of 8,732 rows and filled the swap; the CPU path takes
   0.1 to 0.2 s per forward at 2 threads under the current load. It was stopped and its partial result discarded
   (kept in `logs/failed_first_try/`). Every UrbanSound8K clip is now zero-padded at the end of its 16 kHz waveform
   to the longest clip in the dataset (64,587 samples, 4.04 s, 404 frames at 32 kHz) before the EfficientAT front
   end, so every clip has one input shape. The Stage 2 AST at 400 frames pads its UrbanSound8K input the same way.
   Costs are counted at the padded length (404 frames for every clip), which raises the CNN's UrbanSound8K cost by
   about 12% over the mean unpadded length. ESC-50 is unchanged: every clip is 5 s, so no padding happens there.
2. Motivation and timing, stated plainly. This is a compute and memory constraint, not a result. At this point the
   ESC-50 CNN heads had trained (their clean test-fold F1 is in each run's summary.json, and was looked at), but no
   UrbanSound8K CNN embedding, head or score existed, and no comparison of any kind had been computed.
3. The extraction script also empties the MPS allocator cache after every chunk, which does not change any value.
