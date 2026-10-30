# Idea 6 Stage 3c: analysis plan

Written and committed on 23 Oct 2026 before any head of a new system is trained and before any test fold of a new
system is scored. At commit time the new embeddings were being extracted (EfficientAT mn05, mn20 and dymn04 on both
datasets; mn10 on UrbanSound8K at exact clip lengths); no head existed for them, no real-int8 model had been built,
and no test fold of any new system had produced a prediction. Results folder:
`~/Documents/qmoe-results/2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1/` (this file is copied there).

Why. Stage 3b (`2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db`) found that an AudioSet-pretrained small CNN
(EfficientAT mn10) as a static system removes Stage 2's result (P1 fails on every task) and that a cascade gated by
mn10's margin adds 0.7 to 2.1 points over the hull, passing only on UrbanSound8K (P2). Its audit
(`2026-10-23_1152_redesign-idea6-stage3b-audit_code-6c3aa8b`) left three gaps: (a) the EfficientAT family's other
members (mn05, mn20, dymn04) were not in the front or in the cheap-tier family; (b) UrbanSound8K clips were zero-padded
to 4.04 s before the CNN (amendment 1 of Stage 3b), which moves the mn10 embedding to cosine 0.90 of the unpadded one,
with an unknown effect on F1; (c) mn10's W8A8 numbers are fake quantization, and the one earlier check of a real int8
cheap tier (the EfficientNet, latency study `2026-10-22_1621_redesign-idea6-latency_code-c892c2f`) lost about 6 points.
This stage closes the three gaps. Stage 3b's plan (`scripts/redesign/idea6/stage3b/ANALYSIS_PLAN.md`, commit 5280b3b
and amendment 1 at 6bd2bec) holds for everything not restated here: tasks, audio, costs, the gate, the hull, the
statistics and the P1 and P2 procedures.

## Carried over from Stage 3b (restated)

- Tasks: ESC-50 50 classes (`esc50`), ESC-50 five fixed groups (`esc5`), UrbanSound8K (`urban`) form every primary
  family; ESC-50 official categories (`esc5off`) is secondary with raw p values.
- Audio: Stage 2's `audio_shift.py`, same CSVs, background clips, offsets, SNRs and field assignments (test clips
  seed 5001, inner-validation clips seed 5002). Field mixture: test rows take the test assignment's condition,
  inner-validation rows the validation one.
- Costs: priced bit-ops (one 32 x 32 rescale per output element of every quantized layer), the primary and only
  tested cost view. CNN costs as Stage 3b: front end at 32 x 32, backbone MACs from hooks at the clip's frame count,
  mean over clips; FP32 at 32 x 32, W8A8 at 8 x 8 plus the rescale; plus the head.
- Gate: escalate when the cheap head's top-1 minus top-2 softmax probability is below a threshold set per (seed,
  fold) once on the fold's clean inner-validation split at the budget's rate, applied unchanged to every condition.
  Rules `ast` and `avg`; budgets b05, b10, b20, b30, b50.
- Hull at cost c: the best static system costing at most c, or the random mix of one costing at most c with one
  costing more at the share that makes the expected cost c, chosen on mean F1; exact expected macro F1 on test rows,
  200 simulated draws (seed 12346) on inner validation.
- Statistics: d per (seed, fold), seeds averaged within each fold, two-sided one-sample t-test over the 5 or 10
  fold means decides; Nadeau-Bengio and the exact sign-flip p reported next to it. A test passes when mean d > 0
  and its Holm-adjusted fold-level p < 0.05. Non-inferiority within 1 point: one-sided fold-level t-test of H0
  mean d <= -0.01.
- P1: Stage 2's H1 design, unchanged, against the hull of every static system on the field mixture.
- P2: one design per task picked on clean inner validation from cheap tier in {EfficientNet W8A8, mn04 W8A8, mn10
  W8A8} x 4 heads x 19 (17) AST tiers x 2 rules x 5 budgets, tested once on the field mixture against the hull of
  every static system at its mean field-mixture cost.

Both are recomputed here with the Stage 3b systems only (a reproduction check: the numbers must equal Stage 3b's
`analysis/<task>/selected.csv` to the last digit) and are not tested again.

## New systems

EfficientAT siblings (`QWave/efficientat_mn.py` for mn05 at width 0.5 and mn20 at width 2.0; `QWave/efficientat_dymn.py`
for the dynamic MobileNet dymn04 at width 0.4), AudioSet weights of release v0.0.1 (`mn05_as_mAP_443.pt`,
`mn20_as_mAP_478.pt`, `dymn04_as.pt`; no checksum is published, the SHA-256 of the downloaded files is recorded in the
manifest), loaded with `torch.load(weights_only=True)` and strict keys. Each rebuild equals EfficientAT's own code at
commit a425fdc on the same weights to 0.0 (logits, embeddings and log mel on random input and six ESC-50 clips;
`logs/check_equiv.json`), its 10 s MAC count equals the published one (0.159, 2.053, 0.116 G against 0.16, 2.06,
0.12) and its AudioSet top labels are sensible.
- Front end, embedding (global average pool of the last 1 x 1 convolution: 480-d mn05, 1920-d mn20, 384-d dymn04),
  W8A8 rules, conditions (clean and all 12 noisy on both datasets) and UrbanSound8K padding to 64,587 samples: as
  Stage 3b's mn04 and mn10.
- dymn04's W8A8: every static Conv2d and Linear as the MN; every dynamic convolution's aggregated kernel (a softmax
  mix of four kernels, one per clip) is quantized per output channel to 8 bits per clip at run time, its input with
  the MN's rules. The kernel aggregation (four kernels times the kernel size, 0.9 M MACs per 10 s clip against 116 M
  in the convolutions) is priced at 32 x 32 in both configurations.
- Heads: `esc`, `8`, `4`, `bitnet`, protocol 3 recipe on the clean FP32 embeddings, applied to both backbones; seeds
  0 to 4 (ESC-50 tasks), 0 to 2 (UrbanSound8K). Tags `mn05_<task>_s<seed>`, `mn20_...`, `dymn04_...`.
- The hop and mel-resolution variants of mn10 (`mn10_as_hop_*`, `mn10_as_mels_*`) are not run: they change the front
  end's resolution, not the network, and the family question here is width and dynamic blocks.

Static systems added to every hull: `cnn:<mn05|mn20|dymn04>:<fp32|w8a8>:<head>` (24). Stage 3b's statics stay.

Real int8 mn10 (`scripts/redesign/idea6/stage3c/ort_int8.py`, conda env `qmoe-ort`: onnxruntime 1.31.0, onnx 1.23.2,
torch 2.7.0 for the export only). The feature extractor of `QWave.efficientat_mn.load_mn("mn10")` with hardswish
rewritten as x * clip(x + 3, 0, 6) / 6 and the squeeze-excitation mean as a global average pool (both exact, max
abs difference 5e-7 in float), exported at opset 17 with dynamic batch and frame axes, `quant_pre_process`, then
`quantize_static` in QDQ format: int8 symmetric per-output-channel weights, uint8 per-tensor activations. The
optimized int8 graph runs every operation as an integer kernel (QLinearConv, QGemm, QLinearAdd, QLinearMul,
QLinearSigmoid, QLinearGlobalAveragePool; one QuantizeLinear at the input, one DequantizeLinear at the output); the
node types of the graph that runs are recorded. The front end stays in float (it is priced at 32 x 32 anyway).
- Calibration method: chosen once, label-free, on ESC-50 fold 1: of MinMax, Percentile and Entropy, the one with
  the lowest mean relative embedding error against ORT FP32 on 64 training clips of that fold that are neither
  calibration clips nor in any seed's inner-validation set. The choice is then fixed for every fold and dataset.
- Per fold (5 on ESC-50, 10 on UrbanSound8K): one int8 model calibrated on 128 clean clips drawn with numpy seed 0
  from the fold's training clips that are in no seed's inner-validation split (so the thresholds are set on clips
  the calibration never saw, and the test fold is never seen). The log-mel input is the same front end on the same
  padded (UrbanSound8K) or 5 s (ESC-50) audio as the W8A8 model.
- Embeddings: fold f's model embeds fold f's test clips on clean and all 12 noisy conditions, fold f's
  inner-validation clips (every seed) on clean, and, for the retrained heads, every training clip of fold f on
  clean. Noisy inner-validation rows are not extracted (the analysis never reads them).
- Scoring, two ways. (a) Stage 3b's mn10 heads and their scalers (fitted on the clean FP32 fit rows, as Stage 2's
  `score_eff.py`) applied to the int8 embeddings. (b) Heads retrained per fold on fold f's int8 clean embeddings of
  fold f's training clips, protocol 3 recipe, same seeds and head kinds as the pick needs, restricted to that fold
  (`cross_validation.only_folds`, a new benchmark.py option that leaves every other fold's computation unchanged;
  the inner split of fold f must equal the Stage 3b run's, checked), applied to fold f's int8 rows.
- Cost: the real int8 mn10 is priced as the W8A8 mn10 (the same MACs at 8 x 8 plus one rescale per output element;
  ORT's per-tensor requantization is no dearer than the per-slice one priced there).
- The same build is run for the cheap backbone of a P2 or P3 pick when it is another MN width (mn04, mn05, mn20). A
  DyMN pick gets no real-int8 model (ORT static quantization needs constant kernels); that is reported as a limit.

UrbanSound8K at exact lengths (P5): mn10 FP32 and W8A8 on clean and all 12 noisy conditions with no padding
(`extract_cnn.py --tail-cpu`): within each chunk the clips of one exact length run on MPS in batches of exactly 64
when at least 32 share that length (a short last batch is filled with repeated rows whose outputs are dropped, so
one graph per such length is compiled; on UrbanSound8K only the 4 s length qualifies), every other clip runs on the
CPU at its exact length. On three clips the MPS path and the CPU path agree to 1e-7 and the padded path reproduces
Stage 3b's stored embedding exactly. Costs use the clips' own lengths.

## P3: with the whole EfficientAT family, does a cheap-tier cascade beat the hull? (primary, Holm over the 3 tasks)

Design family per task: cheap backbone in {EfficientNet W8A8, mn04 W8A8, mn05 W8A8, mn10 W8A8, mn20 W8A8, dymn04
W8A8} x cheap head in {esc, 8, 4, bitnet} x AST tier in Stage 3b's 19 (17 on UrbanSound8K) x rule in {ast, avg} x
budget in {b05, b10, b20, b30, b50}: 4,560 designs on the ESC-50 tasks, 4,080 on UrbanSound8K.

Selection, per task, on clean inner-validation rows only (Stage 2's procedure): each design's macro F1 minus the
expected macro F1 of the hull of every clean static system (Stage 2's, Stage 3b's and the 24 new ones) at the
design's mean inner-validation cost, averaged over (seed, fold); the largest mean gain is picked, ties to the lower
cost. The pick is tested once on the field mixture against the hull of every static system (the new statics
included) at its mean field-mixture cost. Per task: PASS or FAIL.

## P4: does the real int8 mn10 hold up? (primary, two parts)

P4a, non-inferiority (families of 3, Holm within): the real-int8 mn10 static, scored way (a) with the head kind of
each task's Stage 3b P2 pick (4-bit on ESC-50/50, FP32 on UrbanSound8K; on the five groups, whose pick uses mn04,
the FP32 head, and the mn04 real-int8 model with its 8-bit head is reported alongside), against the W8A8 mn10 static
with the same head, within 1 point, one-sided: family 1 on clean test folds, family 2 on the field mixture. Per
task and condition: HOLDS or FAILS. Way (b) is reported with the same test next to it. The relative embedding
error of int8 against FP32 and the escalation rate of the P2 design with the int8 tier are descriptive.

P4b, do the Stage 3b P2 verdicts hold with the real tier? (family of 3, Holm within, computed both ways): each
task's Stage 3b P2 design with the cheap tier's scores replaced by the real-int8 model's (thresholds re-set on the
int8 model's own clean inner validation at the budget's rate, nothing else changes), on the field mixture, against
the hull of every static system with the real-int8 statics of that backbone added (at the W8A8 price). Per task:
PASS or FAIL, way (a) and way (b). The verdict "holds" when it equals Stage 3b's (PASS stays PASS, FAIL stays FAIL).
The same substitution for the P3 pick is reported as secondary (S5).

## P5: does padding change UrbanSound8K? (primary, two parts, UrbanSound8K only)

P5a: mn10 at exact lengths against mn10 padded, same Stage 3b heads and scalers, static macro F1 on clean test folds
and on the field mixture, for each (backbone, head) of the eight (FP32 and W8A8 x four heads); d = unpadded minus
padded per (seed, fold). Reported: mean d, fold-level 95% CI, fold t p. Verdict per (backbone, head, condition):
"differs" when |mean d| > 0.5 points and the two-sided fold t p < 0.05, "within" when the 95% CI lies inside
(-0.5, +0.5) points, "undecided" otherwise. The headline is the W8A8 backbone with the FP32 head (the Stage 3b P2
cheap tier) on clean and field.

P5b: Stage 3b's UrbanSound8K P2 design and this stage's P3 pick with the exact-length mn10 as cheap tier (thresholds
re-set on its clean inner validation) against the hull with the exact-length statics added (at their own, lower
cost): PASS or FAIL, raw p (one task, no family). The verdict "holds" when it equals the padded one.

Full set of primary families: P3 (3 tests), P4a clean (3), P4a field (3), P4b (3, way a decides; way b reported),
P5a (the headline pair, no family) and P5b (one test). Holm within each family; nothing across families.

## Secondary (stated, Holm within each, raw p alongside)

- S1, clean audio: the P3 pick on clean test folds against the clean hull (family of 3).
- S2, humility of every pick's cheap tier (P3 pick; the P2 pick with the real-int8 tier, both ways; the P2 and P3
  picks with the exact-length tier): escalation rate with the clean thresholds at the pick's budget and at b05,
  b10, b20, b30 on every extracted condition, steps down in SNR where the rate does not rise; gate AUROC of
  1 - margin for "cheap wrong" on every condition and for "cheap wrong, AST tier right" on clean and field.
- S3, baselines for every pick on the field mixture (families of 3 for the P3 pick): the random gate at the same
  rate (exact expected F1), idea 7's `noise_lr` audio gate at the same count, the best static at no higher cost;
  the oracle at the same count is reported.
- S4, the Stage 3b P2 picks, unchanged, against the enlarged hull (family of 3): does the UrbanSound8K PASS survive
  the new statics?
- S5, the P3 pick with the real-int8 cheap tier (P4b's substitution), both ways (family of 3).
- S6, non-inferiority of the real-int8 mn10 with every head kind, way (a), clean and field (descriptive table).

## Exploratory (no verdict)

- E1: static F1 and priced cost of every static system on every condition, the new front (every width, dymn04,
  the real-int8 mn10, the exact-length mn10).
- E2: every P3 design on the field mixture against the enlarged hull, Holm over all designs of the three tasks.
- E3: W8A8 minus FP32 per new CNN and head on every condition; W8A8 embedding error; real-int8 embedding error.
- E4: P1 and P2 recomputed with Stage 3b's systems only must equal Stage 3b's `selected.csv` (reproduction).
- E5: the calibration-method choice (the three errors) and the node types of the int8 graphs.

## Rules fixed now

- Nothing above changes after a test fold of a new system is scored; any later analysis is labelled post hoc.
- Every new system must be complete (every seed, head and needed condition) before P3 is computed. If a system
  cannot be completed, it is left out of every hull and family and the README says so.
- If an extraction or a run fails, it is rerun from its checkpoint with the same code; a code change is a deviation
  and is reported.
- Extraction runs on MPS (ORT inference on the CPU); CPU work at nice 10 with 2 threads per job.
- Real data only; every number in the README comes from a file in the results folder.
