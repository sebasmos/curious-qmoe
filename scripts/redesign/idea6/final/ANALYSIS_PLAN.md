# Idea 6 final round: real shift on the small CNNs and non-additive corruptions

Written on 24 Oct 2026 before any head was applied to a corrupted clip and before any new embedding was scored.
Branch `redesign/idea6-final`, from `redesign/idea6-stage3c-unpadded` at 545bb54. Nothing here selects a design on
test data: every design below is fixed by an earlier stage or by the task statement. Two paper-review objections
motivate it (`2026-10-23_2141_paper-draft-06-review`, objections 2 and 3): the only real shift so far was run on the
old EfficientNet tier, and the only corruption was additive noise.

## Out of scope

- Quinn: its audio is not public, so no number from it can be reproduced or extended here.
- AudioSet exposure: ESC-50 and UrbanSound8K are Freesound clips and the backbones are AudioSet-pretrained. No
  experiment in this round can test whether a clip was seen in pretraining. Both parts below are shift checks, not
  exposure checks, and the README says so.
- The five-group ESC-50 task and the official-five task. Part B runs on ESC-50 with 50 classes and UrbanSound8K only.

## Systems (all W8A8 fake-quantized backbones, as in Stage 3c)

| name | dataset | embedding | heads (existing, not retrained in part B) |
|---|---|---|---|
| mn20-Q8 | ESC-50 | 5 s clips, exact length | Stage 3c `outputs/mn20_esc50_s{0..4}/<kind>` |
| mn10-Q8 | ESC-50 | exact length | Stage 3b `outputs/mn10_esc50_s{0..4}/<kind>` |
| mn10-Q8 | UrbanSound8K | exact length | Stage 3c follow-up heads retrained on exact-length embeddings (`outputs/mn10unp_urban_s{0..2}/<kind>`; FP32 head from Stage 3c `outputs_unp`) |
| mn20-Q8 | UrbanSound8K | zero-padded to 64,587 samples | Stage 3c `outputs/mn20_urban_s{0..2}/<kind>` (padded, as trained) |
| AST-Q4 | ESC-50 | `w4g128` at 512 frames | Stage 2 `outputs/astshort_esc50_s{0..4}/{esc,4}` |
| AST-Q4 | UrbanSound8K | `w4g128` at 400 frames | Stage 2 `outputs/astshort_urban_s{0..2}/{esc,4}` |

Head kinds: `esc` (FP32), `8`, `4`, `bitnet`. Seeds 0 to 4 (ESC-50), 0 to 2 (UrbanSound8K). Each (seed, fold) head is
applied with the scaler of its own fit portion, as `score_eff.py` does, and every head is re-checked against its
`summary.json` F1 before scoring. UrbanSound8K mn20 stays padded in part B because its heads were trained on padded
embeddings and may not be retrained there; part A trains new heads and uses exact lengths for both CNNs.

## Part A: real shift (E6 and E7) on the small CNNs

Class map (verified in code against `ESC-50-master/meta/esc50.csv` and `UrbanSound8K/metadata/UrbanSound8K.csv`;
the run stops if any pair disagrees): ESC-50 targets 0 dog, 42 siren, 43 car_horn, 44 engine against UrbanSound8K
classIDs 3 dog_bark, 8 siren, 1 car_horn, 5 engine_idling; common labels 0 to 3.

E6, cross-dataset. For each direction (ESC-50 to UrbanSound8K, UrbanSound8K to ESC-50) and each CNN (mn20-Q8,
mn10-Q8, both at exact length on both datasets): a FP32 head (`esc`) is trained on the source dataset's four shared
classes (160 ESC-50 clips, 5 official folds; 3,358 UrbanSound8K clips, 10 folds), protocol 3 recipe, seeds 0 to 2,
standard normalization, inner split grouped by recording, CPU, as Stage 2's E6. Each (seed, fold) head scores its own
test fold (in-domain) and every shared-class clip of the target dataset (shifted). Thresholds come from that fold's
clean inner validation on the source dataset at b05, b10, b20, b30, b50, the margin rule. AST tiers: `w4g128@512`
(the ESC-50 pick's tier; UrbanSound8K clips are at most 398 frames and fit), and Stage 2's `w4g128@1024` and `n4@1024`;
each has its own FP32 head on the same four classes. The tier decides only the cascade accuracy, not the escalation.
The "picked" CNN of a direction is the one picked for its source dataset in Stage 3c (ESC-50 source: mn20-Q8;
UrbanSound8K source: mn10-Q8).

E7, unseen classes. The pick's cheap system (ESC-50: mn20-Q8 with head `4`, b05; UrbanSound8K: exact-length mn10-Q8
with the FP32 head `esc`, b30) scores the other dataset's clips of classes outside its training set, exactly Stage
2's definition: for the ESC-50 task the UrbanSound8K clips of air_conditioner, children_playing, drilling, gun_shot,
jackhammer, street_music (the clip count is taken from the data); for UrbanSound8K the
ESC-50 clips of the 46 classes outside the shared four. ESC-50 also gets mn10-Q8 with head `4` as a second CNN.
Clips of the other dataset use the same extraction and length handling as that CNN's own dataset (exact length).

Reported for both: escalation rate in-domain and shifted at the five budgets; cheap-tier and cascade accuracy
(E6 only; unseen classes have no label to be correct on); detection AUROC, with positives the shifted clips and
negatives the in-domain test clips of the fold, for three gates: the margin gate (score 1 - margin), idea 7's
`noise_lr` gate (logistic regression on the ten audio statistics, C = 1, standardized, trained to separate the
field-mixture noisy clips from clean ones), and the random gate (AUROC 0.5 by construction, stated and not
simulated). For E6 `noise_lr` is fitted per (seed, fold) on all source-dataset clips outside the fold's test fold
(any class, field-mixture validation assignment, seed 5002), because the four-class inner validation sets are too
small (about 25 ESC-50 clips); for E7 it is fitted on the task's inner validation rows as in the main analysis.
Also reported, as in Stage 2: cheap-wrong AUROC of the margin in-domain and shifted.

Tests (fold-level, seeds averaged within fold, two-sided one-sample t-test on fold differences; Holm within each
family; every family is named and its size given):
- A1: shifted minus in-domain escalation, picked CNN per direction, 2 directions x 5 budgets = 10 tests.
- A2: the same for the other CNN, 10 tests.
- A3: detection AUROC of the margin minus that of `noise_lr`, and of the margin minus 0.5, picked CNN per
  direction: 2 directions x 2 = 4 tests.
- A4: the same for the other CNN, 4 tests.
- A5 (E7): unseen minus in-domain escalation at the pick's budget, 2 tasks, 2 tests (the mn10-Q8 second CNN on
  ESC-50 adds 1 test, family A6 of size 1, reported raw).
- A7 (E7): detection AUROC margin minus `noise_lr` and margin minus 0.5, 2 tasks x 2 = 4 tests.
All of Part A is exploratory (a humility check, as in Stage 2); the results are set next to Stage 2's EfficientNet-tier
rows (`e6_cross_dataset.csv`, `e7_ood.csv`) without a test between tiers.

## Part B: non-additive corruptions

Conditions added to Stage 2's thirteen (those are not re-extracted or touched): nine corruptions of the 16 kHz
waveform that `load_clip` returns, in code `scripts/redesign/idea6/final/corrupt.py`.

| kind | levels (mild to severe) | definition |
|---|---|---|
| `reverb` | RT60 0.3, 0.6, 1.2 s | convolution with one synthetic impulse response per RT60: a direct sample of amplitude 1, then Gaussian noise multiplied by exp(-ln(1000) t / RT60) over RT60 seconds, tail energy equal to the direct energy (direct-to-reverberant ratio 0 dB), the whole response scaled to unit energy; seed `[20241024, RT60 in ms]`; output cut to the clip's length; no level correction |
| `clip` | 0.5, 0.25, 0.1 of the clip's own peak | `numpy.clip(x, -q peak, q peak)`; no level correction; a silent clip is unchanged |
| `lowpass` | 4000, 2000, 1000 Hz | 8th-order Butterworth, `sosfiltfilt` (zero phase), at 16 kHz; no level correction |

The mean level change of each condition (dB of RMS against the clean clip) is written to the sidecars. Severity is
the order of the table. Downloads: none.

Extraction: MPS for the networks, at most two extraction processes at once, fixed input lengths (ESC-50 5 s; AST
512 and 400 frames; UrbanSound8K mn20 padded to 64,587 samples; UrbanSound8K mn10 at exact length with Stage 3c's
`--tail-cpu` mode, which compiles one graph on MPS and runs the rare lengths on the CPU), swap watched. CPU work at
`nice -n 10`, 2 threads. Extractors are Stage 3c's `extract_cnn.py` and Stage 2's `extract_ast_short.py`, copied with
the condition parser and the corruption call as the only changes (the diff is in the commit). Before any new
condition is extracted, each copy must reproduce the stored clean embeddings of its system on the first 64 clips.
Idea 7's ten audio statistics (`clip_stats`, unchanged) are computed on the same corrupted waveforms for the
noise-detector gate; the clean rows must equal idea 7's stored clean statistics.

Scoring: `score_eff.py` unchanged, with the heads of the table above, on the nine new files per system. Thresholds
are the clean-inner-validation thresholds of each (seed, fold), exactly as Stage 3c sets them (`thr_rate` on the
clean inner-validation margins), and are applied unchanged to every corrupted condition.

Cascade designs (fixed now):
- ESC-50: Stage 3c's pre-registered pick: mn20-Q8 + head `4` -> AST `w4g128@512` with head `esc`, rule `avg`,
  budget b05.
- UrbanSound8K: Stage 3b's P2 structure with the exact-length cheap tier and the AST-Q4 tier asked for here:
  mn10-Q8 + head `esc` -> AST `w4g128@400` with head `esc`, rule `avg`, budget b30. (Stage 3b's P2 used `w8g64@400`;
  that tier is not extracted here.)

Comparators, per condition (clean plus the nine): the restricted hull (best single static or time-sharing pair at
the cascade's mean cost on that condition, over the statics extracted here: each cheap CNN of the dataset with its four
heads, AST-Q4 with its two heads; not the full hull of Stage 3c, which holds statics that are not extracted on these
conditions); the random gate (exact expected F1 of mixing the cheap and expensive predictions at the cascade's
escalation rate); idea 7's `noise_lr` gate (the same number of escalated clips, chosen by the highest `noise_lr`
score, the detector fitted per (seed, fold) on the field-mixture inner-validation statistics exactly as in Stage 3c
and applied to the condition's statistics); the oracle at the same count (descriptive). Costs are Stage 3c's: front
end + backbone (rescaled to 1024 as there) + head, AST through `ast_bitops_tok` at the tier's token count.

Tests (fold-level t, Nadeau-Bengio and sign-flip p reported as in Stage 3c; Holm uses the fold-level t):
- B1, escalation trend, the only family with a stated direction. For each dataset, the picked cascade's cheap tier
  at its budget, each corruption, the three adjacent steps clean -> level 1, level 1 -> level 2, level 2 -> level 3:
  one-sided (rate rises) paired t-test over fold means. 3 corruptions x 3 steps x 2 datasets = 18 tests, Holm over
  18. A corruption "rises with severity" on a dataset when all three of its steps have Holm p < 0.05 with a positive
  difference. All other cheap tiers, head kinds and budgets get the rates and the same steps without p values.
- B2, cascade against the three comparators (hull, random gate, noise-detector gate) on each of the nine conditions,
  two-sided fold-level t: 9 conditions x 3 comparators x 2 datasets = 54 tests, Holm over 54. Descriptive and
  exploratory; the clean row is reported next to them and is not in the family.
- B3, gate AUROC per condition, no test: the margin's AUROC for cheap-wrong clips (and for cheap-wrong,
  AST-right clips), `noise_lr`'s AUROC for cheap-wrong clips, 0.5 for the random gate, and the AUROC of the margin and
  of `noise_lr` for telling the corrupted clips of a fold from its clean ones.

A gate "fails" in a condition when either indicator holds, and both are reported separately: (i) the cascade does not
beat the random gate at the same escalation count (mean difference at or below zero, or Holm p of B2 at or above 0.05);
(ii) the margin's cheap-wrong AUROC is at or below that of `noise_lr`.

## Checks before results are read

1. Each extractor copy reproduces stored clean embeddings on the first 64 clips (maximum absolute difference
   reported; the MPS kernels are deterministic for equal batch composition).
2. `corrupt.py` unit tests: determinism, clip peak, zero-phase low-pass attenuation, measured RT60 of the impulse
   response within 15 percent of the nominal.
3. Costs: recomputed ESC-50 and UrbanSound8K costs equal Stage 3c's `costs.json` entries.
4. Reproduction on clean: the ESC-50 pick's clean F1, cost, random-gate F1 and noise-gate F1 equal the P3 clean row
   of Stage 3c's `selected.csv` to 1e-9; the UrbanSound8K exact-length mn10 statics equal the follow-up's
   `static.csv` clean rows to 1e-9. The run stops otherwise.
5. Part A: the E6 clean checks, the class map against the meta files, and the head F1 re-check of `score_eff.py`.

## Reporting

The README gives every family with its size, says which results are exploratory, lists every deviation from this
file with the time it was made, and states in a section of its own what the round cannot show (no unexposed audio,
no Quinn). Writing: plain English, no em dashes.
