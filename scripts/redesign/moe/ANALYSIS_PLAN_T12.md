# T12 micro-pilot: an expert that can change what it computes (UrbanSound8K fold 1)

Written 25 Oct 2026 before any training run and before any test label was read. Status: feasibility signal on one fold. One fold and three seeds cannot prove or refute the
design, and nothing here is a confirmatory test. Its job is to decide whether the full run (section 8 of the README, drafted in the results folder) is worth its compute.

## 1. Question

T1, T1b, T1c and T2b used heads on frozen embeddings. The specialist (trained on the clips the cheap model flags) never beat its uniform twin by 1.0 point. T12 asks the
same question for an expert whose last two AST blocks can change: does a specialist trained on flagged clips beat a uniform twin trained for the same number of steps, on
the flagged test clips?

## 2. What was done before this plan was committed (no label involved)

1. Cost benchmark (`t12_cost.py`): step time and memory with random inputs of the real shapes. No data, no labels.
2. Token cache (`t12_cache.py`): tokens entering block 10 of the FP32 AST at 400 frames (fp16) for all 8,732 UrbanSound8K clips (clean), and for the 873 fold-1 test clips under
   bg 0 dB and the field mix (test assignment, seed 5001, the T1c recipe). The same pass stores the frozen FP32 pooled embedding. Audio is read and mixed; no label is read.
   Checks (`check_cache.json`, 64 clips, run before this plan was committed): the w4g128 pipeline reproduces the stored 400-frame UrbanSound8K embeddings exactly (relative L2
   0.0 median and max), and the fp16 cache reproduces the FP32 pooled output (relative L2 median 1.3e-4, max 2.1e-4; largest token magnitude 27.1, far from the fp16 limit).
3. Cost measurements are in the results folder (`cost/`). They set the batch size (32) and show that the schedules of section 5 take minutes, not hours.

## 3. Fixed inputs (all stored, none changed)

- Test fold 1 (873 clips). Training clips: folds 2 to 10 (7,859). Inner split: `fit_p` (7,090) and `iv_p` (769), grouped by source recording, taken from the pilot file
  `2026-10-24_2013_moe-difficulty-pilot/t1/raw/s<seed>_f1.npz`, seeds 0, 1, 2.
- Flagged training clips: `S2` of that file (out-of-fold margin of E1 below its 30 percent quantile; 30.0 percent of the training clips).
- Flagged test clips of a condition: E1 (mn10 W8A8) probability margin on that condition's logits below `tau1` of `2026-10-25_0052_moe-t1c/urban/raw_strength/s<seed>_f1.npz`
  (the T1c definition, threshold from clean inner validation).
- Conditions: clean (control, not part of the decision), bg 0 dB on every fold-1 clip (U-0dB), field mix of the fold-1 clips (U-field). These two shifted conditions passed the T1c
  strength gate (E2a minus E1 on flagged rows +8.2 and +4.6 points).

## 4. Arms (same input, same budget)

All arms use the FP32 MIT/ast-finetuned-audioset-10-10-0.4593 at 400 frames, the clip only (no E1 logits as input).

- frozen: the repo head (`heads.fit_head`, the T1c uniform-head recipe, selection on flagged inner-validation accuracy) on the frozen FP32 pooled embedding.
- twin: blocks 0 to 9 frozen; blocks 10 and 11 carry LoRA (rank 16, alpha 32) on query, key, value, attention output and both MLP matrices; final LayerNorm and a new linear head
  trained. Each step draws 32 of the 7,090 fit clips uniformly without replacement inside the batch.
- spec: identical, except that each fit clip is drawn with weight 1 if flagged and 0.10 if not (the R10 recipe of T1c, as sampling instead of loss weights, so both arms spend
  exactly the same number of steps and clips).
- Optimizer: AdamW, weight decay 0.01, LoRA lr 3e-4, head and LayerNorm lr 1e-3, 5 percent linear warmup then linear decay to zero, gradient clip 1.0, inverse-count class weights
  as in the repo heads, no label smoothing, no SpecAugment (the cache has no augmentation; this is a stated limit). Same initialisation seed in both arms.
- Checkpoint: evaluated four times per run (every S/4 steps) on the flagged inner-validation clips; the best (strict improvement) is kept. Same rule in both arms.

## 5. Schedule probe (inner validation only)

The twin alone, seed 0, with schedules of 222, 444 and 888 steps (about 1, 2 and 4 passes over the fit clips), each with its own warmup and decay. S* is the smallest
schedule whose best flagged inner-validation accuracy is within 0.5 points of the best of the three. If none is within 0.5 of the 888-step value, S* = 888. S* is then used for
all arms and seeds. No test clip is touched in the probe (`--probe`).

## 6. Scoring and the pre-registered rule

For each seed, flagged-clip accuracy of each arm on each condition; averaged over the three seeds clip by clip. Contrast C = spec minus twin, in points, on the flagged clips.

- Primary: C_primary = mean of C on U-0dB flagged and C on U-field flagged.
- POSITIVE: C_primary at least +1.0 and C positive in both conditions. The full run is recommended on the effect-size evidence.
- CLEARLY NEGATIVE: C_primary at most -1.0. Stop, report as a null (a fine-tuned specialist also loses to its twin).
- INCONCLUSIVE: anything between. The full run then depends on the cost and on the secondary readings below, and the recommendation says so.
- Interval: paired bootstrap over flagged clips (10,000 draws, seed 12), reported for each contrast. The expected width is about plus or minus 2 points per condition (540 and 340
  flagged rows), so a decision on a point estimate near the thresholds is weak by design.

Secondary readings (descriptive, no verdict): clean-flagged C (control); twin minus frozen, spec minus frozen and each minus E1 on the flagged clips; per-seed contrasts and
discordant-pair counts; overall (all-clip) accuracy of each arm.
Validity check V1: twin minus frozen on the shifted flagged clips at least +0.5. If not, fine-tuning two blocks did not change what the expert computes in a useful way, and the
premise of the design is unsupported whatever C says; the report states it.

## 7. Quantization route (descriptive)

For seed 0 and each fine-tuned arm: merge the LoRA, quantize all 72 encoder Linear weights to w4g128 with `QWave/backbone_quant.py` (8-bit activations, post-training), and score
the same fold-1 clips with the trained head. Report the flagged and overall accuracy drop against the FP32 model. QAT is not run in the pilot; it appears in the full plan only
if the post-training drop exceeds 1.5 points.

## 8. Not decided here

k = 1 and k = 4, full-block fine-tuning (the cost table measures them; the pilot trains LoRA on k = 2 only), masked views, distillation from the full-view model, other folds, ESC-50,
TAU. No result of this pilot changes a test bed or a threshold of the full plan.

## 9. Amendment 1 (after the schedule probe, before any pilot run; inner validation only, no test clip read)

The probe of section 5 with the learning rates of section 4 (LoRA 3e-4, head 1e-3) peaked at 53 percent flagged inner-validation accuracy after about 110 steps and fell to 44 to 47
percent later, at every schedule length (222, 444, 888 steps). The same shape appeared at every setting below. Settings tried, all twin, seed 0, flagged inner-validation accuracy
(best of four evaluations), 328 flagged clips:

| LoRA and LayerNorm lr | Head lr | Token keep in training | Steps | Best flagged val acc | At step |
|---|---|---|---|---|---|
| 3e-4 (LayerNorm 1e-3) | 1e-3 | 1.0 | 222 / 444 / 888 | 53.4 / 52.4 / 47.0 | 110 / 111 / 888 |
| 1e-4 | 1e-3 | 1.0 | 444 | 56.7 | 111 |
| 1e-4 | 3e-4 | 1.0 | 444 | 50.6 | 111 |
| 3e-5 | 1e-3 | 1.0 | 444 | 57.3 | 111 |
| 3e-5 | 3e-4 | 1.0 | 444 | 50.3 | 444 |
| 3e-5 | 1e-3 | 1.0 | 111 / 222 / 444 | 51.8 / 57.3 / 57.3 | 54 / 110 / 111 |
| 3e-5 | 1e-3 | 0.5 | 111 / 222 / 444 | 51.2 / 53.7 / 55.5 | 54 / 110 / 111 |

For scale, the frozen head scores 52.7 on the same clips (it was selected on them). Fine-tuning gains about 2 to 3 points over it on the validation clips at the peak, then loses it:
two blocks fit the correlated training clips within a fraction of one pass. This is a finding about the design (it overfits fast on UrbanSound8K without augmentation), not a bug found
by inspection of test results.

Changes, fixed from here on: LoRA and LayerNorm lr 3e-5 (the LayerNorm now shares the LoRA group), head lr 1e-3, no token dropout (the `--keep` option stays at 1.0), S* = 222 steps
(smallest schedule within 0.5 points of the best, section 5), four evaluations every 55 steps with the best kept. Nothing else in sections 1 to 8 changes. Section 6 is unchanged.
The probe used flagged inner-validation labels of seed 0 only; seeds 1 and 2 are untouched until the pilot.
