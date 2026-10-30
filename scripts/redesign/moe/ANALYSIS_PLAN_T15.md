# Difficulty mixture of experts, T15: experts that see different views of the same audio

Written and committed on 25 Oct 2026 (branch `moe/t15`, from the T13 head `94f6af6`) before any view was extracted, any head was trained on a new view and any test clip was scored.
T18 (scale and modality hierarchy) shares this extraction and has its own file, `ANALYSIS_PLAN_T18.md`. Machine: Apple M2 Pro. Extraction and the waveform CNN run on the MPS GPU
through the lock directory `~/Documents/qmoe-work/.mps-lock` (one process at a time, released between processes); heads, scoring and analysis run on the CPU at `nice -n 10`.

## 1. Question and family

Experts that differ in what they see (a frequency band, a time-frequency tiling, the time domain, a second pretrained model) can make errors that overlap less than two
experts reading the same mel image. Does a uniform mixture that includes such views beat the equal-cost hull and the best single model under shift, and is the new experts' error
correlation with the small CNN lower than the AST's?

Family. T15 is one of 15 pre-registered experiments of the difficulty mixture-of-experts question: T1, T1b, T1c, T2b, T7, T9, T12, T13, T14, T15, T16, T17, T18, T3, T5.
T15 has **3 primary tests** (ESC-50 with 50 classes, UrbanSound8K, TAU shift, section 8). Holm is applied with family size 15: the 3 p-values of T15 are sorted and compared with
0.05/15, 0.05/14, 0.05/13, because the other 12 experiments are not decided here and their p-values enter as 1. This is the decision rule. The README also shows Holm over the 3 tests
alone (thresholds 0.05/3, 0.05/2, 0.05/1) and labels a result that passes only there "passes within T15 only".

## 2. Data

Datasets, folds and rows are those of the stored experts: UrbanSound8K (8732 clips, 10 official folds), ESC-50 (2000 clips, 5 official folds, 50 classes), TAU 2020 Mobile
development (23035 segments of 10 s, five location-grouped folds over the device-A rows, heads trained on device A only, test conditions `clean` = device-A rows of the test fold and
`shift` = every other device's rows). Row order is the order of the metadata CSVs in `~/Documents/DATASETS/*/ast_768/`. Seeds: UrbanSound8K 0 to 2, ESC-50 0 to 4, TAU 0 to 2.
Audio is the 16 kHz mono mean-removed waveform of `audio_shift.load_clip`.

Conditions, all produced with stage 2's code (`audio_shift.py`, unchanged: `NoiseMaker`, `mix`, `field_assignment`, noise seeds `crc32(name|kind)`) so that every expert, old and new,
sees the same audio:

| Condition | UrbanSound8K | ESC-50 | TAU |
|---|---|---|---|
| clean | all rows | all rows | all rows (device A rows train and test, other devices' rows are the shift) |
| field mixture | test assignment seed 5001 and validation assignment seed 5002 (half the clips clean, the rest spread over the nine cells {bg, pink, white} x {10, 5, 0} dB); only the rows each assignment gives to a cell are extracted | same | not used |
| 0 dB (T1c definition) | bg 0 dB, all rows (the stored UrbanSound8K AST scores exist only for bg 0 dB) | bg, pink and white at 0 dB, all rows | not used |
| shift | not used | not used | the other devices' rows of the clean array |

The background noise of UrbanSound8K is drawn from ESC-50 audio and the reverse, as in stage 2 (`--bg-audio-dir` of the stored runs). Check C1 (same audio): the extractor's
`bg_plan.json` for each dataset must equal the stored one in `emb/*/bg_plan.json` of the idea 5 stage 2 folder for every clip it covers, and a re-extraction of the native MN10 view must
reproduce the stored `mn10_w8a8__clean.npy` (ESC-50: stage 3b; UrbanSound8K: the padded stage 3b file, because the new views use one padded graph) to a max abs difference reported in
the README.

## 3. Views (each is an expert: a frozen backbone and an `esc` FP32 head)

All EfficientAT views use the released mn10 AudioSet weights (`mn10_as_mAP_471.pt`, `torch.load(weights_only=True)`), the W8A8 backbone of `QWave.efficientat_mn.QuantCNN`
(per-channel 8-bit weights, 8-bit dynamic input scales as in the repo, batch norm and pooling in FP32) and the stage 3b pooled 960-d embedding. The native front end is
32 kHz, pre-emphasis 0.97, n_fft 1024, Hann window 800, hop 320, 128 Kaldi mel filters 0 to 15 kHz, `(log(mel + 1e-5) + 4.5) / 5`.

| Id | View | Definition |
|---|---|---|
| V0 | native control | the native front end into mn10 W8A8. Not a new expert; it is check C1 and the padded like-for-like reference |
| V1 | low band | native mel map, every mel row whose filter centroid lies above 2 kHz set to the floor value `(log(1e-5) + 4.5)/5`; the layout stays 128 rows |
| V2 | high band | native mel map, every mel row whose filter centroid lies below 2 kHz or above 8 kHz set to the floor value |
| V3 | short window | n_fft 512, Hann window 512, hop 128 (at 32 kHz), 128 Kaldi mel filters 0 to 15 kHz, same log scaling |
| V4 | long window | n_fft 2048, Hann window 2048, hop 512 (at 32 kHz), same filters and scaling |
| V5 | waveform CNN | 1D CNN trained from scratch on the 16 kHz waveform, section 4 |
| V6 | CNN14 | PANNs CNN14 (Kong et al. 2020), Zenodo record 3987831, file `Cnn14_mAP=0.431.pth`, MD5 `541141fa2ee191a88f24a3219fff024e` (listed by the record's API, 327428481 bytes), `torch.load(weights_only=True)`, W8A8 on every Conv2d and Linear as above, the 2048-d `embedding` (ReLU of `fc1`) |

Notes. (a) The index.md text calls 2048/512 "the standard long window". The native front end of mn10 is 1024/800/320, so V3 and V4 are both deviations from it and the existing MN10-Q8
is the native expert. The numbers 512/128 and 2048/512 are read at the model's sample rate (32 kHz): window and hop of 16 ms and 4 ms, and 64 ms and 16 ms. (b) V1 and V2 keep the 128-row
layout a frozen AudioSet CNN expects. The alternatives (render 128 mel rows over the band, or crop the rows) are excluded from this experiment. Masked rows are still computed by the
network, so V1 and V2 are priced like the native view. (c) V3 has 2.5 times the native number of frames and V4 0.625 times; the price counts the real frames. (d) CNN14 is reimplemented in
`QWave/panns_cnn14.py` (ConvBlock and Cnn14; front end 32 kHz, n_fft 1024, window 1024, hop 320, 64 mel rows 50 to 14000 Hz, slaney scale and norm, `10 log10(max(mel, 1e-10))`, no SpecAugment)
after reading the authors' `models.py`; no downloaded code is imported or run. Check C2: the front end and the embeddings of 8 clips are compared with the vendored torchlibrosa reference
run in isolated mode (`python -I`, outside the download folders); the max abs difference goes in the README. Check C3: the FP32 AudioSet clipwise output on a few clips with a clear AudioSet
counterpart (dog bark, siren, engine idling) lists the expected class in the top 5, as a sanity check of the checkpoint. (e) Fixed shapes: every UrbanSound8K clip is zero-padded at the end
to 64587 samples (16 kHz) before the front end, as in stage 3b; ESC-50 clips are 80000 samples and TAU segments 160000. The existing UrbanSound8K MN10-Q8 expert is the exact-length one (stage 3c).
With retrained heads, padded and exact-length mn10 agree within 0.1 points (stage 3c `padding_posthoc.csv`), and the padded stage 3b MN10 scores are given beside it as the like-for-like
reference in the precondition table.

Costs, priced as in the repo: `frontend MACs x 1024 + backbone MACs x 64 + backbone output elements x 1024 (rescale price) + head bit-ops` (`QWave.costs.cost` of the `esc` head).
EfficientAT views: `feature_macs` at the view's frame count; front end counted as in `efficientat_mn.frontend_macs` with the view's n_fft, window, hop and nonzero mel weights. CNN14: MACs of
every Conv2d and Linear from forward hooks at the clip's frame count, front end counted likewise. Waveform CNN: MACs of its layers from hooks at 8 x 8 bits. Measured CPU latency (batch 1,
2 threads, median of 50 after 10 warm-ups, FP32 kernels, one clip) is reported per view for backbone and front end; fake quantization is not an int8 speed and is labelled so.

## 4. The waveform CNN (V5)

Input: the 16 kHz waveform divided by its RMS (plus 1e-8) per clip. Architecture: Conv1d(1, 32, kernel 80, stride 8) + BatchNorm + ReLU, then five dilated blocks
Conv1d(c_in, c_out, kernel 3, dilation d, padding d) + BatchNorm + ReLU with (c_in, c_out, d) = (32, 48, 1), (48, 64, 2), (64, 96, 4), (96, 128, 8), (128, 192, 16), each followed by max-pool 2
except the last; the sum of the global average and global max of the last map, dropout 0.3, Linear(192, K). Parameters about 0.13 million (reported), under the 0.5 million limit. Training per outer
fold and seed on the fit rows (training folds minus the inner validation split, the same split as the other heads): AdamW, lr 2e-3, weight decay 5e-3, batch 32, cross-entropy with inverse
class-frequency weights and label smoothing 0.1, random circular time shift of up to a quarter of the clip, clean audio only. Epoch cap 10 for the precondition and 40 for the full run;
early stopping on inner-validation macro-F1 with patience 8, best checkpoint kept, plateau factor 0.6 with patience 3. Post hoc W8A8 fake quantization of every Conv1d and Linear
(per-output-channel 8-bit weights, per-sample 8-bit dynamic input) for all scoring, as the repo does for the other backbones. Its logits are the expert's output (no separate head).

## 5. Preconditions, evaluated on inner validation only, before any other number is read

Precondition fold: fold 1 of UrbanSound8K and fold 1 of ESC-50. Metric: macro-F1 on that fold's clean inner-validation clips (never a test clip), the same clips for the view and for the
reference, seeds 0, 1 and 2 averaged (the waveform CNN: seed 0 only). Reference "MN10 alone": the stored MN10-Q8 `esc`-head scores of the same fold and seeds.

* Frozen-embedding views (V1, V2, V3, V4, V6): **cut** if the view's inner-validation macro-F1 is more than 5.0 points below MN10 alone. Evaluated per dataset.
* V5: **dropped** if its clean inner-validation macro-F1 stays below 0.60 in all of its first 10 epochs (fold 1, seed 0). Evaluated per dataset.
* A view is available on a dataset if it passes there. It is available on TAU if it passes on both UrbanSound8K and ESC-50 (TAU has no precondition fold). A view available nowhere is
  reported and dropped. Views that fail are still listed in the precondition table with their numbers.
* The shifted conditions, TAU and all test folds are extracted and scored only for views that survive. T18's low-resolution candidates are extracted in the same clean pass and decided by T18's own precondition.

## 6. Mixtures and the selection on inner validation

Pool of a task: MN10-Q8 (E1) and AST-w4g128 at 1024 frames (E2, "AST-Q4") from the stored scores, plus the surviving views of that task. Mixture candidates: every uniform
average of the experts' softmax probabilities over a subset of size 2 to 4 of this pool. The existing best pair {E1, E2} is the baseline. Comparators and the hull use the larger singles universe:
MN10-Q8, MN20-Q8, AST FP32, AST w4g128, short AST (not for TAU), and every surviving view.

The mixture analysed as primary is chosen per task on inner-validation clips of that task's shifted pool (the same condition family as the test), before any test clip is scored: the candidate
with the largest inner-validation accuracy minus the hull's inner-validation accuracy at the candidate's priced cost (Stage 2's `hull_pair` on inner-validation accuracy), ties to the cheaper.
Shifted pools. UrbanSound8K: {field mix, bg 0 dB}. ESC-50: {field mix, bg 0, pink 0, white 0}. For each fold the pool value is the mean over its cells. The field-mix inner validation uses the
validation assignment (seed 5002), the test rows the test assignment (seed 5001). TAU: its inner validation holds device-A rows only (the heads are trained on device A), so no shifted
inner-validation rows exist; the selection and the best-single choice for TAU use clean device-A inner validation. This is forced by the data and is listed as a deviation.

Best single at equal cost: among the singles with priced cost at most the mixture's, the one with the highest inner-validation accuracy on the same pool (clean device-A for TAU).
Hull: the time-sharing hull of the singles at the mixture's cost, built on the test accuracy of the pooled cells (generous to the hull, as in T13).

Secondary arms, descriptive, outside the pass rule: (i) the margin router: the average of the cheap experts of the selected mixture (all but the AST) answers when its probability margin
(top1 minus top2) is at least a threshold set on clean inner validation so that 30 percent escalate; the rest are averaged with the AST; (ii) every candidate mixture on every cell
(not only the selected), with oracle accuracies (any expert right) and the best-pair oracle; (iii) macro-F1 versions.

## 7. Error correlation

Per fold, the Pearson correlation (phi) of the 0/1 error indicators of two experts on the task's pooled shifted test rows (seeds averaged inside the fold), for every pair of the pool, with
descriptive tables on clean rows. The clause: for every new view in the selected mixture, phi(view, E1) is lower than phi(E2, E1) in a majority of folds. If the selected mixture has no new
view the clause fails.

## 8. Primary tests, pass rule, stop rules

Primary unit: accuracy on the task's pooled shifted test rows, per outer fold, seeds averaged inside a fold. Contrasts: selected mixture minus best single at equal cost, and selected mixture
minus hull. Nadeau-Bengio corrected t as in T13 (n_test/n_train: ESC 1/4, UrbanSound8K 1/9, TAU the shift rows of the test fold over the device-A training rows). A task passes if all hold:
(1) both mean contrasts are at least +1.0 point; (2) each is at least +1.0 in a majority of folds (3 of 5, 6 of 10; the 4-of-5 and 7-of-10 counts are tabled); (3) the larger of the two
Nadeau-Bengio p-values clears the Holm threshold of section 1; (4) the error-correlation clause of section 7 holds. T15 passes if at least two of the three tasks pass.
Stop rules: **S-a** no selected mixture is more than 0.5 points above the baseline pair {E1, E2} in the pooled shifted accuracy on any task; **S-b** no new view survives its precondition.
The verdict is PASS, NO PASS or STOP.

Reported regardless of verdict: every view's precondition numbers, every pair and mixture on every cell with accuracy, F1 and priced cost, the error correlations and oracles, measured
CPU latency, the singles' accuracies on each pool.

## 9. Order and honesty clauses

1. This file and `ANALYSIS_PLAN_T18.md` are committed. 2. The extraction, check and training scripts are committed, then the clean extraction on UrbanSound8K and ESC-50 of all views
(including T18's candidates). 3. Fold-1 heads and the V5 precondition; the precondition table is written to `precond/` before anything else is read. 4. Extraction of the survivors on the
shifted conditions and on TAU. 5. All-fold heads and scoring of the inner-validation clips only; the pre-selection script (`t15_preselect.py`) is committed and run. 6. The scoring script that
reads test labels is committed, then run. 7. Analysis.
`scripts/benchmark.py` writes the held-out fold's own F1 into `summary.json`; it is read only by `score_eff.py`'s equality check, never for a decision. Smoke runs on small subsets read no test label and are deleted. Any change after the first extraction is recorded as an amendment with its commit. Heads are trained on clean training clips only
by `scripts/benchmark.py` (protocol 3: `inner_split_groups=recording`; TAU: five folds, recording token 2, device-A rows), kind `esc`.
Not tested here: fine-tuning any backbone, view-specific heads on noisy audio, distillation, re-rendering the bands over 128 rows, BEATs.
