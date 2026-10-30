# T21 analysis plan: confirmatory test of the fixed cascade on data never seen

Written 2026-10-26 14:10 (Irish time) by the T21 agent, committed on branch `moe/t21` (worktree `~/Documents/qmoe-work/t21`, from `moe/t13` at `94f6af6`) before any audio of the new datasets is downloaded, before any extraction and before any score. Only the metadata CSVs of FSDKaggle2018 (Zenodo 2552860, `FSDKaggle2018.meta.zip`, md5 `f16828ac8be0e5285b9175a12f90d784`, matches Zenodo) and of FSD50K (Zenodo 4060432, `FSD50K.metadata.zip` and `FSD50K.ground_truth.zip`, md5 match Zenodo) have been read, to fix the fold design. No audio, no embedding and no label-level result of either dataset has been seen.

Amendments are new commits with a date, what changed and what had been seen by then.

## A. Sections 6.1 and 6.2 of the round plan, verbatim

Source: `mit-projects/36-qmoe/blindspot-reviews/reviews/review-fable5-1-max-effort.md` (coordinator, 2026-10-26).

## 6. T21: confirmatory test of the existing cascade on data never seen

### 6.1 Hypothesis and design

| Item | Content |
|---|---|
| Thesis | The margin-routed two-expert cascade beats uninformed time-sharing at equal cost by about 1 to 2 points under shift. This effect has appeared in the same direction on three of four tasks and failed only the family correction; a single pre-registered test on unseen data certifies or refutes it without a family penalty. |
| Fixed design (no choice after seeing the new data) | Cheap tier: MN10-Q8 (EfficientAT MN10, W8A8 simulated and real ONNX int8 with scaler refit; the FP32 head). Strong tier: AST-Q4 (w4g128) at the clips' real length, FP32 head. Gate: the cheap tier's margin; decision rule: average of both tiers. Only the budget (5, 10, 20, 30, 50%) and the head scaler are set on the new dataset's clean inner validation. Heads trained with the protocol 3 recipe on clean training clips. |
| New datasets (never used in this project) | D1: FSDKaggle2018 (Freesound, 41 classes, single-label, about 9.5k train and 1.6k test clips, official split, manually verified test labels; Zenodo 2552860; about 5 GB). Build 5 folds inside the train set, stratified by class; test on the official test set with the manually verified subset as the primary test; treat the official split as the sixth "fold" for the paired test only if folds are not available (state the design). D2: FSD50K single-label subset (Zenodo 4060432; take clips with exactly one leaf label and classes with at least 60 such clips in the dev set; expect 100 or more classes; the official dev/eval split; 5 folds inside dev grouped by uploader if the metadata gives it, else stratified; about 30 GB, only if disk allows; the fine-grained task is where the gain was largest). Both are Freesound sources like ESC-50 and UrbanSound8K; the backbones were trained on YouTube AudioSet, so clip overlap is not expected; say what cannot be ruled out. |
| Shift condition | Built with stage 2's mixer: background noise from UrbanSound8K clips at 10 and 0 dB, pink and white at 10 and 0 dB, and the field mix with the same assignment rule. The new dataset's clean test is the secondary bed. |
| Comparators | The equal-cost hull of {MN10-Q8, AST-Q4, always-average}, random time-sharing, a random gate at the same count, the best single chosen on inner validation of the condition family. |

### 6.2 Pre-registered rules

| Rule | Statement |
|---|---|
| Primary H1 (one test per dataset) | On the field mix, the fixed cascade at its inner-validation budget beats the equal-cost hull by at least 1.0 macro F1 point, fold-level paired t with Nadeau-Bengio, two-sided alpha 0.05, Holm over the datasets run (2 at most). No family penalty from the campaign; the family-of-17 number is reported alongside for transparency. |
| Secondary H2 | The same on clean audio. |
| Secondary H3 (humility) | Escalation with the clean threshold rises at every noise step; exceptions listed; gate AUROC for "cheap wrong and AST right". |
| Stop S1 | The cascade is within 0.3 points of the hull on both datasets. |
| Precondition | The AST-Q4 static beats MN10-Q8 by at least 2.0 points on the field mix of the new dataset (otherwise there is no headroom and the dataset cannot test the claim; report and pick another). |
| What a pass lets the paper claim | A replicated, pre-registered gain of about 1 to 2 points over uninformed time-sharing from routing by the quantized cheap expert's own confidence, on datasets chosen before the design was fixed, with escalation rising under shift. Modest and honest. |

### 6.3 Steps

| S | What | Inputs | Produces | Check |
|---|---|---|---|---|
| 21.1 | Worktree `~/Documents/qmoe-work/t21` from `moe/t13`; results folder; `ANALYSIS_PLAN.md` (sections 6.1, 6.2 verbatim, the fixed design named with config codes); commit before any download finishes. | | plan commit | |
| 21.2 | Download D1 (and D2 if disk allows), verify checksums against Zenodo, record licence; store under `~/Documents/DATASETS/FSDKaggle2018/` and `FSD50K/` with `audio_raw/`; build the CSVs with folds and labels; record class counts. | network | CSVs | Checksums match; folds stratified; no test clip in any fold |
| 21.3 | Extraction (GPU lock): MN10 W8A8 and the real ONNX int8 path, AST w4g128 at the clips' real length, on clean and the shift conditions with stage 2's mixer and plan. | 21.2 | `emb/` | Names checksum across configs; same audio for both tiers |
| 21.4 | Heads (protocol 3 recipe); scaler refit for the int8 path; scores. | 21.3 | `scores/` | |
| 21.5 | Precondition, cascade with the fixed design, hull, tests, humility; verdict per dataset. | 21.4 | `analysis/` | The design was not changed after seeing the data (diff the plan) |
| 21.6 | README, manifest, commit, final report (at most 300 words). | | | |

Estimated machine time: D1 one day (download, extraction, analysis); D2 one to two days more. Odds of H1 on D1: better than even, by the coordinator's judgment.

## B. Rules of section 2 of the round plan that apply here, verbatim

## 2. Rules for every slice (copy into every agent prompt)

| Rule | Exact requirement |
|---|---|
| Code location | The single code folder is `/Users/cajas.sebastian/Desktop/repositories/curious-qmoe` (branch `paper/quantized-gate`, with `moe/*` branches). Never work on the Desktop. Create a worktree outside it: `git -C /Users/cajas.sebastian/Desktop/repositories/curious-qmoe worktree add -b moe/t19 ~/Documents/qmoe-work/t19 moe/t13` (same for t20, t21). Archive worktrees to `~/Documents/qmoe-archive/worktrees/` when done; never delete branches. |
| Git identity | Commit as `Sebastian Cajas <sebasticajas@gmail.com>`. No Co-Authored-By line of any kind. Never push. |
| Documents | Do not edit Overleaf (`curious-moe-overlead`), `index.md`, `master-table.md`, `BACKLOG.md` or any `paper-versions` folder. The coordinator logs results. |
| Pre-registration | Write `ANALYSIS_PLAN.md` in the results folder and commit it on the branch BEFORE any extraction that produces test-fold scores and before any test statistic. Amendments are new commits, dated, listing what changed and what had already been seen. Every rule in sections 4 to 6 goes into the plan verbatim. |
| Family | Campaign family for T19 and T20 is 17 (T1, T1b, T1c, T2b, T7, T9, T12, T13, T14, T15, T16, T17, T18, T3, T5, T19, T20). Report Holm within the experiment and the family-of-17 number next to it. T21 is a confirmatory single test per dataset on unseen data; its claim rests on that single test (alpha 0.05 two-sided, Holm over its own datasets); report the family-of-17 number alongside for transparency. |
| Protocol | Official folds (ESC-50 5, UrbanSound8K 10, TAU 5 location-grouped over device A); inner validation grouped by source recording (ESC-50: second token of the file name; UrbanSound8K: fsID; TAU: location); thresholds and budgets chosen on clean inner validation only; heads trained on clean training clips with the protocol 3 recipe (`scripts/benchmark.py`); macro F1; seeds ESC-50 0 to 4, UrbanSound8K 0 to 2, TAU 0 to 2. |
| Comparators | The hull is the time-sharing hull of every static system available at that cost plus the always-average point(s); the best single model is chosen on inner validation of the same condition family (field mix, noise, shift), never on clean validation. Random time-sharing and a random gate at the same escalation count are reported as well. |
| Statistics | Fold level (seeds averaged within folds), paired t with the Nadeau-Bengio correction, sign-flip floor stated (0.0625 with 5 folds), bootstrap 95% intervals, Holm over the tasks of the primary test. No pair-level p values in the verdict. |
| Costs | Priced bit-ops: MACs times weight bits times activation bits, group and channel scales priced at 32 x 32 per rescale MAC, mel and fbank front ends included (`QWave/costs.py`, idea 5b `common.py`, stage 3c cost tables). Report the unpriced view as secondary. Measured CPU latency where a real kernel exists; say plainly that no int4 activation kernel exists on this Mac, so A4 costs are a bit-op proxy. |
| Noise and shift | Stage 2's exact mixing code and plan (`bg_plan.json`) so every system sees the same audio: background (UrbanSound8K clips for ESC-50 and the reverse), pink and white at 20, 10, 5, 0 dB; the field mix; the three non-additive corruptions (reverberation RT60 0.3, 0.6, 1.2 s; clipping 0.5, 0.25, 0.1; low-pass 4, 2, 1 kHz) from `2026-10-24_1507_redesign-idea6-final_code-ed82400`; TAU device shift. |
| GPU | Apple M2 Pro, 32 GB. One MPS process at a time, enforced by the lock: `mkdir ~/Documents/qmoe-work/.mps-lock` (atomic). If it fails, another agent holds the GPU: retry every 60 s inside Bash calls of at most 9 minutes. Write your agent name and PID into the lock directory, release with `rmdir` in a shell trap on every exit, and release between processes so others can interleave. Fixed input shapes (one graph). Check `sysctl vm.swapusage` before each process and wait if used exceeds 9.5 GB. CPU jobs at `nice -n 10` with `OMP_NUM_THREADS=2`, one at a time while the GPU is busy. |
| Results | `~/Documents/qmoe-results/<YYYY-MM-DD_HHMM>_moe-t19_code-<commit>/` with `run_manifest.json` (commit, machine, software versions, dataset sha256), `ANALYSIS_PLAN.md`, `README.md` (provenance, commands, verdict per rule, deviations), `scripts/`, `logs/`, `tables/`. Big caches may live beside it but are not copied. |
| Writing | Plain English, no em dashes, no "not X but Y", no colon titles, every number traceable to a file. |
| Audit | A PASS is not "Accepted" until an independent auditor agent (own code, read-only) reproduces the numbers, checks leakage and costs, and recomputes the hull. The coordinator launches it. |


## C. Choices fixed here, before any audio or score

### C1. The fixed design with config codes

| Part | Code | Exact meaning |
|---|---|---|
| Cheap tier, primary | `mn10_w8a8` | EfficientAT MN10 (`mn10_as_mAP_471.pt`, `QWave.efficientat_mn.build_backbone("w8a8")`): per-output-channel 8-bit weights, 8-bit dynamic activations, simulated in float, run on the CPU at each clip's own length (no padding). This is the "MN10-Q8" of every earlier cascade number. |
| Cheap tier, deployment path (secondary) | `mn10_int8` | The same MN10 exported to ONNX and quantized with ONNX Runtime `quantize_static` (QDQ, int8 symmetric per-channel weights, uint8 per-tensor activations), calibration method Percentile (the stage 3c choice in `2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1/models/mn10_calib_select.json`, not reselected), one model per outer fold calibrated on 128 clean clips drawn with numpy seed 0 from the fold's training clips that are in no seed's inner-validation split. Code: stage 3c `ort_int8.py` functions (`exportable`, `quantize`, `session`), each clip at its own length. |
| Strong tier | `ast_w4g128_own` | AST (`MIT/ast-finetuned-audioset-10-10-0.4593`, revision `f826b80d`), `QWave.backbone_quant` config `w4g128` (4-bit group-wise weights, group 128, 8-bit activations, 16-bit attention probabilities), at the clip's own frame count (section C3). |
| Heads | `esc` | The FP32 head of the protocol 3 recipe (`scripts/benchmark.py`, `config/esc50.yaml` values, `inner_split_groups=recording`, validation ratio 0.1, CPU, 2 threads), trained on the clean FP32 embeddings of the training clips of each fold (`mn10_fp32` for the cheap tier, `ast_fp32_own` for the strong tier), as the stage 3b and 3c heads were. |
| Head scaler (chosen on clean inner validation) | `fp32fit` or `refit` | Per tier and dataset, one of two scalers: the StandardScaler fitted on the FP32 clean fit rows (the default of `score_eff.py`), or `QWave.deploy.refit_scaler` on the deployed tier's clean fit rows (`mn10_w8a8`, `mn10_int8` from the fold's own int8 model, `ast_w4g128_own`). The one with the higher mean macro F1 on clean inner validation (over folds and seeds) is used everywhere for that tier; a tie keeps `fp32fit`. |
| Gate | margin | Top-1 minus top-2 softmax probability of the cheap tier's head. Escalate a clip when its margin is below the threshold set on the fold's clean inner validation at the budget rate (`thr_rate` of stage 2's `analyze.py`). |
| Decision on escalated clips | `avg` | argmax of the mean of the two tiers' softmax probabilities. Non-escalated clips keep the cheap tier's argmax. |
| Budget (chosen on clean inner validation) | 5, 10, 20, 30, 50% | One budget per dataset and cheap-tier path: the one with the largest mean (over folds and seeds) gain of the cascade's clean inner-validation macro F1 over the clean inner-validation hull of {`mn10_w8a8`, `ast_w4g128_own`, always-average} at the cascade's inner-validation cost (the stage 3c `select` rule, 200 random-mixing draws, `A.RNG_IV`); ties go to the cheaper budget. |

Nothing else is chosen after seeing the new data. No other cheap tier, strong tier, head kind, rule, signal, length rule, condition or comparator is added to the primary test.

### C2. Datasets and folds

**D1, FSDKaggle2018** (Zenodo 2552860; 41 classes; 9,473 official train clips of which 3,710 manually verified, 1,600 official test clips, all manually verified; licence CC BY 4.0 for the dataset, CC BY, CC0 and 3 CC BY-NC clips individually, from `LICENSE-DATASET` and the CSVs). Metadata checked: `freesound_id` is unique per clip in both CSVs and no id is shared between train and test, so grouping by Freesound id is grouping by file; the CSVs carry no uploader, so clips of one uploader can fall in different folds (stated as a limitation).

- Primary design: 5 outer folds over the 5,310 manually verified clips (3,710 verified train plus the 1,600 test), stratified by class with `StratifiedKFold(5, shuffle=True, random_state=42)` on the pool sorted by file name (per class 84 to 366 clips, so every fold holds every class). Inner validation by the protocol 3 rule grouped by file (`recording_token=0` of the file name, each clip its own group). Head seeds 0 to 4. Non-verified train clips are not used in the primary design.
- Secondary design: the official split. Heads trained on all 9,473 official train clips (verified and non-verified labels, as the official task intends), inner validation by the same rule, tested on the 1,600 official test clips; seeds 0 to 4; one split, so it is descriptive (clip-level bootstrap intervals, no fold-level test).
- The official test set is fully inside the primary pool, so the secondary design reuses the primary extraction for its test clips; the non-verified train clips are extracted on clean audio only.

**D2, FSD50K single-label subset** (Zenodo 4060432), only if the disk rule allows (section C7). A clip qualifies when its smeared label set is one chain: exactly one label that is a leaf (a class that is no other class's ancestor, where B is an ancestor of A when every dev clip labelled A is also labelled B; this gives 145 leaves on the dev labels, against 144 in the README, and the difference is reported), and every other label is an ancestor of it. Classes kept: at least 60 qualifying dev clips (113 classes, 26,272 dev clips, 4,396 eval clips on the metadata). Primary: 5 folds over the qualifying dev clips with `StratifiedGroupKFold(5, shuffle=True, random_state=42)` grouped by the `uploader` field of `dev_clips_info_FSD50K.json`; inner validation grouped by uploader as well. Secondary: heads on all qualifying dev clips, tested on the qualifying eval clips. Seeds 0 to 2 (larger set). If only part of D2 can be stored, D2 is not run (no partial subset).

### C3. Length rule and how each tier reads a clip

- Audio: stage 2's `load_clip` (mono, mean removed, resampled to 16 kHz), then the first 163,840 samples (10.24 s) are kept. Every condition is applied to this truncated waveform with stage 2's unchanged mixer, so both tiers read the same audio.
- AST: the fbank of the truncated waveform (HF `ASTFeatureExtractor`, as stage 2), T frames (at most 1,022), time patches t = floor((T - 16) / 10) + 1, read unpadded: only the clip's own t time patches per frequency row, with the position embeddings of the first t time positions (as `QWave.ast_frames.shorten_ast`). For speed on MPS with few graph shapes, clips are run in 8 length buckets (128, 256, ..., 1024 frames) with every patch beyond the clip's t masked out of attention as a key (score set to -1e9 before the softmax) and its value row zeroed before the per-channel value quantization. With that mask the [CLS] and [DIST] outputs equal the unpadded run up to float rounding; this is checked on 32 clips against single-clip runs of `shorten_ast` at each clip's own frame count before any extraction (pass: mean relative embedding difference below 1e-3, reported). Clips shorter than 16 frames cannot occur (shortest 0.3 s, 28 frames).
- MN10: its own front end (`QWave.efficientat_mn.model_input`, 32 kHz log mel) on the same truncated 16 kHz waveform, each clip at its own length, on the CPU.

### C4. Conditions

Clean; `bg` (UrbanSound8K clips, the stage 2 background bank and seeds), `pink`, `white`, each at 10 and 0 dB; the field mix (stage 2's `field_assignment`: half the clips clean, the rest spread over {bg, pink, white} x {10, 5, 0} dB, test seed 5001, validation seed 5002, the 5 dB cells extracted only for the rows the two assignments give them). The field mix and the clean test are the beds; the six full noise cells feed H3. Extraction set per dataset: `mn10_fp32` clean; `mn10_w8a8` every condition; `mn10_int8` every condition (test rows by their fold's model, clean by every fold's model for the inner-validation and refit rows); `ast_fp32_own` clean; `ast_w4g128_own` every condition.

### C5. Costs (priced bit-ops per clip, then averaged over the clips actually scored)

- MN10: the front end (`frontend_macs(n) x 32 x 32`) plus the backbone at W8A8 with channel rescales at 32 x 32 (`backbone_bitops(..., "w8a8", 1024)`) for the clip's own length, plus the FP32 head (bit-ops of the protocol 3 network at 41 or the D2 class count). The int8 path is priced as W8A8 (as stage 3c).
- AST: the fbank front end (`ast_fbank_frontend_macs(n) x 1024`) plus `ast_bitops_tok("w4g128", 1024, tokens, patches)` with the clip's own token count (12 t patches, 12 t + 2 tokens), plus the FP32 head.
- Cascade cost: mean over test clips of (cheap cost of the clip + escalated indicator x AST cost of the clip). Static costs: means over the same clips. Always-average: the sum of both statics. Unpriced view (MACs) reported as secondary. No latency is measured in T21.

### C6. Analysis (fixed before scores)

Per dataset, per (seed, fold): test macro F1 of `mn10_w8a8`, `ast_w4g128_own`, always-average, and the cascade at the chosen budget, on clean, each noise cell and the field mix (field rows assembled from the cell files by the test assignment). Seeds are averaged within folds; every test is at fold level.

- Precondition: fold-mean macro F1 of `ast_w4g128_own` minus `mn10_w8a8` on the field mix at least 2.0 points. If it fails on a dataset, the dataset is reported and its H1 is not tested (it does not enter Holm).
- Primary H1: cascade minus the equal-cost hull on the field mix. Hull: stage 3c's `hull_pair` over the static points {`mn10_w8a8`, `ast_w4g128_own`, always-average} (mean test F1 and mean cost on that condition) at the cascade's mean cost; the hull's F1 per (seed, fold) is the exact expected macro F1 of random time-sharing between its two endpoints (`mix_expected_f1`). Test: fold-level differences, paired t with the Nadeau-Bengio correction (ratio n_test / n_train of the fold, as T13's `nb`), two-sided, Holm over the datasets whose precondition passed. PASS when the mean difference is at least 1.0 point and the Holm p is below 0.05. Reported beside it: the 95% Nadeau-Bengio interval, a clip-level bootstrap 95% interval (2,000 draws, clips resampled within folds, hull by linear interpolation of the endpoint F1s inside each draw), folds positive, the sign-flip floor (0.0625 with 5 folds), and the family-of-17 number min(1, 17 x raw p).
- Secondary H2: the same on clean audio.
- Secondary H3 (humility): with the clean inner-validation threshold held fixed, the test escalation rate (fold mean) for each noise kind rises from clean to 10 dB and from 10 dB to 0 dB, and the field mix is above clean; exceptions listed. Gate AUROC of (1 - margin) for "cheap wrong and AST right" (and for "cheap wrong") per condition.
- Stop S1: the cascade is within 0.3 points of the hull (absolute mean difference below 0.3) on both datasets; if only D1 runs, S1 is evaluated on D1 alone and that is said.
- Comparators reported for every bed: random time-sharing at the cascade's cost (the hull), a random gate at the same escalation count with the same `avg` decision (`mix_expected_f1` between the cheap tier's and the average's predictions at the cascade's rate), the best single of {`mn10_w8a8`, `ast_w4g128_own`} chosen on inner validation of the same condition family (field-mix validation rows for the field bed, clean for clean), and always-average.
- Secondary, not in Holm: the same H1 test with the deployment cheap tier `mn10_int8` (its own scaler and budget choice by the rules of C1), and the official-split design (descriptive).
- Every number written to `tables/` with its source file; `verdict.json` holds the rules applied verbatim.

### C7. Disk and order

D1 first, end to end. D2 only after D1 is finished, only after `~/Documents/qmoe-work/t20-tau-cache-done` or `t20-tau-cache-skipped` exists, and only if free space after downloading (24.7 GB of zips) and extracting the qualifying clips stays above 25 GB; otherwise D2 is reported as not run for disk.

### C8. Checks before any score

Zenodo md5 of every archive; row counts equal the CSVs; names checksum equal across every embedding file of a dataset; no NaN; no test clip in any training or inner-validation split or in any int8 calibration list (asserted); thresholds and budget computed from inner-validation rows only (asserted on indices); the plan file diffed against this commit before the verdict.

## Amendments

### Amendment 1, 2026-10-26 14:17, before any FSDKaggle2018 audio is on disk (only the metadata has been read)

The mask check of C3 (masked buckets against single-clip unpadded runs) was run on the CPU on UrbanSound8K clips (8 and 12 clips, buckets 256 and 512). FP32: mean relative difference 1.8e-6, inside the 1e-3 bar. `w4g128`: 0.050. The same unpadded reference moved by 1e-6 Gaussian noise on its fbank input differs from itself by 0.052 on the same clips (and by 0.05 to 0.06 on random inputs, for `w8g128` too): the 8-bit dynamic activation rounding amplifies any float-level change (a different input width in the patch convolution here, or the device, or the batch) to about 5% of the embedding, and that is a property of the simulation every earlier stage used, not of the mask. The check's pass rule becomes: FP32 mean relative difference below 1e-3, and for a quantized config a mean relative difference no larger than 1.5 times the reference's own floor under 1e-6 input noise, both measured on the new dataset's clips on the extraction device before extraction. Nothing else changes.

### Amendment 2, 2026-10-26 20:40, before any score (embeddings extracted, MN10 heads trained, no logits or F1 of any deployed tier computed or read)

ONNX Runtime's static calibration stacks every calibration input's intermediate outputs into one array, so the 128 calibration clips must share one length. Each calibration clip's clean log-mel is tiled along time (or cut) to 1,025 frames (10.24 s) for calibration only. The int8 models then embed every clip at its own length, as planned. Also recorded: the ORT FP32 export differs from the torch FP32 embeddings by at most 0.3% relative on 16 clips, because the mel cache the ORT path reads is float16.

### Amendment 3, 2026-10-26 23:22 (commit d480027; the header first said 23:25, an estimate written before the commit), before any FSD50K audio is downloaded (D1 embeddings extracted, MN10 scores of D1 computed and checked against summary.json; no D1 cascade, hull or AST test number seen)

- D2 CSVs (`t21_build_d2.py`, metadata only) reproduce C2: 200 labels, 145 leaves, 113 classes, 26,272 qualifying dev clips from 3,112 uploaders, 4,396 qualifying eval clips. `StratifiedGroupKFold` by uploader gives unequal folds (3,889, 7,958, 4,391, 5,715, 4,319 clips) and 7 classes are absent from one fold's test; the plan's grouping is kept as written and this is reported. The protocol 3 inner split groups by uploader through the head CSV's name column (`u<uploader index>-<fname>`, `recording_token=0`).
- D2 secondary (official split) is evaluated on clean audio only: the 4,396 eval clips are extracted on clean audio, as the extra clips of D1 were. The primary design is unchanged (every condition).
- The AST mask check (C3) is not repeated on D2: it tests the bucket masking code, which does not depend on the dataset.
- The official design's costs use each clip's own length for the extra rows (before, the analysis padded them with the pool mean; in D1 this only touches the official design's inner-validation cost).
- D2 head seeds 0 to 2, as C2 states.
