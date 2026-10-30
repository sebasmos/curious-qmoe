# T24 analysis plan: the fixed T21 cascade on one more dataset never used before

Written 2026-10-28 18:27 (Irish time, equal to UTC on this date) by the T24 agent on branch `moe/t24` of `curious-qmoe` (worktree `~/Documents/qmoe-work/t24`, from `moe/t21` at `7957360`), committed together with every script the analysis uses, before any download. Nothing of FSDnoisy18k or of TUT Acoustic Scenes 2017 has been downloaded, not even the metadata: only the public Zenodo record pages (file names, sizes, md5, licence, description) were read. No audio, no embedding and no score of either dataset exists.

The scripts were run once end to end on synthetic random embeddings (`benchmark.py` heads, `t24_score.py`, `t24_analyze.py`, the combine step), to find bugs before any data. A session before this one wrote drafts of `t24_build_d3.py`, `t24_build_d4.py`, `t24_common.py`, `t24_prep_audio.py`, `t24_check_prep.py`, `t24_extract.py` and the `fixed_test` option of `scripts/benchmark.py`, uncommitted and with nothing downloaded. They were read line by line, kept, and are committed here.

Amendments are new commits with a date, what changed and what had been seen by then.

## A. The round plan, verbatim

Source: `AnpassenNN/mit-projects/36-qmoe/blindspot-reviews/reviews/review-fable5-1-max-effort.md` (coordinator, last committed `e40a4df`, 2026-10-28 17:38 UTC). Section 12 (T24) first, then sections 6.1 and 6.2 (T21, which section 12 keeps unchanged), then the section 2 rules.

## 12. T24: one more dataset never used before (owner's go, 2026-10-28)

Section 6.2's precondition says that a dataset failing it is reported and another is picked. FSD50K failed it and the step was not taken, so H1 rests on FSDKaggle2018 alone. T24 takes that step. It is section 6 continued and adds no new design. This section was written and committed before any T24 download or score.

| Item | Exact requirement |
|---|---|
| Design | Section 6.1 unchanged: cheap tier MN10-Q8, strong tier AST-Q4 (w4g128) at the clips' real length, gate the cheap tier's margin, average rule, budget (5, 10, 20, 30, 50%) and head scaler set on clean inner validation only, heads with the protocol 3 recipe on clean training clips, stage 2's shift mixer with the field mix. Code from `moe/t21` (archived worktree `~/Documents/qmoe-archive/worktrees/moe-t21`); new branch `moe/t24` from `moe/t21`. |
| Candidate criteria | Single-label general or environmental sound; never used in this project (excludes ESC-50, UrbanSound8K, TAU, Quinn, FSDKaggle2018, FSD50K); public with a stated licence; an official test split, or folds groupable by uploader, speaker or recording; at least 10 classes and about 900 test clips; raw audio that fits the disk rule below. |
| Candidate order | D3: FSDnoisy18k (Zenodo 2529934; 20 classes; 947 manually verified test clips; train labels noisy, so heads train on the full train set as published and inner validation is built from the train set's clean subset where the metadata allows, grouped by Freesound uploader). D4: chosen by the agent by searching against the criteria and committed in `ANALYSIS_PLAN.md` before any download, with the reasons. Stop at the first candidate whose precondition passes; every candidate tried is reported. |
| Folds | 5 folds inside the train set grouped by uploader (or speaker or recording), stratified by class where possible; test on the official test set. The paired test uses the 5 fold-trained models on the fixed test set, as in T21 D1. |
| Precondition | Section 6.2 unchanged: AST-Q4 beats MN10-Q8 by at least 2.0 points on the field mix (fold 1 is not enough; all folds, as T21 did). |
| H1, H2, H3, S1 | Section 6.2 unchanged. Holm now runs over the datasets where H1 is tested (D1 FSDKaggle2018 and the T24 dataset); T21's D1 p (0.0017) is re-adjusted with it and both adjusted numbers are reported. The campaign family for the transparency number becomes 20. |
| Overlap check | List Freesound IDs (or source IDs) shared with FSDKaggle2018 and FSD50K and with the UrbanSound8K clips the mixer uses as noise; report the count and the result with them dropped. |
| Disk rule | Free disk stays above 25 GB (35 GB on 28 Oct). Delete the archive after checksums pass and audio is extracted; keep only the sample rate the models read. If space runs short, the coordinator deletes the T20 cache (`~/Documents/qmoe-work/t20-cache`, 4.6 GB) and says so. |
| Audit | A pass gets an independent audit agent before it is called Accepted; a fail or stop needs the agent's own recompute from raw scores. |
| What the paper may claim | Pass: the pre-registered gain replicates on a second dataset never used. Fail or stop: H1 rests on FSDKaggle2018 alone, stated in the limitations. |

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

---

## B. Candidate order, fixed before any download

Stop at the first candidate whose precondition passes. Every candidate tried is reported. If both fail, H1 rests on FSDKaggle2018 alone and that is reported.

### D3: FSDnoisy18k (Zenodo 2529934), named by section 12

20 classes, single-label, Freesound clips of 0.3 to 30 s (44.1 kHz, 16-bit mono). 17,585 train clips, of which about 10% were checked by people (the "clean" subset, 51 to 170 per class) and the rest carry tag-derived labels with roughly 60% label noise. The 947 test clips were all checked by people (30 to 72 per class). Licence: CC BY for the dataset, CC BY or CC0 for every clip. Archives: `audio_train.zip` 9,238,586,835 bytes (md5 `34dc1d34ca44622af5bf439ceb6f0d55`), `audio_test.zip` 304,186,599 bytes (`1ac73d70b4ef3f81900d98c261a832de`), `meta.zip` 79,343 bytes (`96e27a4a63b7a2870522ddcedb5d8296`), `doc.zip` 90,519 bytes (`093a1ca185ec341ca4eac14215e7f96b`). It meets every criterion of section 12. It shares Freesound as a source with FSDKaggle2018, FSD50K and UrbanSound8K, so section C9's overlap check matters most here.

### D4: TUT Acoustic Scenes 2017 (Zenodo 400515 development, 1040168 evaluation), chosen by search

15 acoustic scene classes (bus, cafe/restaurant, car, city center, forest path, grocery store, home, lakeside beach, library, metro station, office, residential area, train, tram, urban park), 10 s segments, 44.1 kHz 24-bit binaural, recorded in Finland in 2016 by Tampere University of Technology. Development set 4,680 segments (312 per class) with a recording-location id per segment; official evaluation set 1,620 segments (108 per class) with labels released after DCASE 2017. Licence: free use for academic, non-commercial purposes (Zenodo "other-nc", stated in the dataset's EULA). Archives: development 10 audio zips of about 1.07 GB each plus meta, doc and error zips (10.70 GB in all); evaluation 4 audio zips (3.60 GB) plus meta and doc.

Why it is the D4, against the section 12 criteria:

- Single-label environmental sound: one scene label per segment. It is acoustic scenes, not isolated events, so a result here speaks to scenes; that is stated with any result.
- Never used in this project. TAU Urban Acoustic Scenes 2020 Mobile (used in this project) comes from the same group but is a different recording campaign (12 European cities, 2018 to 2019, several devices); TUT 2017 recordings are from Finland, 2016, one binaural device. No segment is shared. The TAU result in this project and a TUT 2017 result are not independent in task type, which is said with any result.
- Public with a stated licence: yes (academic, non-commercial).
- Official test split: yes, 1,620 evaluation segments, above the 900 of the criteria. Development folds groupable by recording location (the official DCASE 2017 setup groups by location too).
- 15 classes, at least 10.
- Disk: 14.3 GB of zips, fetched and deleted one at a time after the audio is prepared (6,300 x 10 s at 16 kHz float32 is about 4.0 GB kept).
- Not Freesound, so the Freesound-id overlap check of section 12 cannot apply. Segments are the group's own recordings, so overlap with FSDKaggle2018, FSD50K or UrbanSound8K is impossible by source. The backbones were trained on YouTube AudioSet; overlap with that is not expected and cannot be ruled out.

My odds that D4 passes its precondition are below even: on scene audio a log-mel CNN is usually close to a transformer, and the bar needs a 2.0 point gap on the field mix.

### Candidates searched and rejected

| Candidate | Why not |
|---|---|
| DataSEC (Zenodo 23097290, 22 classes, 4,292 clips) | Part of its audio comes from AudioSet, the backbones' training data; no official split; no recording, uploader or source id per file to group folds; 1,900 of 4,292 clips are one class; CC BY-NC-SA |
| ReaLISED (Zenodo 6488321, 18 indoor event classes, 2,479 clips) | No official split and no session, room or recording id to group folds by (file names carry class, index, action, material and intensity only), so folds could leak; 1.5 s mean clip length |
| CochlScene (Zenodo 7080122, 13 scenes, 76,115 clips) | 53.9 GB of split zips, which breaks the 25 GB free-disk rule |
| GISE-51 (Zenodo 4593514) and ARCA23K (Zenodo 5117901) | Built from FSD50K clips and splits, which section 12 excludes |
| FSDKaggle2019, TAU Urban Acoustic Scenes 2019 | Subsets of FSD50K and of the TAU recordings already used |
| NIGENS, Nonspeech7k, VocalSound, SINS (DCASE 2018 task 5), MSoS | Fewer than 900 clips with no test split (NIGENS), or fewer than 10 classes |

## C. Choices fixed here, before any download

### C1. The fixed design with config codes (T21's C1 unchanged except as said)

| Part | Code | Exact meaning |
|---|---|---|
| Cheap tier | `mn10_w8a8` | EfficientAT MN10 (`mn10_as_mAP_471.pt`, `QWave.efficientat_mn.build_backbone("w8a8")`), 8-bit per-channel weights, 8-bit dynamic activations, simulated in float on the CPU at each clip's own length. T21's extractor `t21_extract_mn.py`, unchanged. |
| Strong tier | `ast_w4g128_own` | AST (`MIT/ast-finetuned-audioset-10-10-0.4593`, revision `f826b80d`), `QWave.backbone_quant` config `w4g128`, at the clip's own frame count with T21's masked length buckets on MPS. T21's extractor `t21_extract_ast.py`, unchanged. |
| Heads | `esc` | The FP32 head of the protocol 3 recipe (`scripts/benchmark.py`, `config/esc50.yaml` values, CPU, 2 threads), trained on the clean-audio FP32 embeddings (`mn10_fp32`, `ast_fp32_own`), fold design of C2. |
| Head scaler (clean inner validation) | `fp32fit` or `refit` | As T21: StandardScaler on the FP32 fit rows, or `QWave.deploy.refit_scaler` on the deployed tier's clean fit rows; the higher mean clean inner-validation macro F1 over folds and seeds wins, a tie keeps `fp32fit`. The fit rows are the rows benchmark.py fitted on (`fit_idx` in each `split.json`). |
| Gate | margin | Top-1 minus top-2 softmax probability of the cheap tier; escalate below the threshold set on the fold model's clean inner validation at the budget rate (`thr_rate`). |
| Decision on escalated clips | `avg` | argmax of the mean of the two tiers' softmax probabilities. |
| Budget (clean inner validation) | 5, 10, 20, 30, 50% | T21's rule: the largest mean gain of the cascade's clean inner-validation macro F1 over the clean inner-validation hull of {`mn10_w8a8`, `ast_w4g128_own`, always-average} at the cascade's inner-validation cost (200 draws, `A.RNG_IV`); ties go to the cheaper budget. |

Not run in T24, declared now: the ONNX Runtime int8 cheap tier (a secondary path in T21 that no rule uses; its float16 mel cache would need about 13 GB for D3's 18,532 clips on top of the audio, and the extra CPU time would delay the answer) and an official-split secondary design (the primary design already tests on the official test set). Nothing else is chosen after seeing data.

### C2. Fold design (the paired test uses the five fold-trained models on the fixed official test set)

Code: `t24_build_d3.py`, `t24_build_d4.py` (metadata only), `benchmark.py` with `cross_validation.fixed_test=true`.

**D3, FSDnoisy18k.** The train set as published (17,585 clips: clean and noisy) is split into 5 folds with `StratifiedGroupKFold(5, shuffle=True, random_state=42)` on the train rows sorted by file name, stratified by class, grouped by Freesound uploader. The uploader of a clip comes from FSD50K's clip-info files (`dev_clips_info_FSD50K.json`, `eval_clips_info_FSD50K.json`, already on disk from T21) when the clip's Freesound id is there; otherwise the clip is its own group. The coverage (share of train clips with a known uploader) is reported. If the FSDnoisy18k metadata turns out to name the uploader itself, or the file names are not Freesound ids, an amendment made after reading only the metadata and before any audio states the change. Model k (k = 1..5):

- fits on every train row outside fold k (clean and noisy labels, "the full train set as published" minus fold k);
- uses as inner validation (head early stopping, scaler, budget, best-single, threshold) the clean-labelled train rows of fold k;
- does not use the noisy rows of fold k at all (they share uploaders with fold k's validation rows);
- is tested on the 947 official test clips.

Seeds 0 to 2 (a large train set; T21 used 3 seeds on FSD50K and 5 on the small FSDKaggle2018 pool).

**D4, TUT Acoustic Scenes 2017.** The 4,680 development segments are split into 5 folds with `StratifiedGroupKFold(5, shuffle=True, random_state=42)` on the rows sorted by name, stratified by scene, grouped by recording location. Every development label is reliable, so model k fits on the development rows outside fold k, uses all of fold k as inner validation, and is tested on the 1,620 evaluation segments. Seeds 0 to 4 (a small set, as T21's D1).

For both: each fold model's test rows are the whole official test set; no test clip is in any fit or validation split (asserted in `t24_score.py`); fold sizes and per-class counts are written to `csv/<d>/classes.json`.

### C3. Audio and length rule (T21's C3)

Stage 2's `load_clip` (mono mean, mean removed, resampled to 16 kHz), first 163,840 samples (10.24 s). To keep only what the models read (section 12 disk rule), `t24_prep_audio.py` writes exactly that float32 array as a 16 kHz float WAV per clip, and `t24_common.load_clip_t24` reads it back unchanged; every condition is then applied by stage 2's unchanged mixer, as in T21. Equality was checked before any download on 24 UrbanSound8K files of mixed rates and channel counts (`t24_check_prep.py`: maximum absolute difference 0.0). Background noise files go through T21's reader as before. The AST mask check of T21's amendment 1 is not repeated: it tests the bucket masking code, which does not depend on the dataset (as T21 did for FSD50K).

### C4. Conditions (T21's C4)

Clean; `bg` (UrbanSound8K, stage 2's bank and per-clip seeds), `pink`, `white`, each at 10 and 0 dB; the field mix (stage 2's `field_assignment`, test seed 5001, validation seed 5002). Extraction sets:

- every clip of `all.csv` on clean audio: `mn10_fp32`, `mn10_w8a8`, `ast_fp32_own`, `ast_w4g128_own`;
- every row of `eval.csv` (D3: the clean-labelled train rows and the test rows; D4: every row) under the six noise cells and the 5 dB cells the field mix needs: `mn10_w8a8`, `ast_w4g128_own`.

The field assignment runs over the rows of `eval.csv` in file order (train rows sorted by name, then test rows sorted by name): test rows read the cell of the test assignment, inner-validation rows that of the validation assignment, clean cells read the clean extraction.

### C5. Costs (T21's C5)

Priced bit-ops per clip at its own length, front ends included; cascade cost = cheap cost of every clip + the AST cost of each escalated clip; always-average = both; unpriced MACs as secondary. No latency.

### C6. Analysis (`t24_analyze.py`, committed with this plan)

Per dataset, per (seed, fold model): test macro F1 of `mn10_w8a8`, `ast_w4g128_own`, always-average and the cascade at the chosen budget, on clean, each noise cell and the field mix. Seeds are averaged within fold models; every test is over the five fold models.

- **Precondition.** Section 12 says "all folds, as T21 did". T21 applied the rule to the mean over all five folds (its D1 passed at +2.57 with two folds under 2.0). T24 uses the same reading: the mean over all five fold models of `ast_w4g128_own` minus `mn10_w8a8` macro F1 on the field mix must be at least 2.0 points. Every fold's value and the number of folds at or above 2.0 are reported beside it. A dataset that fails is reported; its H1 is not tested and it does not enter Holm.
- **H1.** Cascade minus the equal-cost hull on the field mix. Hull: stage 3c's `hull_pair` over {`mn10_w8a8`, `ast_w4g128_own`, always-average} (mean test F1 and cost on that condition) at the cascade's mean cost; per fold model, the exact expected macro F1 of random time-sharing between the two endpoints (`mix_expected_f1`). Fold-level paired t with the Nadeau-Bengio correction, two-sided. Ratio n_test / n_train per fold model, with n_train the rows that model used (its fit rows plus its inner-validation rows), averaged over the five models (D3 about 947 / 14,400; D4 1,620 / 4,680). PASS when the mean difference is at least 1.0 point and the Holm-adjusted p is below 0.05.
- **Holm.** Over the datasets where H1 is tested: FSDKaggle2018 (T21 D1, p_nb 0.00171982442287984, read from T21's `tables/d1/verdict.json` and checked) and the T24 dataset whose precondition passes (at most one, since T24 stops there). Both adjusted p values are reported. If no T24 candidate passes its precondition, D1 stays alone at 0.0017.
- **Family transparency.** min(1, 20 x raw p) for the campaign family of 20, reported beside it.
- **Intervals.** The 95% Nadeau-Bengio interval over fold models measures training variation only, since the five models share one test set. A clip-level bootstrap (2,000 draws, seed 20261028) resamples the test clips once per draw and uses the same resampled clips for every fold model and seed (hull by linear interpolation of the endpoint F1s inside each draw); it measures test-sample variation. Both are reported; the decision follows the NB test as section 6.2 states. Folds positive and the sign-flip floor (0.0625) are reported.
- **H2.** The same on clean audio (mean at least 1.0, raw NB p below 0.05, as T21).
- **H3.** With the clean inner-validation threshold fixed, the test escalation rate (mean over fold models) rises from clean to 10 dB and from 10 dB to 0 dB for each noise kind, and the field mix is above clean; exceptions listed. Gate AUROC of (1 - margin) for "cheap wrong and AST right" and for "cheap wrong", per condition.
- **S1.** The cascade is within 0.3 points of the hull (absolute mean difference) on both datasets where H1 runs (D1 and the T24 dataset). Reported per dataset too.
- **Comparators** on every bed: the hull, a random gate at the same escalation count with the `avg` decision, the best single of the two tiers chosen on inner validation of the same condition family (field-mix validation rows for the field bed, clean for clean), always-average.
- A candidate whose precondition fails still gets H2, H3 and the cascade-minus-hull number reported as descriptive, as T21 did for FSD50K.

### C7. Order and stopping

D3 end to end first (download, CSVs, extraction, heads, scores, analysis). If its precondition passes, H1 is tested on D3, the T24 verdict is written and D4 is not started. If it fails, D4 runs end to end under the same rules. If D4 fails too, T24 reports both and stops.

### C8. Checks before any score (T21's C8)

Zenodo md5 of every archive before it is opened; row counts equal the CSVs; names checksum equal across every embedding file of a dataset; no NaN where rows are needed; MN10 and AST read the same clip lengths; no test clip in any fit or inner-validation split (asserted); thresholds and budget from inner-validation rows only; each head's test F1 rescored from the saved model must equal its `summary.json`; the plan file diffed against this commit before the verdict.

### C9. Overlap check (section 12, `t24_overlap.py`)

For D3: the Freesound ids shared with FSDKaggle2018 (its `freesound_id` columns), with FSD50K (its `fname` ids, dev and eval) and with the UrbanSound8K recordings the mixer uses as background (the `fsID` part of the bank's file names), and the eval rows whose bg noise, replayed through stage 2's `NoiseMaker` by name, was drawn from a recording that is itself a D3 clip. Counts are reported for all clips and for test clips. The sensitivity run drops every test clip in any of those sets and repeats the field and clean comparisons (`t24_analyze.py --drop`). It is reported beside the primary result and does not replace it. For D4 the check reports that no Freesound ids exist.

### C10. Machine rules

GPU only through `~/Documents/qmoe-work/fair_gpu_run.sh claude-t24`, one MPS process at a time, shares of at most 75 minutes. Free disk above 25 GB at every step (each archive deleted after its md5 passes and its audio is prepared; downloads refused when they would cross the line); work proceeds only while `memory_pressure` reports at least 25% free. Downloads go to new directories under `~/Documents/DATASETS/FSDnoisy18k/` and `~/Documents/DATASETS/TUT2017/`, are read with `python -I` as data, and no code from them is run.

### C11. Before any fail or stop is reported

An own recompute from the saved score files (`t24_recompute.py`, independent of `t24_analyze.py`'s functions, committed before the first score file exists) must reproduce the precondition, H1 and H2 numbers.

## Amendments

### Amendment 1, 2026-10-28 18:28, after reading the D3 metadata only (no audio of D3 or D4 on disk, no embedding, no score)

FSDnoisy18k's `train.csv` and `test.csv` share three file names (Freesound ids): `393581.wav`, `414515.wav` (both Fireworks in both files) and `417524.wav` (Fire in train, Fireworks in test). All three train rows are noisy (manually_verified 0). They are dropped from the train set, so the train set is 17,582 clips and no test sound is trained on; the 947 test clips stay whole. File names are Freesound ids and the metadata does not name the uploader, so the uploader rule of C2 (FSD50K clip info, else the clip's own group) applies as written.

Fold facts from the metadata, with the committed C2 rule unchanged: 6,929 of the 17,582 train clips (39%) have an uploader in FSD50K's clip info; the other 10,653 are their own groups (12,455 groups in all). Fold sizes 3,482, 2,894, 3,562, 3,862, 3,782; clean-labelled (inner-validation) rows per fold 299, 231, 381, 479, 382. One uploader holds 216 of the 1,772 clean train clips, so some folds' inner validation has no clip of a class (minimum 0 per class and fold: Bass guitar in folds 2 and 5, Crash cymbal in fold 1). Stratifying on class times the clean flag was tried on the metadata and also leaves zeros, so the committed rule is kept. Inner-validation macro F1, the head's checkpoint choice (validation macro F1 in `QWave/train_utils.py`), thresholds and the budget all use the classes present in each fold's inner validation, as T21's F1 rule does; the test set has every class (30 to 72 clips each).

### Amendment 2, 2026-10-28 23:14, after D3's precondition failed and after reading the D4 metadata only (no D4 audio on disk)

D3 result that triggers D4 (plan C7): AST-Q4 minus MN10-Q8 on the D3 field mix is +1.40 (fold models 0.94, 3.60, 2.83, -0.43, 0.08), under the 2.0 bar, so D3's H1 is not tested and D4 runs. The D3 run had no amendment beyond amendment 1. D4 metadata facts, with the committed C2 rule unchanged: 248 development locations, each one recording; 109 evaluation locations, none shared with development; fold sizes 891, 942, 942, 957, 948; four classes have no segment in one fold's inner validation (home in fold 3, library in fold 2, office in fold 5, residential area in fold 4), handled as in D3 (classes present). The development set's `error.txt` marks 84 segments with temporary microphone failure or radio interference; they are kept as published, as the official DCASE 2017 setup keeps them. `t24_build_d4.py` now strips a trailing carriage return from each metadata line (the files use CRLF; `read_text` already normalised them, so the CSVs are the same either way). D4 seeds 0 to 4 as C2 states.
