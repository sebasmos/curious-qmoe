# T23 analysis plan: a pretrained waveform expert as a second modality in the margin-routed quantized cascade

Written 2026-10-28 between 01:20 and 02:00 (Irish time) by the T23 agent and committed on branch `moe/t23` (worktree `~/Documents/qmoe-work/t23`, from `moe/t21` at `7957360`) before any checkpoint is downloaded, before any extraction and before any score of the waveform expert exists. What has been seen so far: the stored score files' names and availability (`2026-10-25_0145_moe-t13_code-eaaedc9/inputs/availability.csv`), the stored cost tables, T15's results folder and code (README numbers only), and the cost arithmetic of section C2, which uses no weights and no audio. No stored logit has been opened for this experiment.

Results folder: `~/Documents/qmoe-results/2026-10-28_0119_moe-t23_code-4d6bde8/`. Amendments are new commits with a date, what changed and what had been seen by then.

## A. Section 11.2 of the round plan, verbatim

Source: `mit-projects/36-qmoe/blindspot-reviews/reviews/review-fable5-1-max-effort.md` (coordinator, commit `69afa7f8` of the AnpassenNN repo).

### 11.2 T23: a waveform expert, the second modality

| Item | Exact requirement |
|---|---|
| Waveform expert | An AudioSet-pretrained model that reads the raw waveform (no mel or fbank front end), from a public checkpoint with a published checksum, for example PANNs Res1dNet31 or the waveform branch of Wavegram-CNN14. Loaded with `torch.load(weights_only=True)` into our own model code; no code from the download is run. Frozen backbone with a protocol 3 head; W8A8 PTQ. If the frozen expert fails P1, one pre-declared fallback: fine-tune its last blocks on clean training clips (grouped inner validation, 3 seeds), then re-check P1 once. The choice is fixed in `ANALYSIS_PLAN.md` before any test score. |
| P1 (precondition) | (10 points rather than T15's 5, because a different modality is expected to score lower alone and the error-correlation clause is the real filter.) Within 10 points of MN10-Q8 on clean inner validation (fold 1) on two of three tasks, and its error correlation (phi) with MN10-Q8 below the AST-Q4's on those tasks. Else T23 stops. |
| P2 (precondition) | It costs less than AST-Q4 in priced bit-ops. Else T23 stops. |
| Systems (exactly two, pre-declared) | (a) MN10-Q8 then the waveform expert, margin gate, average rule; (b) MN10-Q8 then the waveform expert then AST-Q4, coarse to fine as in T22. Budgets chosen on clean inner validation. |
| H1 (primary) | On shifted audio a system beats the equal-cost hull and the best single model by 1.0 point on two of three tasks, majority of folds, Nadeau-Bengio, Holm over tasks and the two systems. |
| H2 | At equal cost the same system beats the two-tier MN10-Q8 to AST-Q4 cascade by 0.5 points on two of three tasks. |
| H3 | Escalation rises with shift. |
| Stop | P1 or P2 fails, or neither system is more than 0.5 points above the two-tier cascade on any task. |

Both agents report in the section 8 format. A pass is audited before it is called Accepted; a fail or stop needs the agent's own recompute from raw scores.

## B. Section 2 of the round plan and the section 11 differences, verbatim

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

## 11. Extension round (owner's go, 2026-10-28): the two mixture-of-experts ideas never tested properly

The round above closed every mixture-of-experts line except two, both stopped before their real test: T18's scale hierarchy (its coarse tier was a frozen model fed low-resolution input, which handicaps it) and T15's waveform expert (trained from scratch, too weak on ESC-50 and TAU). Both build experts that differ in what they see, the one route to specialization the evidence has not closed. Section 2's rules apply unchanged, with these differences: the campaign family is 19 (the 17 plus T22 and T23); branch from `moe/t21` (it has the fair queue and the FSD-free stage 3c tooling); GPU access only through `~/Documents/qmoe-work/fair_gpu_run.sh <tag> <command>` (FIFO queue, one MPS process at a time); free disk must stay above 25 GB (stop and report below it).


## C. Choices fixed here, before any download or score

### C1. The checkpoint

| Item | Value |
|---|---|
| Model | PANNs Wavegram-CNN14 (Kong et al., IEEE/ACM TASLP 2020, arXiv 1912.10211), the authors' `Wavegram_Cnn14`: a learned time-domain front end (strided Conv1d plus three Conv1d blocks) builds a 2-D "Wavegram" from the raw 32 kHz waveform, then the CNN14 2-D blocks. No mel or fbank front end anywhere. AudioSet mAP 0.389 (paper Table XI). |
| File | `Wavegram_Cnn14_mAP=0.389.pth`, 324,062,653 bytes |
| URL | `https://zenodo.org/api/records/3987831/files/Wavegram_Cnn14_mAP=0.389.pth/content` (Zenodo record 3987831, "PANNs ... (Pretrained Models)", licence CC BY 4.0; code MIT) |
| Published checksum | md5 `1e3506ab640371e0b5a417b15fd66d21` (Zenodo publishes md5 only; sha256 is computed after download and recorded in `run_manifest.json`) |
| Where | `~/Documents/qmoe-weights/panns-wavegram/`, a new empty directory. The file is read only by `python -I` with `torch.load(weights_only=True)` into our own `QWave/panns_wavegram.py` (commit `4d6bde8`); strict load after dropping keys of parameter-free modules (`spec_augmenter.*` if present, listed). No code from the download or from the authors' repository is run. |
| Why this one | The two published candidates both pass P2 (section C2). P1's accuracy clause is the binding risk: in T15 even CNN14 (mAP 0.431, log-mel) sat 5.8 to 5.9 points under MN10-Q8 on clean inner validation, and a waveform model is expected to sit lower. Wavegram-CNN14 has the higher AudioSet mAP of the two (0.389 against 0.365 for Res1dNet31), so it is the one with the better chance of passing the 10-point clause. This is the only checkpoint downloaded; no other candidate is scored on inner validation. |
| Functional check (before any task score) | Strict load passes; parameter count equals the paper's 80,991,759 (our random-init code already gives exactly this count); AudioSet top-3 tags on 6 clips with obvious content (ESC-50 dog, siren, rain, rooster, church bells, crying baby, chosen by file name) are printed to `logs/sanity_audioset.txt`. This reads no task label beyond the file name and produces no task score. |

### C2. P2 cost arithmetic (done before the download; `inputs/p2_costs.json`, script `t23_costs.py`)

Convention, as T15's `t15_costs.py` and `QWave/costs.py`: front end (the 16 to 32 kHz resampler, taps per output sample, plus `pre_conv0`, which reads the raw waveform and stays FP32) at 32 x 32 bits per MAC; every other Conv1d, Conv2d and fc1 at W8A8, 8 x 8 bits per MAC; one rescale per output element of a quantized layer at 32 x 32; plus the protocol 3 `esc` head (2048 inputs) bit-ops. MACs come from forward hooks on a zero input of our own model code, so they need no weights.

Check of the MAC convention: at 10 s and 32 kHz our hooks give 22.09 G MACs for Wavegram-CNN14 against the paper's 44.234 G "multi-adds" (ratio 2.003), and T15's hooks give 20.04 G for CNN14 against the paper's 42.220 G (ratio 2.107). The paper's multi-adds count about two operations per MAC. Res1dNet31's published 32.688 G multi-adds is therefore about 16.3 G MACs per 10 s.

| Task | Waveform expert W8A8, padded shape that runs (bit-ops per clip) | Own clip length (mean) | Conservative (paper multi-adds taken as MACs, x 64) | AST-Q4 stored | MN10-Q8 stored | Expert / AST-Q4 | P2 |
|---|---|---|---|---|---|---|---|
| ESC-50 (80,000 samples at 16 kHz) | 7.41e11 | 7.41e11 | 1.42e12 | 2.812e12 (`w4g128@512+esc`, stage 3c) | 3.48e10 | 0.26 (conservative 0.50) | pass |
| UrbanSound8K (padded 64,587) | 5.96e11 | 5.37e11 | 1.14e12 | 2.086e12 (`w4g128@400+esc`, stage 3c) | 2.53e10 | 0.29 (0.55) | pass |
| TAU (160,000) | 1.49e12 | 1.49e12 | 2.83e12 | 6.985e12 (`ast:w4g128+esc`, option B) | 6.83e10 | 0.21 (0.41) | pass |

The FSDKaggle2018 table the prompt points to (`2026-10-26_1408_moe-t21_code-731570e/tables/d1/costs_per_clip_summary.csv`) gives a mean of 2.60e12 over clips of mixed length and a maximum of 6.986e12 at the full 1,024 frames; the maximum equals TAU's stored 6.985e12, so the per-task numbers above are the right comparators. The primary price of the waveform expert is the padded shape that runs, because that is what is executed (AST-Q4 on UrbanSound8K is likewise priced at its fixed 400 frames). Res1dNet31 would cost about 0.20 of AST-Q4 at 5 s and also passes. P2 passes on all three tasks; it is re-checked with the loaded checkpoint (same code, so the numbers cannot change) and recorded in `tables/p2.csv`.

The expert costs 21 to 24 times MN10-Q8. Unlike T22's coarse tier, it is cheap relative to the AST, so a gate in front of it saves real cost.

### C3. How the expert reads a clip, and the PTQ

- Audio: stage 2's `load_clip` (mono, mean removed, resampled to 16 kHz). Conditions are applied to this 16 kHz waveform with stage 2's unchanged mixer and plan (`audio_shift.py`: `NoiseMaker`, `mix`, `field_assignment`, test seed 5001, validation seed 5002) and the final run's unchanged `corrupt.py` (reverb, clip, low-pass, deterministic per condition). This makes the audio identical, sample for sample, to what the stored MN10-Q8 and AST-Q4 heard.
- Shape: each clip is zero-padded at the end to one fixed 16 kHz length per task (ESC-50 80,000, UrbanSound8K 64,587, TAU 160,000; T15's shapes), then the batch is resampled to 32 kHz with `QWave.efficientat_mn.resample_to_32k` (torchaudio sinc, as MN10's front end). One input shape per task (one graph on MPS). The expert hears audio band-limited to 8 kHz, like every stored tier; it was trained on 32 kHz audio, and this mismatch is part of what P1 measures.
- Embedding: ReLU(fc1(max + mean over time of the frequency-averaged conv_block6 output)), 2048-d, eval mode, the authors' forward.
- W8A8 PTQ (`QuantWavegram`): per-output-channel symmetric round-to-nearest 8-bit weights on every Conv1d, Conv2d and fc1 except `pre_conv0`; 8-bit symmetric dynamic inputs (Conv1d one scale per clip over channels and time; Conv2d `effnet_quant.act_scale_dims`, one scale per clip for the 3 x 3 kernels; fc1 one scale per row); BatchNorm, ReLU and pooling in FP32; `pre_conv0` in FP32 and priced with the front end at 32 x 32. No calibration data, so nothing is fitted on any clip.
- Extraction set per task: FP32 backbone on clean (head training, repo convention); W8A8 on clean, the 12 noise cells (bg, pink, white at 20, 10, 5, 0 dB, all rows), the field rows of any missing field cell (none, since all 12 full cells are extracted), and the 9 corruptions (ESC-50 and UrbanSound8K); TAU: FP32 and W8A8 on every row of `tau2020mobile.csv` (device A and the shifted devices), clean.

### C4. Heads and scores

- The protocol 3 `esc` head of `scripts/benchmark.py`, trained on the clean FP32 embeddings of each fold's training clips, applied to the W8A8 embeddings of every condition (the repo convention, as T15's views and stage 3b/3c's MN10). Settings copied from T15's `t15_heads.sh`: `config/esc50.yaml` with `label_col=label` for ESC-50, `config/urban8k.yaml` for UrbanSound8K and TAU (TAU: `n_splits=5`, `recording_token=2`, training CSV of device-A rows built by option B's `make_train_csv.py`), `normalization_type=standard`, `inner_split_groups=recording`, CPU, 2 threads. Seeds: ESC-50 0 to 4, UrbanSound8K 0 to 2, TAU 0 to 2.
- Scaler: the StandardScaler of the fold's FP32 fit rows (stage 2 `score_eff.py` default, the one the stored tiers used). No refit choice.
- Scores: stage 2 `score_eff.py` (ESC-50, UrbanSound8K) and option B `score_optb.py` (TAU), which first re-run every fold's head on the training CSV's test rows and require the F1 in `summary.json`. Output per seed: `<cfg>__<cond>.npz` with `y`, `test_fold`, `test_logits`, `iv_idx_k`, `iv_logits_k`, `names_crc32`, the layout of the stored files.

### C5. The one pre-declared fallback (only if the frozen expert fails P1)

Fine-tune the last blocks: `conv_block6` and `fc1` (the last convolutional block and the embedding layer) plus a new linear classifier, starting from the AudioSet weights. Inputs: the frozen FP32 trunk's `conv_block5` outputs on clean training clips, cached per task. Training per (fold, seed), 3 seeds (0 to 2) on every task: the fold's training clips minus its protocol 3 inner-validation split (same grouping, same split as the heads); cross-entropy; AdamW, learning rate 1e-4 for `conv_block6` and `fc1`, 1e-3 for the classifier, weight decay 1e-4, batch 64; BatchNorm running statistics frozen (eval mode), affine parameters trained; the authors' dropout (0.2 before `conv_block6`, 0.5 before the classifier); mixup with alpha 0.2 on the cached block 5 outputs; at most 40 epochs, early stopping with patience 6 on inner-validation macro F1. Then the fine-tuned model is W8A8-quantized by the same recipe, the protocol 3 head is retrained on its FP32 clean embeddings (C4), and P1 is re-checked once on fold 1. Known bias, stated now: early stopping uses the same inner-validation clips P1 is measured on, so the fallback's P1 number is optimistic by an unknown amount. If P1 passes only after the fallback, every fold and seed is fine-tuned the same way, and its embeddings come from the frozen trunk (once per condition) followed by that fold's fine-tuned tail. If P1 still fails, T23 stops.

### C6. Stored arrays (read only, through `t13_common.py`) and what is missing

| Role | Stored expert | ESC-50 | UrbanSound8K | TAU |
|---|---|---|---|---|
| MN10-Q8 (tier 1) | `mn10` | stage 3b `scores_cnn` (clean, noise, field); final run (corruptions) | stage 3c unpadded `mn10unp` (clean, noise, field); final run (corruptions) | option B `mn10_w8a8__all.npz` (all devices) |
| AST-Q4 (last tier) | `astshort` (w4g128 at the clip's real length: 512 frames ESC-50, 400 UrbanSound8K) for ESC-50 and UrbanSound8K, `astw4` (w4g128, 1,024 frames, the real length) for TAU, as T21's design | stage 2 `scores_ast_short` (all 12 noise cells, field), final run (corruptions) | stage 2 (clean, bg 10 and 0 dB, field rows only of the other field cells), final run (corruptions) | option B `w4g128__all.npz` |
| Other statics for the hull | `mn20`, `astw4` (1,024 frames), `astfp32` | where available | where available | `mn20`, `astfp32` |

Missing conditions (from `availability.csv`): UrbanSound8K `astshort` has no full bg 20 and 5 dB, pink or white cells; `astw4` and `astfp32` have no corruptions on either task. Decision: restrict, no extraction. On a cell where AST-Q4 is missing, system (b), the AST-based statics and their averages are not scored; system (a) and every escalation rate (which need only MN10-Q8 and the expert's margins) are. The primary beds (field mix, TAU device shift) and the clean bed are complete for every expert in the systems.

### C7. Alignment check (hard gate before any score is computed; `inputs/alignment.json`)

Written by `t23_align.py` after extraction and head training, before any scoring script runs. The analysis and the recompute scripts read it and exit unless `all_ok` is true. Checked per task:

1. Clip identity: the `names_crc32` of every embedding sidecar equals the `names_crc32` of the stored MN10-Q8 and AST-Q4 score files (TAU: the option B file's crc and `n_a`), and the row count is equal.
2. Labels and folds: our score files' `y` and `test_fold` (TAU also `device`) equal the stored MN10-Q8's and AST-Q4's, for every seed.
3. Inner validation: for every (seed, fold), our `iv_idx_k` equals the stored MN10-Q8's and AST-Q4's (same rows, same order). A head whose split differs is retrained; nothing is scored until all match (T15 met this race).
4. Background draws: our `bg_plan.json` (bg clip and 16 kHz offset per clip) equals the reference plans of stage 3b `emb_cnn/{esc50,urban8k}`, stage 3c `emb_cnn/{esc50,urban8k_unpadded}` and stage 2 `emb_ast_short/{esc50,urban8k}`, for every clip in each; the references are first checked against each other.
5. Lengths: the true 16 kHz length of every clip equals T15's `emb/<task>/lengths.npy`.
6. Field rows: the assignment arrays (`field_assignment(n, 5001)`, `(n, 5002)`) are recomputed and n equals the stored row count; every field row of our files is finite.
7. Corruptions: deterministic (no per-clip draw); the mean level change per condition recorded in our sidecars is compared with the final run's sidecars where they record it (reported).

### C8. P1 procedure (fold 1 only, clean inner validation)

Extract FP32 and W8A8 clean embeddings for all three tasks; train fold-1 heads only (`only_folds=[1]`) for every seed of the task; score only the fold-1 inner-validation rows (T15's `t15_precond_heads.py` logic, adapted: dim 2048, TAU added). Per task: macro F1 of the deployed W8A8 expert and of the stored MN10-Q8 on the same rows (seed mean); phi (Pearson correlation of the two 0/1 error indicators) of expert vs MN10-Q8 and of stored AST-Q4 vs MN10-Q8 on the same rows, per seed then averaged. A task passes when expert F1 minus MN10-Q8 F1 is at least -10.0 points and phi(expert, MN10-Q8) is below phi(AST-Q4, MN10-Q8). P1 passes when two of three tasks pass. No test-fold row is read. Output `tables/p1.csv`, `tables/p1_by_seed.csv`.

### C9. The two systems (exactly these)

Margins are top-1 minus top-2 softmax probability. Thresholds are set on each (seed, fold)'s clean inner validation at the stated rate (stage 2 `thr_rate`), then held fixed on every test condition.

- (a) MN10-Q8 then the expert: escalate when MN10-Q8's margin is below the threshold at rate r_a; escalated clips output argmax of the mean of the two softmax outputs; the rest keep MN10-Q8. Budgets r_a in {5, 10, 20, 30, 50}% of clips.
- (b) MN10-Q8 then the expert then AST-Q4 (T22's rule): the margin of the tier that answered decides whether the next tier runs, and the output averages the tiers that ran. Tier 2 runs on clips with MN10-Q8 margin below the rate-r2 threshold; tier 3 runs on the tier-2 clips whose margin of mean(MN10-Q8, expert) is below a threshold set so that r3 of all inner-validation clips reach tier 3. T22's grid: r2 in {30, 50, 70}%, r3 in {5, 10, 20, 30}% (12 pairs).
- Budget choice (T21's rule, C1 there): for each system and task, the budget (pair) with the largest mean over (seed, fold) gain of the system's clean inner-validation macro F1 over the clean inner-validation hull (C10's member set, inner-validation rows) at the system's inner-validation cost, 200 random-mixing draws with `A.RNG_IV`; ties go to the cheaper budget.
- Cost of a system on a set of clips: c(MN10-Q8) + (tier-2 rate) x c(expert) + (tier-3 rate) x c(AST-Q4), with the per-task constants of C2 and C6 (stored constants for the stored tiers, as T13 and T15).

### C10. Comparators

- Hull: the time-sharing hull (stage 2 `hull_pair`) at the system's mean test cost over every static available on that condition: the singles MN10-Q8, MN20-Q8, the expert, AST-Q4, `astw4` (ESC-50, UrbanSound8K), `astfp32`, plus the always-average of every subset of two or three of {MN10-Q8, expert, AST-Q4}. The hull's F1 per (seed, fold) is the exact expected macro F1 of random time-sharing between its two endpoints (`mix_expected_f1`).
- Best single: of {MN10-Q8, expert, AST-Q4}, the one with the highest inner-validation macro F1 on the same condition family (field-mix validation rows, seed 5002 assignment, for the field bed; clean for clean). TAU has no shifted inner validation, so its device-shift bed uses clean device-A inner validation (as T15).
- Two-tier cascade at equal cost (H2 and the stop rule): MN10-Q8 then AST-Q4, margin gate, average rule (T21's design), with its clean inner-validation threshold set so that its clean inner-validation cost equals the system's clean inner-validation cost (escalation rate (C_sys - c_MN10) / c_AST). Equal cost is fixed on inner validation, so no test input sets it. Reported beside it, not used for a verdict: the two-tier at its own budget chosen by T21's rule, and its time-sharing value at the system's test cost.
- Also reported: random time-sharing (the hull), a random gate at the same escalation counts with the same decision rule, and always-average of the three.

### C11. Rules applied (beds: field mix for ESC-50 and UrbanSound8K, device shift for TAU; clean as the secondary bed)

All tests at fold level (seeds averaged within folds), paired t with the Nadeau-Bengio correction (ratio n_test / n_train of the fold, as T13's `nb`), two-sided. Reported with every difference: mean, 95% Nadeau-Bengio interval, folds positive, sign-flip floor (0.0625 with 5 folds, 0.002 with 10), clip-level bootstrap 95% interval (2,000 draws, clips resampled within folds, hull by linear interpolation of its endpoints), and the family-of-19 number min(1, 19 x raw p).

- H1 (primary): per system and task, D_hull = system minus hull and D_best = system minus best single on the shifted bed. A task passes for a system when both means are at least 1.0 point, both are positive in more than half the folds, and the Holm-adjusted p of the larger of the two Nadeau-Bengio p values is below 0.05, Holm over the 6 tests (3 tasks x 2 systems). H1 passes when one system passes on at least two of three tasks.
- H2: for the same system, D_two = system minus the equal-cost two-tier cascade on the shifted bed is at least 0.5 points and positive in more than half the folds, on at least two of three tasks.
- H3: with the clean inner-validation thresholds held fixed, the fold-mean test escalation rate (tier 2; for (b) also tier 3) rises at every noise step (clean, 20, 10, 5, 0 dB for each of bg, pink and white on ESC-50 and UrbanSound8K), the field mix is above clean, and TAU's shifted devices are above device A. Exceptions are listed. Escalation under the 9 corruptions and gate AUROC (1 - margin, for "MN10-Q8 wrong and expert right" and for "MN10-Q8 wrong") are reported, not tested.
- Stop: P1 or P2 fails, or neither system is more than 0.5 points above the equal-cost two-tier cascade (mean D_two above 0.5) on any task on the shifted bed.
- Secondary, not in Holm: the same differences on clean audio, every noise cell and every corruption (where the members exist, C6), the error correlation phi of every expert pair on test rows per condition, and the oracle (any expert right).

### C12. Separate recompute

`t23_recompute.py` reads only the raw score files (`test_logits`, `iv_logits`, `iv_idx`, `y`, `test_fold`, device) and the cost constants, and recomputes, with its own code (no import of the analysis script): the chosen budgets' thresholds, both systems' fold-level macro F1 and cost on the beds, the hull, the best single, the two-tier at equal cost, and the H1 and H2 differences. Its numbers are written next to the analysis's in `tables/recompute_check.csv`; a difference above 0.01 points is reported and blocks the verdict until explained.

### C13. Order, GPU, disk

P2 table and this plan committed; download and checksum; functional check; fold-1 clean extraction and P1 on all three tasks (GPU through `fair_gpu_run.sh claude-t23`, each process under about 80 minutes); P1 verdict written. Only if P1 and P2 pass: full extraction per task and condition (split into processes under 80 minutes), all heads, the alignment check, scores, analysis, recompute, README and manifest. Free disk is checked before every GPU process; below 25 GB the run stops and reports. Embedding caches stay under 10 GB (about 2.5 GB expected at float32).

## Amendments

### Amendment 1, 2026-10-28 07:05 (Irish time), after P1, before any extraction or score beyond P1

Seen so far: the frozen P1 table (fails: ESC-50 -12.8, UrbanSound8K -12.0, TAU -14.6 points; phi clause holds on all three), the fallback P1 table (passes: -9.4, -9.1, -7.7 points against the -10 bar; phi 0.32, 0.46, 0.37 against the AST-Q4's 0.51, 0.56, 0.47), and the fine-tune training curves of fold 1. No test-fold row has been scored. P1 therefore passes only through the pre-declared fallback, by margins of 0.6 and 0.9 points on two tasks, and with the optimistic bias stated in C5 (early stopping on the same inner-validation rows). Both P1 tables are reported.

The fallback changes the plumbing of C3, C4 and C7 as C5 already implies; nothing in C9 to C11 changes.

1. Seeds: three fine-tuned tails per fold (C5), so ESC-50 uses seeds 0 to 2 for every expert in the analysis (the stored ESC-50 seeds 3 and 4 are dropped). UrbanSound8K and TAU keep 0 to 2.
2. Tails for every (fold, seed) with the C5 recipe unchanged (fold 1 reuses the P1 tails). Heads per (fold, seed): `scripts/benchmark.py` with `only_folds=[f]`, the T15 settings of C4, on a CSV of tail (f, s)'s FP32 clean embeddings; run directory `outputs_ft/wg_<task>_s<s>_f<f>/esc`. `score_eff.py` and `score_optb.py` cannot be used (they assume one CSV per run); our scorer `t23_ft_embed.py` does their check first: each head re-run on its CSV's fold-f test rows must reproduce `summary.json` to 1e-6, then the scaler parameters are saved and the CSV is deleted (disk).
3. The deployed W8A8 expert is computed as the W8A8 trunk output up to `conv_block5` (cached in float16, in memory for noise and corruption conditions) followed by the fold's W8A8 quantized tail. The float16 rounding of the cache is a change from one end-to-end pass; it is measured once (frozen tail on the cache against the frozen end-to-end embedding, ESC-50 clean) and reported.
4. Every row is scored by its own fold's tail and head, under every condition; inner-validation rows of (f, s) by tail and head (f, s). Logits are written in the stored layout (C4); no per-condition embedding is written to disk.
5. Inner-validation rows are used three times: tail early stopping, protocol 3 head validation, and the budget and threshold choice. Test rows are used by none of these. The alignment check (C7) additionally asserts, for every (f, s), that the tail's training rows hold no fold-f test row (TAU: no row of another device) and records the counts.
6. C7 item 3 reads the inner-validation split of every (f, s) from `outputs_ft/.../fold_f/split.json`, for all folds; item 4 and 5 compare the `bg_plan.json` and `lengths.npy` written by the condition pass.
