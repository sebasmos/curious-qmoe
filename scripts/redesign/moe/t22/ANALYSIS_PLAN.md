# T22 analysis plan: the scale hierarchy with a coarse expert trained on its own view

Written 2026-10-28 between 01:08 and 01:17 (Irish time) by the T22 agent and committed on branch `moe/t22` (worktree `~/Documents/qmoe-work/t22`, from `moe/t21` at `7957360`) before any coarse expert is trained, before any coarse-view embedding or score exists, and before any test statistic. What had been seen by then: the stored score arrays' shapes, fold sizes and inner-validation sizes (no accuracy, no F1 of any stored array was computed in this session), the T15/T18, T13, T20 and T21 README files and plans (which report earlier numbers), and the P2 cost numbers of section C3, which come from the architecture and the fixed padded lengths alone (no audio was read for them). Nothing else.

Amendments are new commits with a date, what changed and what had been seen by then.

## A. Rules copied verbatim from the round plan

Source: `mit-projects/36-qmoe/blindspot-reviews/reviews/review-fable5-1-max-effort.md` in AnpassenNN (last commit touching it `69afa7f8`), sections 11 (header), 11.1 with its cost note, and 2.

## 11. Extension round (owner's go, 2026-10-28): the two mixture-of-experts ideas never tested properly

The round above closed every mixture-of-experts line except two, both stopped before their real test: T18's scale hierarchy (its coarse tier was a frozen model fed low-resolution input, which handicaps it) and T15's waveform expert (trained from scratch, too weak on ESC-50 and TAU). Both build experts that differ in what they see, the one route to specialization the evidence has not closed. Section 2's rules apply unchanged, with these differences: the campaign family is 19 (the 17 plus T22 and T23); branch from `moe/t21` (it has the fair queue and the FSD-free stage 3c tooling); GPU access only through `~/Documents/qmoe-work/fair_gpu_run.sh <tag> <command>` (FIFO queue, one MPS process at a time); free disk must stay above 25 GB (stop and report below it).

### 11.1 T22: the scale hierarchy with a coarse expert trained on its own view

| Item | Exact requirement |
|---|---|
| Coarse expert | An EfficientAT CNN (MN04 or MN10 architecture, AudioSet weights as the start) fine-tuned on a low-resolution view: audio resampled to 8 kHz, a mel front end with at most 64 bands and a hop at least twice MN10's. Trained per fold on clean training clips only (SpecAugment and mixup allowed), early stopping and learning rate on grouped inner validation, 3 seeds, then W8A8 PTQ with the stage 3c recipe. The view, the architecture and the training recipe are fixed in `ANALYSIS_PLAN.md` before any test score. |
| Hierarchy | Tier 1 the coarse expert at 8 bits; tier 2 MN10-Q8 at full resolution (stored arrays); tier 3 AST-Q4 (stored arrays). The margin of the tier that answered decides whether the next tier runs; the output averages the tiers that ran. Thresholds come from a grid of budget pairs (tier 2 rate 30, 50, 70%; tier 3 rate 5, 10, 20, 30%) chosen on clean inner validation. |
| P1 (precondition) | The coarse expert alone is within 5 points of MN10-Q8 on clean inner validation (fold 1, macro F1) on two of the three tasks. Else T22 stops. |
| P2 (precondition) | The coarse expert costs at most 50% of MN10-Q8 in priced bit-ops, front end included. Else T22 stops. |
| H1 (primary, T18's rule) | On shifted audio (field mix for ESC-50 and UrbanSound8K, device shift for TAU) the hierarchy beats the equal-cost hull, the best single model and the always-average of all tiers by 1.0 point on two of three tasks, majority of folds, Nadeau-Bengio, Holm over tasks. |
| H2 (does the hierarchy add anything) | At equal cost the hierarchy beats the two-tier cascade MN10-Q8 to AST-Q4 (the T21 design, its own budget picked on clean inner validation) by 0.5 points on two of three tasks. |
| H3 | Escalation to finer scales rises with shift at every noise step, field above clean. |
| Stop | P1 or P2 fails, or no budget pair is more than 0.5 points above the two-tier cascade on any task. |

Cost note, written before any T22 data: from T21's per-clip cost table (`2026-10-26_1408_moe-t21_code-731570e/tables/d1/costs_per_clip_summary.csv`), MN10-Q8 costs 0.031 T per clip on average against 2.60 T for AST-Q4, so MN10 is about 1.2% of one AST pass. A coarse tier in front of MN10 can therefore save at most about 1% of an AST pass. T22 can only win through accuracy, by averaging in a third view, and earlier evidence for that is weak (T15's extra views added +0.07, 0.00 and -0.68; T13's triples about +0.4). The coordinator's odds for H1 are close to zero. It runs because the owner asked for the idea to be tested properly.


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

## B. Data, tiers and conditions (what is stored and what is not)

Tasks: ESC-50 with 50 classes (5 official folds, 2000 clips of 5 s), UrbanSound8K (10 official folds, 8732 clips up to 4.04 s), TAU 2020 Mobile (5 location-grouped folds over device A, 23035 clips of 10 s; 14400 device A).

Tier 2 is MN10-Q8 (EfficientAT MN10, W8A8, FP32 `esc` head), tier 3 is AST-Q4 (`w4g128` at 1024 frames, FP32 `esc` head), both read from the stored arrays listed in `2026-10-25_0145_moe-t13_code-eaaedc9/inputs/file_list.csv` through `scripts/redesign/moe/t13_common.py` (UrbanSound8K tier 2 is the exact-length `mn10unp` model). Nothing is retrained for tiers 2 and 3.

Seeds. Section 11.1 says 3 seeds. The stored ESC-50 arrays have seeds 0 to 4 and section 2 lists 0 to 4 for ESC-50, but the coarse expert is trained for seeds 0, 1, 2 only. Every tier and every comparator therefore uses seeds 0, 1, 2 on all three tasks (seed s of the coarse expert is paired with seed s of the stored heads and uses seed s's inner split). This is a restriction of the design to the seeds the new tier has.

Stored conditions (from `availability.csv`; checked again in the alignment step):

| Task | Tier 2 (MN10-Q8) | Tier 3 (AST-Q4) | All three tiers |
|---|---|---|---|
| ESC-50 | clean, 12 noise cells (bg, pink, white at 20, 10, 5, 0 dB), field, 9 corruptions | clean, 12 noise cells, field | clean, 12 noise cells, field |
| UrbanSound8K | clean, 12 noise cells, field, 9 corruptions | clean, bg 10 dB, bg 0 dB, field | clean, bg 10, bg 0, field |
| TAU | clean (device A rows), shift (other devices) | clean, shift | clean, shift |

Missing in the stored arrays: AST-Q4 under the 9 corruptions (ESC-50 and UrbanSound8K) and under 10 of the 12 UrbanSound8K noise cells. They are not extracted here (the AST is not part of T22's new work, and the primary beds exist). The coarse expert is extracted under every condition tier 2 has (clean, 12 noise cells, 9 corruptions; TAU all clips once). The hierarchy's accuracy is scored only where all three tiers exist. The escalation rates of H3 need tiers 1 and 2 only (the gates read p1 and the average of p1 and p2), so H3 uses all 12 noise steps on ESC-50 and UrbanSound8K. Corruption cells are descriptive (escalation rates, tier 1 and tier 2 accuracy).

The field mix is never extracted as such: the coarse expert is extracted on clean plus the 9 field cells, and the field rows are assembled by stage 2's `field_assignment` (test seed 5001, inner-validation seed 5002) through `analyze.ScoreDir`, exactly as for the stored tiers.

## C. The coarse expert (fixed here, before any training)

### C1. View

`scripts/redesign/moe/t22/t22_view.py`. The 16 kHz waveform every tier reads (stage 2's `audio_shift.load_clip`: mono, mean removed, 16 kHz; then the condition: `NoiseMaker` and `mix` for bg, pink and white, `corrupt.corrupt` for the corruptions, called exactly as the stage 3b and final-round extractors call them), resampled to 8 kHz with torchaudio's default sinc resampler; pre-emphasis 0.97; symmetric Hann window of 200 samples (25 ms), n_fft 256, hop 160 samples (20 ms, twice MN10's 10 ms); centred STFT with reflect padding; power; 64 Kaldi mel filters from 0 to 4 kHz (`get_mel_banks`, the routine MN10's front end uses); `(log(mel + 1e-5) + 4.5) / 5`.

Padding: after the condition is applied to the clip at its own length (so the background draws in `bg_plan.json`, which depend on the clip length, match the stored ones), the 16 kHz waveform is zero-padded at the end to one length per task: ESC-50 80000 samples (no change), UrbanSound8K 64587 (the longest clip, the stage 3b padded length), TAU 160000 (no change). Training and inference both use the padded input, so each task has one input shape (64 x 250, 64 x 202, 64 x 500). The coarse expert's cost is counted at the padded length, which over-prices it on UrbanSound8K (conservative for P2).

### C2. Architecture and training recipe

- Architecture: EfficientAT **MN10** (width 1.0), features initialised from the released AudioSet checkpoint `mn10_as_mAP_471.pt` (loaded with `torch.load(weights_only=True)` through `QWave.efficientat_mn.load_mn`), the AudioSet classifier dropped, a new head Dropout(0.2) then Linear(960, K) on the global average pool. MN04 is the fallback that is not used (P2 ratio about 0.10, section C3); MN10 is chosen because P2 holds with room and P1 is the harder precondition.
- Rows: for outer fold f and seed s, the inner-validation rows are the stored `iv_idx_f` of seed s in MN10-Q8's clean npz (the same rows every stored tier used; T20's `inner_split`). Fit rows are the training rows (fold != f; TAU also device A only) minus those inner-validation rows. Test rows are never read during training. Clean audio only.
- Input: a cache of the clean coarse-view log-mels (all ESC-50 and UrbanSound8K clips, the 14400 TAU device A clips) computed once on the CPU with `t22_view.logmel` from the padded clean waveform, float32, under `~/Documents/qmoe-work/t22-cache` (about 2.5 GB, deleted at the end).
- Loss: cross-entropy with inverse class-frequency weights (fit rows) and label smoothing 0.1. Mixup on each batch with probability 0.5, lambda from Beta(0.3, 0.3), loss mixed with the same lambda.
- SpecAugment per clip: one frequency mask of width uniform in 0 to 8 bands (of 64) and one time mask of width uniform in 0 to 10% of the frames, masked values set to 0.
- Optimiser: AdamW, weight decay 0.01, batch 64, body learning rate from the grid {1e-4, 3e-4, 1e-3} and head learning rate 10 times the body's; linear warm-up over the first epoch, then cosine decay to 0 at the last epoch. Batch norm in training mode.
- Epochs: at most 60 (ESC-50), 30 (UrbanSound8K), 20 (TAU). Inner-validation macro F1 of the FP32 model after every epoch; the checkpoint is the epoch with the best inner-validation macro F1 (ties to the earlier epoch); early stop after 10 epochs without improvement.
- Learning rate choice: per task, on fold 1, seed 0, the three body learning rates are trained and the one with the highest best-checkpoint inner-validation macro F1 is kept (ties to the lower rate). It is then fixed for every fold and seed of that task. P1 is measured on the same fold-1 inner validation; the stored MN10-Q8 heads also used that inner validation for early stopping, so both sides carry the same kind of selection.
- Seeds 0, 1, 2: `torch.manual_seed(seed)` before the head is built; batch order and augmentation from generators seeded with 1000 + seed and 2000 + seed.
- Quantisation: after training, W8A8 PTQ with the stage 3c recipe (`QWave.efficientat_mn.QuantCNN`: per-output-channel 8-bit weights, 8-bit dynamic activations on every Conv2d and squeeze-excitation Linear in `.features`; batch norm, activations and pool in FP32). The new Linear head stays FP32, as the stored tiers' `esc` heads do. Tier 1 is this W8A8 model.
- Training runs on MPS through `~/Documents/qmoe-work/fair_gpu_run.sh claude-t22 ...`, each GPU process under about 80 minutes (folds split across processes as needed). Inference (P1 scoring and every extraction) runs on the CPU (nice 10, 2 threads) with the W8A8 model, in one code path, so every number comes from the same kernels.

### C3. Costs (priced with the repo's rules, front end included)

`scripts/redesign/moe/t22/t22_costs.py`, output `inputs/p2_costs.json`. Priced bit-ops per clip = front-end multiply-add equivalents x 32 x 32 (resampler taps per output sample, pre-emphasis, per frame the 200-sample window, a 256-point real FFT priced by `effnet_quant.rfft_macs`, two per power bin, the nonzero mel weights, a log and a scale per mel value; the counting rule of `QWave.efficientat_mn.frontend_macs`) + backbone MACs x 8 x 8 + one rescale per backbone output element x 32 x 32 (forward hooks on every Conv2d and Linear of `.features` at 64 x frames) + the FP32 head priced by `QWave.costs.cost`. The unpriced view (MACs) is reported beside it. The counting reproduces the stored MN10-Q8 cost exactly when applied to MN10's own front end and input (ESC-50 3.4816e10, TAU 6.8339e10, UrbanSound8K padded 2.8364e10).

P2 is judged against the stored MN10-Q8 costs the earlier rounds used (stage 3c `analysis/<task>/costs.json`, key `cnn:mn10:w8a8:esc`, UrbanSound8K `cnn:mn10unp:w8a8:esc`; option B for TAU). Computed before any training, from the architecture and the padded lengths only:

| Task | Coarse MN10 (bit-ops per clip) | MN10-Q8 stored | Ratio | MN04 ratio (not used) |
|---|---|---|---|---|
| ESC-50 | 8.01e9 | 3.48e10 | 0.230 | 0.103 |
| UrbanSound8K | 6.55e9 | 2.53e10 | 0.259 | 0.115 |
| TAU | 1.58e10 | 6.83e10 | 0.231 | 0.104 |

P2 (at most 0.5 on every task) therefore holds for MN10. This number is final unless the architecture changes, which would be an amendment.

Tier costs in the hierarchy and the comparators are task-level means per clip: coarse (constant, padded), and the stored keys of `t13_common.cost_table` for MN10-Q8, MN20-Q8, AST-FP32, AST-Q4 and the short AST. A clip's cost is the sum of the tiers that ran on it.

## D. Preconditions

- **P1**: on fold 1, clean inner validation (the stored `iv_idx_1` of each seed), macro F1 of the W8A8 coarse expert minus macro F1 of MN10-Q8 (stored `iv_logits_1`, clean), each averaged over seeds 0, 1, 2. P1 holds on a task if the difference is at least -5.0 points; P1 passes if it holds on at least two of the three tasks. The FP32 coarse expert is reported beside it. If P1 fails, T22 stops: no further training, no extraction of test scores, report.
- **P2**: section C3. If it had failed, T22 would stop before training.

## E. The hierarchy (exactly as 11.1, operational details fixed here)

For a clip: p1 = softmax of tier 1. Gate 1: if margin(p1) < tau1, tier 2 runs and p12 = (p1 + p2) / 2, else the output is p1. Gate 2: if margin(p12) < tau2, tier 3 runs and the output is p123 = (p1 + p2 + p3) / 3, else p12. Margin is the top-1 minus top-2 probability of the current output; "the tier that answered" is read as the running average, as in T18's primary. The variant in which gate 2 reads tier 2's own margin is reported as a secondary arm, never tested.

Thresholds, per seed and outer fold, from the clean inner-validation clips of that fold: tau1 is the quantile of margin(p1) such that a fraction r2 escalates to tier 2 (stage 2's `thr_rate`); tau2 is the quantile of margin(p12) among the clips that reached gate 2 such that a fraction r3 / r2 of them escalates, so a fraction r3 of all inner-validation clips reaches tier 3. Grid: r2 in {0.30, 0.50, 0.70} x r3 in {0.05, 0.10, 0.20, 0.30}, 12 pairs. The thresholds are then applied unchanged to the test rows of every condition. TAU: clean device A inner validation.

Pair choice (one per task, before any test score is read): the pair with the largest clean inner-validation macro F1 minus the clean inner-validation hull at the pair's mean inner-validation cost (the hull of section F built on inner-validation F1 of the same members); ties to the cheaper pair. All 12 pairs are reported on every cell.

## F. Comparators

- **Equal-cost hull**: stage 2's `hull_pair` at the hierarchy's mean test cost over the static members coarse-Q8, MN10-Q8, MN20-Q8, AST-Q4, AST-FP32, the short AST w4g128 (ESC-50 and UrbanSound8K only; not stored for TAU), and the always-average points of each pair of the three tiers and of all three tiers. Built per condition on that condition's test F1 (seed-fold mean), then evaluated per seed and fold (expected F1 of random time-sharing, `mix_expected_f1` as in T21), as T13 and T21 do.
- **Best single at equal or lower cost** (T18's rule): among the singles coarse-Q8, MN10-Q8, MN20-Q8, AST-Q4, AST-FP32, short AST whose mean cost is at most the hierarchy's, the one with the highest inner-validation macro F1 on the same condition family (field mix inner validation for ESC-50 and UrbanSound8K). TAU has no shifted inner validation (inner validation is device A only), so TAU uses clean device A inner validation, forced by the data (as in T15/T18).
- **Always-average of all tiers**: (p1 + p2 + p3) / 3 on every clip, cost c1 + c2 + c3.
- **Two-tier cascade (H2)**: MN10-Q8 then AST-Q4, gate on MN10-Q8's margin, average rule, budget from {5, 10, 20, 30, 50%} chosen on clean inner validation by T21's rule (largest inner-validation gain over the inner-validation hull of {MN10-Q8, AST-Q4, their average}, ties to the cheaper). Its cost-matched variant: the same cascade with its threshold set on clean inner validation so that its mean inner-validation cost equals the hierarchy's chosen pair's mean inner-validation cost (per seed and fold).
- Reported, never tested: random time-sharing, a random gate at the same escalation counts, oracle (best tier per clip), error correlation (phi) of tier 1 with tiers 2 and 3, and the own-margin gate variant.

## G. Tests and decision rules

Primary metric: macro F1 (accuracy reported beside it). Statistics: fold level (seeds 0 to 2 averaged within a fold), paired t with the Nadeau-Bengio correction (ratio n_test / n_train of the outer split; TAU: shifted test rows over device A training rows), two-sided; sign-flip floor 0.0625 with 5 folds, 0.002 with 10; clip-level bootstrap 95% intervals for the hull contrast (2000 draws, seed 20261028).

- **H1 (primary)**, bed: field mix (ESC-50, UrbanSound8K), device shift (TAU). A task passes if (a) the hierarchy minus each of the three comparators (hull, best single at equal or lower cost, always-average of all tiers) is at least +1.0 point on the fold mean; (b) each contrast is at least +1.0 in a majority of folds (3 of 5, 6 of 10); (c) the largest of the three Nadeau-Bengio p values, Holm-adjusted over the three tasks, is below 0.05. H1 passes if at least two of the three tasks pass. The family-of-19 number (19 x p, capped at 1) is reported beside the Holm value; the verdict uses Holm over tasks.
- **H2**, same bed: the hierarchy minus the two-tier cascade at its own clean-validation budget. A task passes if the fold mean is at least +0.5, at least +0.5 in a majority of folds, and Nadeau-Bengio p (Holm over the three tasks) is below 0.05, provided the hierarchy's mean test cost is at most 1.05 times the two-tier cascade's (the "equal cost" clause). If the hierarchy costs more than that, the same test is run against the cost-matched two-tier variant instead. Both contrasts and both costs are always reported. H2 passes on two of three tasks.
- **H3**: with the chosen pair and the clean thresholds, both escalation rates (fraction reaching tier 2, fraction reaching tier 3) rise at every noise step: for bg, pink and white, clean to 20, 20 to 10, 10 to 5, 5 to 0 dB (12 steps) on ESC-50 and UrbanSound8K, and field above clean; TAU: every non-A device above device A. Judged on the fold mean of seed-averaged rates; folds rising are tabled. H3 holds on a task if every step rises for both rates; exceptions are listed. H3 passes if it holds on all three tasks.
- **Stop**: P1 or P2 fails; or, on the H1 bed, no budget pair of the 12 is more than 0.5 points (fold mean) above the cost-matched two-tier cascade (matched to that pair's inner-validation cost) on any task. Evaluating the 12 pairs on test for the stop rule does not change the pair chosen in section E.

Verdict per rule (P1, P2, H1, H2, H3, Stop) with PASS, FAIL or STOP; nothing else is a verdict.

## H. Alignment check (hard, before any scoring)

`t22_align.py` writes `inputs/alignment.json` and the scorer refuses to run unless it says `"ok": true`. For each task and seed it compares the coarse expert's score files with both stored sources (tier 2: stage 3b, stage 3c unpadded, the final round and option B; tier 3: idea 5 stage 2 and option B), clip by clip:

1. `names_crc32` of the CSV row order equals the stored value of every tier-2 and tier-3 file used.
2. `y`, the fold vector and (TAU) the device vector are identical arrays.
3. Every `iv_idx_f` of seed s is identical to tier 2's and tier 3's.
4. The background plan: the coarse extraction's `bg_plan.json` (background clip and offset per clip) equals, for every clip, the stored plans of the tier-2 source (stage 3b `emb_cnn/esc50`, stage 3c `emb_cnn/urban8k_unpadded`) and of the tier-3 source (idea 5 stage 2 `emb/esc50`, `emb/urban8k`).
5. Same audio: the native MN10 W8A8 embedding recomputed through this code path (same loader, same mixer, same corruption) on 64 clips per task in clean, one noise cell and one corruption matches the stored embedding (max absolute difference reported; the bar is 1e-4).
6. Field assignment arrays equal stage 2's for the task's row count.

Any difference: scoring is refused and the difference is reported.

## I. Order of work and recompute

1. This plan, committed. 2. P2 (`t22_costs.py`, done from the architecture). 3. Clean coarse-view cache (CPU). 4. Fold 1: learning-rate probe and seeds 0 to 2 for all three tasks; P1 from inner-validation logits only (`t22_p1.py`); stop if P1 fails. 5. Only if P1 and P2 pass: all folds and seeds, then extraction under every condition of section B, alignment check, scoring (`t22_score.py`), analysis (`t22_analyze.py`), verdict. 6. A separate script (`t22_recompute.py`) reads only the raw score files (coarse npz files and the stored npz files) and recomputes the primary numbers (hierarchy F1, comparators, H1 and H2 contrasts) with its own code; any difference above 0.01 points is reported.

Results folder: `~/Documents/qmoe-results/<YYYY-MM-DD_HHMM>_moe-t22_code-<plan commit>/` with `ANALYSIS_PLAN.md`, `run_manifest.json`, `README.md`, `scripts/`, `logs/`, `inputs/`, `runs/` (checkpoints and training logs), `scores/`, `tables/`.

## Amendments

**Amendment 1 (2026-10-28 01:22, before any GPU training, extraction or test score).** Batch-norm momentum during fine-tuning is set to 0.1 (PyTorch's default) instead of EfficientAT's 0.01. Reason: with 0.01 the running statistics stay at the AudioSet values of the 128-band, 32 kHz input for several hundred steps, so the eval-mode model that the checkpoint rule scores is wrong for the first epochs of a short fine-tune on a different view. Seen by then: one CPU smoke epoch of the trainer on ESC-50 fold 1, seed 0 (inner-validation macro F1 0.0008 after one warm-up epoch, a constant prediction), used only to check the code and timing (about 10 clips per second on the CPU, so training runs on MPS as planned). No test row was read.

**Amendment 2 (2026-10-28 01:29, before any GPU training, any check of section H or any test score).** Timestamps of the header and of amendment 1 corrected to the commit times (01:17 and 01:22). Section H item 5 made precise: the stored embeddings were computed on MPS and this check runs on the CPU, so the bar is on the FP32 MN10 embedding where one is stored (clean and bg 0 dB on ESC-50 and UrbanSound8K, clean on TAU): maximum absolute difference divided by the stored embedding's maximum absolute value at most 1e-4. The corruption cell (reverb 0.6 s) has only W8A8 embeddings stored, where an activation rounding step can flip between devices; its bar is a per-clip cosine similarity of at least 0.9999. Added item 7: the extraction's clean coarse view equals the training cache exactly on the same clips. Seen since amendment 1: the inner-validation macro F1 of the stored MN10-Q8 and AST-Q4 on ESC-50 fold 1, seed 0 (90.78 and 95.97), printed by a timing run of the P1 script on the smoke checkpoint. No test row was read.
