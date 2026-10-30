# T25 analysis plan: the fixed T21 cascade against a wider equal-cost hull on fresh data

Written 2026-10-28 22:50 (Irish time) by the T25 agent and committed on branch `moe/t25` (worktree `~/Documents/qmoe-work/t25`, from `moe/t21` at `7957360`) before any T25 embedding of a test clip is extracted and before any T25 score or statistic exists. Nothing about the new CNNs on FSDKaggle2018 has been seen. What has been seen: T21's published results on FSDKaggle2018 (the cascade, MN10-Q8, AST-Q4, always-average and the narrow hull, `tables/d1/`), and a 40-clip code check of the extractor (MN10, MN04, DyMN04 embeddings of the first 40 pool rows on clean and bg 10 dB, written to a scratch folder and deleted; no head, logit or F1 was computed). The check showed that the extractor reproduces T21's MN10 FP32 embeddings to 5e-7, T21's clip lengths and T21's background plan exactly; MN10 W8A8 embeddings differ from T21's by 0.2% relative because the dynamic 8-bit activation rounding amplifies float-level differences from a different batch composition (the effect T21's amendment 1 documents for the AST). The cascade's MN10-Q8 is never re-extracted: it is T21's stored scores.

Amendments are new commits with a date, what changed and what had been seen by then.

## A. Section 13 of the round plan, verbatim

Source: `mit-projects/36-qmoe/blindspot-reviews/reviews/review-fable5-1-max-effort.md` (coordinator, last commit `4c9c8c0`, 2026-10-28).

## 13. T25: the wider hull on fresh data (owner's go, 2026-10-28)

The paper audit's main scientific finding: on FSDKaggle2018 the equal-cost hull holds only MN10-Q8, AST-Q4 and always-average. On the development data, adding small AudioSet CNNs to the hull turned stage 2's gains into losses (stage 3b), and averaging two small CNNs beat the hull by about 1.5 points under noise at about 1e11 bit-ops. T25 scores those systems on the fresh data. The cascade is not changed. This section was written and committed before any T25 score.

| Item | Exact requirement |
|---|---|
| Fixed cascade | T21's design, budget and scores unchanged (its D1 folds, seeds and test set). Nothing about the cascade is re-tuned. |
| Systems added to the hull | Every stage 3c CNN at simulated 8 bits with the protocol 3 head on clean training clips: MN04-Q8, MN05-Q8, MN20-Q8, DyMN04-Q8 (MN10-Q8 already in). Every pairwise average of two of these five CNNs. Each with T21's folds, seeds, scaler rule and stage 2's shift mixer. Costs priced with `QWave/costs.py` as in T21. |
| Primary H1-wide | On FSDKaggle2018's field mix, the fixed cascade minus the equal-cost hull of {T21's three systems plus the systems above} is at least 1.0 macro F1, fold-level paired t with Nadeau-Bengio, two-sided alpha 0.05, Holm over the datasets run. |
| Second dataset | If T24's dataset passes its precondition, the same test runs there and joins the Holm set; otherwise FSDKaggle2018 alone. |
| Secondary | The same on clean audio; the cascade against the best two-CNN average alone at its own cost; which systems form the widened hull. |
| Family | Transparency number 21. |
| What the paper may claim | Pass: the fresh-data gain holds against every small-CNN system and pair. Fail: the paper reports the gain over the narrow hull, the gap to the widened hull, and that a cheaper two-CNN average is competitive, in the results and the limitations. |
| Audit | A pass gets an independent audit agent before it is called Accepted; a fail needs the agent's own recompute from raw scores. |

## B. Section 2 of the round plan (rules for every slice), verbatim

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


Section 11 adds, for the extension rounds this one follows: GPU access only through `~/Documents/qmoe-work/fair_gpu_run.sh <tag> <command>`; free disk above 25 GB (stop and report below it). The coordinator's T25 prompt adds: proceed only while `memory_pressure` shows free at or above 25%; commits local only; recompute any fail from the raw score files with a separate script before reporting it.

## C. Choices fixed here, before any score

### C1. The fixed cascade (T21, nothing re-tuned, nothing re-scored)

T21's results folder `~/Documents/qmoe-results/2026-10-26_1408_moe-t21_code-731570e` (read only). Primary pool design of D1 (FSDKaggle2018): 5,310 manually verified clips, 5 stratified outer folds of 1,062 (`csv/d1/pool.csv`, copied unchanged into the T25 folder), head seeds 0 to 4, the official test set inside the pool as T21 built it. Cheap tier `mn10_w8a8` and strong tier `ast_w4g128` (own length), their stored score files `scores/d1/pool_mn10_w8a8_s{0..4}.npz` and `scores/d1/pool_ast_w4g128_s{0..4}.npz`. Scalers as T21 chose them (`tables/d1/primary_scaler_choice.csv`: refit and refit), budget as T21 chose it (`tables/d1/verdict.json`: 20%), thresholds recomputed from the stored clean inner-validation logits with stage 2's `thr_rate` exactly as T21 did, margin gate, average rule. The analysis rebuilds the cascade, MN10-Q8, AST-Q4 and always-average per (condition, seed, fold) and stops unless F1, cost and escalation rate equal T21's `tables/d1/primary_by_seed_fold.csv` to 1e-9, and unless the narrow-hull gains equal T21's H1 and H2 numbers (+2.83 field, +2.07 clean) to 1e-9.

### C2. The systems added to the hull (exact list)

Static CNNs at simulated 8 bits (stage 3c's `build_backbone("w8a8")`: per-output-channel 8-bit weights, 8-bit dynamic activations, fake quantization in float), each with the protocol 3 FP32 head:

| Code | Network | Weights (release v0.0.1) | Embedding |
|---|---|---|---|
| `mn04_w8a8` | EfficientAT MN04, `QWave.efficientat_mn` | `mn04_as_mAP_432.pt` | 384 |
| `mn05_w8a8` | EfficientAT MN05 | `mn05_as_mAP_443.pt` | 480 |
| `mn20_w8a8` | EfficientAT MN20 | `mn20_as_mAP_478.pt` | 1920 |
| `dymn04_w8a8` | EfficientAT DyMN04, `QWave.efficientat_dymn` | `dymn04_as.pt` | 384 |
| `mn10_w8a8` | already in (T21's stored scores) | `mn10_as_mAP_471.pt` | 960 |

The stage 3c loaders (rebuilt to 0.0 against the official EfficientAT code in stage 3c) are already on `moe/t21`: `redesign/idea6-stage3c` is an ancestor of `moe/t21`, and `QWave/efficientat_mn.py` and `QWave/efficientat_dymn.py` are byte-identical between the two branches, so nothing needs to be merged.

Pairwise averages: every pair of two of the five CNNs {MN04, MN05, MN10, MN20, DyMN04}, 10 pairs (`mn04+mn05`, `mn04+mn10`, `mn04+mn20`, `mn04+dymn04`, `mn05+mn10`, `mn05+mn20`, `mn05+dymn04`, `mn10+mn20`, `mn10+dymn04`, `mn20+dymn04`). Prediction: argmax of the mean of the two heads' softmax probabilities (T21's `avg` rule), each tier with its own chosen scaler, the head of the same seed for both.

The widened hull's system set is therefore 17 systems: T21's three (`mn10_w8a8`, `ast_w4g128`, always-average of the two), the four new statics, and the ten pairs. Nothing else is added (no CNN with the AST other than T21's always-average, no triples, no other precision).

### C3. Extraction, heads, scores (T21's code paths, model list widened)

- Audio: T21's reader (stage 2's `load_clip`, first 163,840 samples at 16 kHz) and stage 2's unchanged mixer with T21's conditions: clean; bg (UrbanSound8K bank), pink, white at 10 and 0 dB; the field mix (test seed 5001, validation seed 5002; the 5 dB cells extracted only for the rows the two assignments give them). `t25_extract_cnn.py` is `t21_extract_mn.py` with stage 3c's model dispatch; all four CNNs read the same waveform of each chunk. CPU, each clip at its own length, nice 10, 2 threads, one process (as T21 ran MN10). No GPU is used in T25 for D1: FSDKaggle2018 has many distinct clip lengths, which on MPS compiles one graph per length (the trap stage 3b hit), and DyMN's dynamic convolutions stay on the device T21's cheap tier used.
- Extraction set: `<m>_fp32` on clean (head training only), `<m>_w8a8` on every condition above, for m in MN04, MN05, MN20, DyMN04.
- Checks before any score (`t25_check_emb.py`): names checksum and row count equal to T21's pool, finite on every needed row, clip lengths equal to T21's MN10 lengths in every condition, background plan equal to T21's `bg_plan.json`.
- Heads: T21's `t21_train_heads.sh` call unchanged (protocol 3, `config/esc50.yaml`, `inner_split_groups=recording`, `recording_token=0`, CPU, 2 threads, seeds 0 to 4), trained on each CNN's clean FP32 embeddings of the pool (`t25_make_csvs.py`, `t25_train_heads.sh`). The inner-validation rows must equal T21's for every (seed, fold) (asserted).
- Scores: `t25_score.py` is `t21_score.py`'s pool path with the tier list widened (both scalers, every condition, clean and field inner-validation logits; every fold's head must reproduce its summary.json F1 first).
- Scaler per new tier: T21's rule (C1 of T21's plan): `refit` if its mean clean inner-validation macro F1 over folds and seeds is strictly higher than `fp32fit`, else `fp32fit`.

### C4. Pricing

Priced bit-ops per clip at the clip's own length, averaged over the clips scored, exactly T21's C5 for MN10 extended to the new CNNs:

- each CNN: EfficientAT front end `frontend_macs(n) x 32 x 32` + backbone at W8A8 with one rescale per output element at 32 x 32 (`backbone_bitops(feature_macs(frames), "w8a8", 1024)`; for DyMN04 its own `feature_macs` and `backbone_bitops`, which add the per-clip kernel aggregation at 32 x 32 as stage 3c priced it) + the FP32 head's bit-ops (`QWave.costs.cost` of the protocol 3 network at the CNN's embedding size and 41 classes);
- MN10-Q8, AST-Q4 and always-average: T21's `per_clip_costs`, unchanged;
- **a pairwise average costs the sum of both CNNs' full per-clip costs** (front end counted twice). At 10.24 s the front end is 22.0 G bit-ops against backbones of 12.2 G (MN04), 16.5 G (MN05), 13.7 G (DyMN04), 46.8 G (MN10) and 156.9 G (MN20), so pair costs are front-end heavy;
- the cascade's cost: T21's (cheap cost of every clip plus the AST cost of each escalated clip).

Descriptive only, after the verdict and never a test of a rule: the widened hull with each pair priced with one shared front end, and the unpriced view (MACs).

### C5. The hull (T21's code, systems list widened)

Per condition: mean test macro F1 and mean priced cost over the 25 (seed, fold) cells for each of the 17 systems; stage 2's `hull_pair` at the cascade's mean cost (the best static at or below that cost, or the best straight line between a system below and a system above it); per (seed, fold) the hull's F1 is the exact expected macro F1 of random time-sharing between the two endpoints at that mixing weight (`mix_expected_f1` through T21's `exp_f1`). This is T21's `analyze_design` hull code with `sp` holding 17 rows instead of 3. Code: `t25_analyze.py`.

### C6. Rules and statistics

- **Primary H1-wide** (section 13): on FSDKaggle2018's field mix, the fixed cascade minus the widened equal-cost hull is at least 1.0 macro F1 point, fold-level differences (seeds averaged within folds), paired t with the Nadeau-Bengio correction (T21's ratio, n_test / n_train = 0.25), two-sided, alpha 0.05, Holm over the datasets run. PASS when the mean difference is at least 1.0 and the Holm p is below 0.05. Reported beside it: the Nadeau-Bengio 95% interval, T21's clip-level bootstrap 95% interval (2,000 draws), folds positive, the sign-flip p and its floor (0.0625), and the family-of-21 number min(1, 21 x raw p).
- **Holm set**: FSDKaggle2018 (D1), plus the T24 dataset if and only if T24 reports that its precondition passed (section 13, "Second dataset"). If T24 is still running when D1 is finished, D1 is reported alone and T24 is said to be pending; Holm over one dataset is the raw p.
- **Family**: 21 (transparency number only).
- **Secondary H2-wide**: the same test on clean audio.
- **Secondary, the best two-CNN average alone at its own cost**: for each bed, the pair with the highest mean inner-validation macro F1 over folds and seeds on the inner validation of the same condition family (field-mix inner-validation rows for the field bed, clean for the clean bed; section 2's best-single rule applied to pairs). Reported: its test F1 and cost against the cascade's, and the cascade minus the pair with the same fold-level Nadeau-Bengio test (not in Holm). The pair that is best on test is reported as descriptive only.
- **Secondary, which systems form the widened hull**: the two endpoints and the mixing weight on each bed, and the table of all 17 statics (F1, priced cost, MACs) per condition.
- T21's narrow-hull numbers are recomputed by the same code as a check (C1), not as a new result.
- A second dataset (if T24's precondition passed): the same widened hull with T24's cascade scores and design reused unchanged; the four CNNs extracted, headed and scored on T24's folds by the same scripts, and `t25_analyze.py --base <T24 folder> --d <its label>`. Any adaptation the T24 folder layout needs is written as an amendment before its scores are read.

### C7. Fail check and audit

If H1-wide fails, `t25_recompute.py` (own softmax, scaler rule, threshold, sklearn macro F1, own hull over the 17 systems, Monte Carlo time-sharing, own Nadeau-Bengio test; only T21's audited AST and MN10 per-clip costs and the CNN MAC hooks are shared) recomputes H1-wide and H2-wide from the raw score files before the fail is reported. If H1-wide passes, the agent says so and stops; the coordinator launches the independent audit. The recompute runs in both cases.

### C8. Disk, memory, order

D1 end to end first. Free disk checked before each stage (stop below 25 GB); `memory_pressure` free at or above 25% before each stage (wait otherwise). T25 writes about 1 GB of embeddings and scores; no audio or mel cache is written. Then T24's status is read once.
