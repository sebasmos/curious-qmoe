# T23 raw-waveform expert in the margin-routed cascade

Question: does an AudioSet-pretrained raw-waveform expert (PANNs Wavegram-CNN14, W8A8) add to the margin-routed quantized cascade?
Pre-registered plan: `ANALYSIS_PLAN.md` (commit 5b913db, written before the download), amendment 1 (commit d36d3fe, after the P1 tables only).
Branch `moe/t23` in `~/Documents/qmoe-work/t23` (worktree of curious-qmoe), not pushed.

## Verdict

| Rule | Result |
|---|---|
| P2 (cost below AST-Q4) | Pass on all three tasks. Wave W8A8 is 0.26, 0.29, 0.21 of AST-Q4's priced bit-ops (ESC-50, UrbanSound8K, TAU); 0.50, 0.55, 0.41 with the published multiply-add count. `tables/p2.csv`. |
| P1 (fold 1, clean inner validation) | Frozen expert fails (gaps -12.8, -12.0, -14.6 points against MN10-Q8). The pre-declared fallback (fine-tuned conv_block6 and fc1) passes by the pre-registered rule: -9.4, -9.1, -7.7 against the -10 bar, phi with MN10-Q8 0.32, 0.46, 0.37 against AST-Q4's 0.51, 0.56, 0.47. The ESC-50 and UrbanSound8K margins are 0.6 and 0.9 points, and early stopping used the same inner-validation rows, so the pass is optimistic. `tables/p1_frozen.csv`, `tables/p1_fallback.csv`. |
| H1 | Fails. No task passes for either system. |
| H2 | Fails. Both systems are below the equal-cost two-tier MN10-Q8 to AST-Q4 cascade on every shifted bed. |
| H3 | Holds on all three tasks (39 of 39 H3 steps on ESC-50 and UrbanSound8K, TAU shift above device A, no exception). |
| Stop | Fires: neither system is more than 0.5 points above the two-tier cascade on any task. |

## Primary numbers (shifted beds, 3 seeds, Nadeau-Bengio 95% intervals over folds)

System (a) is MN10-Q8 then wave, system (b) is MN10-Q8 then wave then AST-Q4. D = system minus comparator, macro F1 points.

| Task (folds) | System | F1 | D hull [NB] | D best single [NB] | D two-tier equal cost [NB] | folds D two-tier > 0 |
|---|---|---|---|---|---|---|
| ESC-50 field (5) | a | 84.70 | -0.68 [-2.77, 1.41] | -0.82 [-4.34, 2.69] | -1.88 [-4.59, 0.82] | 0 |
| ESC-50 field (5) | b | 87.01 | +0.59 [-1.06, 2.25] | +1.49 [-2.07, 5.06] | -1.01 [-3.16, 1.14] | 1 |
| UrbanSound8K field (10) | a | 80.68 | -0.11 [-1.23, 1.00] | -0.49 [-3.07, 2.09] | -0.43 [-1.24, 0.37] | 2 |
| UrbanSound8K field (10) | b | 82.07 | +0.55 [-0.78, 1.88] | +0.90 [-0.96, 2.75] | -0.66 [-1.84, 0.52] | 5 |
| TAU shift (5) | a | 45.36 | -0.39 [-1.49, 0.71] | +0.03 [-0.93, 0.98] | -1.25 [-3.00, 0.50] | 1 |
| TAU shift (5) | b | 48.45 | +0.61 [-0.73, 1.95] | +3.13 [1.86, 4.40] | -0.84 [-2.64, 0.96] | 1 |

Holm over 6 and family-of-19 adjusted p are 1.0 for every H1 test. Budgets chosen on clean inner validation: (a) 5%, 10%, 30%; (b) (30%, 10%), (30%, 20%), (50%, 30%).
Full tables: `tables/primary.csv`, `tables/<task>_by_condition.csv`, `_by_seed_fold.csv`, `_budget_selection.csv`, `_two_tier_equal_cost.csv`, `_gate_by_seed_fold.csv`, `tables/verdict.json`.

Independent recompute (`t23_recompute.py`, reads only score files and cost constants): `tables/recompute_check.csv`, largest difference 0.0 points on all 12 rows.

## What was run

1. P2 priced before any download (`inputs/p2_costs.json`). Checkpoint Zenodo 3987831, md5 verified, read only with `python -I` and `torch.load(weights_only=True)` into our own model code (`scripts/panns_wavegram.py`).
2. Frozen expert on fold 1: protocol 3 heads on clean FP32 embeddings, W8A8 embeddings, P1 table. Failed.
3. Fallback (plan C5): conv_block6, fc1 and a linear classifier fine-tuned on cached conv_block5 outputs of the frozen trunk, early stopping on the stored MN10-Q8 inner validation of the same seed. P1 re-checked once. Passed.
4. Full run (amendment 1): tails for every (fold, seed), 3 seeds per fold on all tasks; protocol 3 head per (fold, seed); one GPU condition pass per task computes the W8A8 trunk under every stored condition (clean, 12 noise cells, 9 corruptions; TAU all devices) and scores each row with its own fold's tail and head.
5. Alignment gate before scoring (`inputs/alignment_pre_<task>.json`) and after (`inputs/alignment.json`, all_ok): names, labels, folds and inner-validation order equal the stored MN10, AST and MN20 files of every seed; noise draws equal stage 3b's `bg_plan.json` clip by clip; corruption level changes equal the final run's sidecars; lengths equal T15's. Tail training rows hold no fold-f test row, no inner-validation row and (TAU) no other-device row for all 60 (fold, seed) tails.

## Checks on the plumbing

- The compact W8A8 tails (int8 codes plus scales, `t23_tailq.py`) reproduce the P1 W8A8 embeddings exactly (max abs difference 0.0, all tasks, MPS).
- Float16 trunk cache against the frozen end-to-end pass (frozen tail, clean): relative L2 difference 0.47% (ESC-50), 0.43% (UrbanSound8K), 0.55% (TAU device A). `inputs/cache_check_<task>.json`.
- The scorer dry run on ESC-50 fold 1 reproduced the fallback P1 inner-validation F1 to two decimals.
- Every head was re-run on its training CSV's test rows and matched `summary.json` to 1e-6 before any logit was written.

## Deviations

- P1 passed only through the pre-declared fallback, with the early-stopping bias stated in C5.
- ESC-50 uses seeds 0 to 2 for every expert (stored seeds 3 and 4 dropped), because the fallback has three tails per fold (amendment 1).
- Inner-validation rows are used three times (tail early stopping, head validation, budget and threshold choice). Test rows are used by none of these.
- The deployed W8A8 path is trunk cache (float16) plus quantized tail instead of one end-to-end pass (measured above).
- Stored-array gaps (plan C6, restrict): an expert missing a condition is left out of that condition's comparators. UrbanSound8K short AST lacks some full noise cells; astw4 and astfp32 lack corruptions.
- Disk: per-(fold, seed) training CSVs and FP32 embeddings were deleted after each head check, the fold-1 FP32 tails (`tail.pt`) after conversion to the compact form, and the trunk caches after scoring. Saved: tails (`ft/<task>/f<k>_s<s>/tail_q8.pt`, `train_rows.npy`, `train_log.json`), heads with scalers (`outputs_ft/`), score files (`scores/`).
- About 13:15 to 14:20 on 28 Oct another project's job ran on MPS outside the queue; TAU fold 5 was slower and its process stopped itself once (exit 3) and was requeued. No result changed.

## Layout

`inputs/` costs, alignment, cache checks. `ft/` tails. `outputs_ft/` heads. `outputs_p1`, `outputs_p1ft` P1 heads. `emb/<task>/` frozen clean embeddings, lengths, noise plan, condition sidecars. `scores/` logits in the stored layout. `tables/` results. `logs/`. `scripts/` copy of the code at the final commit. `run_manifest.json` commits, versions, checksums.
