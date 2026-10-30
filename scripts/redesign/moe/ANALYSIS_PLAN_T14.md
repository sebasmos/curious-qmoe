# Difficulty mixture of experts, T14: is the margin-routed two-expert quantized system more stable than single models?

Written and committed on 25 Oct 2026 (branch `moe/t14`, from `moe/t13`) before any test statistic of T14 was computed. What exists at this commit: this file only. The input check,
the loaders and the analysis script are committed after it; the input check reads no label. Machine: CPU, `nice -n 10`, 2 threads. Nothing is trained or extracted. Stored arrays
only, real data only; a quantity that is not stored is reported as not stored.

## 1. Question, background, family

The original paper said uncertainty-aware routing "cuts cross-fold variance by 68 to 97%". Two blind re-audits (`2026-10-23_1144_reaudit-arxiv-v4_A`, `_B`) found the claim
unsupported: leaky folds, one baseline run, Levene p 0.18, Quinn variance up 76%. Nothing has tested stability for the new design. T14 does, in three senses: sampling (variation
across folds and seeds), shift (how far F1 falls when the audio changes) and latency (how slow the slow clips are).

Family. T14 is counted with T1, T1b, T1c, T2b, T7, T9, T12, T13 (nine experiments). Holm runs over the 12 primary T14 tests (section 6). The README also shows each adjusted p
multiplied by 9 (Bonferroni over experiments, capped at 1) as an extra line; the verdict uses the Holm over the 12.

## 2. Systems and data

Beds: ESC-50 with 50 classes (`esc50`, seeds 0 to 4, 5 folds), ESC-50 with 5 groups (`esc5`, same), UrbanSound8K (`urban`, seeds 0 to 2, 10 folds). Primary tests use these three.
TAU (seeds 0 to 2, 5 folds) is a supplementary fourth bed, the only one with a device shift, outside the "two of three tasks" count.

The margin-routed pair is the stage 3c pre-registered pick P3 of each task (`analysis/<task>/picks.json`, chosen on clean inner validation before any test fold was scored):

| Task | Cheap expert | Strong expert | Rule, budget |
|---|---|---|---|
| esc50 | mn20 W8A8, 4-bit head | AST w4g128 at 512 frames (esc head) | avg, b05 |
| esc5 | mn20 W8A8, esc head | AST w6 at 1024 frames (esc head) | avg, b20 |
| urban | mn10 W8A8 (padded, stage 3b), esc head | AST FP32 at 1024 frames (esc head) | avg, b50 |
| tau (supplement) | mn20, 8-bit head (option B pick) | AST FP32 (esc head) | avg, b50 |

Routing: the cheap expert's softmax margin below a threshold set on the clean inner-validation clips of the same fold and seed so that the budget share of clean validation clips
escalates; escalated clips use the average of the two experts' softmax vectors, the others the cheap expert alone (the stage 3c code path; the stored `cascades.csv.gz` rows are the
reference). Systems compared, per task:

1. Pair: the margin-routed pair above.
2. Always-average: the same two experts, softmax average on every clip. Recomputed from the stored logits.
3. Cheap expert alone (as in the pair).
4. Strong expert alone (AST at the pair's bit-width and frames).
5. AST FP32 and AST 4-bit at the same frames, where stored (esc50: `fp32@512`, `w4g128@512`; esc5 and urban: `fp32`, `w4g128` at 1024 frames).
6. Best single model at equal cost: among the stored single systems of stage 3c (sets `stage3b` and `new`, no real-int8 sets) whose priced cost is at most the pair's clean
   inner-validation cost (`cost_iv` in `picks.json`) and that are scored on every cell of the task's shift set, the one with the highest clean inner-validation F1
   (`static_iv.csv`). Chosen without test data. Sensitivity comparator (secondary): the same rule with the cost limit set to the largest mean test cost of the pair over the shift
   set, since the pair spends more when it escalates more.

Singles are read from the stored `static.csv`. The pair is recomputed on every cell from the stored per-clip logits. Reproduction gate before any contrast: the recomputed pair F1 on
clean and field equals the stored `cascades.csv.gz` row for every seed and fold to 1e-9, and recomputed singles equal `static.csv`; otherwise stop and report.

Shift set per task: every cell where both pair experts and the compared singles have complete finite logits (the T13 availability rule, label-free): `clean`; noise (bg, pink,
white at 20, 10, 5, 0 dB); `field`; nine non-additive cells (reverberation RT60 0.3, 0.6, 1.2 s; clipping at 0.5, 0.25, 0.1 of peak; low-pass at 4, 2, 1 kHz). From
`2026-10-25_0145_moe-t13_code-eaaedc9/inputs/availability.csv` the expected sets are: esc50 all 23 cells (the best-single pool is then limited to systems scored on the non-additive
cells: W8A8 mn10 and mn20 with four heads); esc5 the 14 additive cells and field; urban clean, bg 10 dB, bg 0 dB and field (the AST at 1024 frames is stored for nothing else).
Sensitivity for esc50: the 14 additive cells with the full stage 3c pool. Device shift exists for TAU only. Supplement for urban: the final-round pair (exact-length mn10 W8A8, FP32
head, w4g128 at 400 frames, avg, b30; part B of `2026-10-24_1507_redesign-idea6-final_code-ed82400`) on its 13 cells; it is a substitute, not the stage 3c pick, and does not enter
the primary count.

## 3. Notion 1, sampling stability

Per task and condition in {clean, field}, per system: the matrix F1[seed, fold] (macro F1 in percent). Fold level (primary): mean over seeds per fold, SD across folds (ddof 1).
Seed level: mean over folds per seed, SD across seeds (3 or 5 values, descriptive). Interaction: residual r_sf = x_sf - x_s. - x_.f + x_..; interaction SD = sqrt(sum r^2 / ((S-1)(nf-1))).
Ratio of variances: pair over each other system on fold-level values, 95% percentile bootstrap over folds (paired, 10,000 resamples, numpy `default_rng(20261025)`). Seed-level
and interaction ratios use the same bootstrap over seeds and over folds. Test: Brown-Forsythe (scipy `levene`, `center='median'`) on the two fold-level vectors, two-sided.
Secondary: Pitman-Morgan test for paired variances. With five folds both tests have little power; the plan expects intervals spanning 1 and many "not determinable" verdicts.

Rule. Sampling stability holds for a task if, in both clean and field, the pair's fold-level variance over the best single's (equal cost) is below 0.7 and the bootstrap interval
excludes 1. The notion holds if it holds on at least two of three tasks. Mirror rule (ratio above 1/0.7 with an interval above 1): "less stable".

## 4. Notion 2, shift stability

Per task, system and fold f: F_{c,f} = mean over seeds of F1 in cell c of the shift set. Measures per fold: worst-case F1 (min over cells); SD of F1 across cells; average drop
(F1 clean minus the mean of the other cells, equal weight per cell); slope of F1 against SNR (OLS per dB over the 0, 5, 10, 20 dB cells of each noise kind, averaged over kinds).
Device shift: TAU supplement, clean (device A) against shift (other devices), per-device cells where stored.

Tests. Worst-case: d_f = pair minus comparator. Drop: d_f = comparator's drop minus the pair's drop. Fold-level paired two-sided t test (Nadeau-Bengio corrected variance, ratio
n_test/n_train, secondary), mean difference in F1 points, 95% t interval, folds with d_f at least 0.5.

Rule. For a task the notion holds if, against the best single at equal cost, both the worst-case and the drop differences are at least +0.5 points and the Holm-adjusted p is
below 0.05. The notion holds if it holds on at least two of three tasks (the request gives no count; this mirrors notion 1). "Less stable": a difference of at most -0.5 with Holm
p below 0.05 on two tasks. Urban's worst case rests on three shifted cells.

## 5. Notion 3, latency stability

From `2026-10-23_1919_redesign-idea6-latency-full_code-cbf6be6` (`tables/cascade_latency.csv`, `tables/steps.csv`, `raw/`): median, p95 and p95 over median per task for the pair, the
cheap expert alone and the AST alone, at 1, 2 and 4 threads, for every runtime combination of the run. The run measured step times; the pair's median and p95 are simulated over
200,000 clips from those passes at the stored escalation rates. Measured for esc50 (the P3 pick) and urban (stage 3b's P2 pick re-evaluated, mn10 to w8g64 at 400 frames, not the
P3 pick); esc5 not measured. I repeat the same simulation from the raw passes with a bootstrap over passes and over the seed-fold escalation rates for an interval; no new timing.
Rule. Holds if the pair's p95 over median is not larger than the AST alone's in the kernel configuration matching the pick. Since an escalated clip pays both paths, the plan
predicts a much larger ratio for the pair whenever fewer than half the clips escalate.

## 6. Primary tests and Holm

Twelve primary tests, Holm over all twelve: notion 1, Brown-Forsythe p for the pair against the best single (3 tasks x {clean, field} = 6); notion 2, fold-level t p for the
worst-case difference and for the drop difference against the best single (3 x 2 = 6). Every other comparison (other systems, sensitivities, supplements, Pitman-Morgan,
Nadeau-Bengio, per-device) is secondary, labelled, with its own Holm line. No tuning on test folds: every selection (pair, threshold, best single) uses clean inner-validation data.

## 7. Reporting and decision

Report every system, every notion, effect sizes with intervals, which hold and which do not. A stability claim may appear in the paper only for a notion that holds under its rule
and is not reversed by the sensitivities or supplements. Limits for the README: five folds (ten on UrbanSound8K), 3 to 5 seeds; seeds share folds, so seed variation is head
initialisation only; the pair's cost grows under shift while the comparator's is fixed; the urban pair's AST is stored for four cells; cascade latency is simulated from measured steps.

## 8. Amendment 1 (25 Oct 2026, owner extension; committed before any statistic or any label-bearing score was read)

The owner asked for the same stability measures for every idea and system tested, wherever per-seed, per-fold outputs exist, in one table (`stability_all_ideas.csv` and `.md`
in the results folder, a compact version at the top of the README). Sections 1 to 7 stand unchanged; the pair analysis stays the primary analysis and the 12 primary tests of
section 6 stay the only members of the primary Holm family. The extension is a descriptive family of its own (Holm line labelled "extension").

Systems and where they are read from (all under `~/Documents/qmoe-results/`, read only). Pick = the design that the stage itself fixed; where a stage fixed none, the README headline
arm is used and the row says so. Nothing is picked on test data.

| Id | System | Source and pick |
|---|---|---|
| P2o | Original MoE, protocol 2: `uniform_ctrl`, `pprior_a1.0`, shared-trunk pair, FP32-expert arms | `2026-10-20_1300_protocol-v2_code-5db7e75/outputs/*/moe/fold_k/metrics.json` (`f1_macro`), esc (5 groups) 5 seeds x 5 folds, urban 3 x 10, quinn 3 x 5, clean only; FP32-expert arms from `..._1801_...`. No pick: headline `pprior_a1.0` against `uniform_ctrl`, both reported |
| V4 | Original paper claim, one 5-fold run per arm | `2026-10-23_1146_reaudit-arxiv-v4_B/results/claimEJ_stats.json` key `E` (arrays of five fold values), baseline for the "68 to 97%" claim; reported with the old-protocol label |
| S3 | Protocol 3 singles (head precision: bits, bitnet, esc, qesc) | `2026-10-20_2127_protocol-v3-singles_code-bc4ff1c/outputs/esc_singles_s*/<head>/fold_k/metrics.json` |
| I1 | Idea 1 head cascades, stage 2, headline arm `c1to8 b05` | `2026-10-21_0002_..._idea1-stage2.../analysis/main/records.jsonl` (esc 5 groups, quinn, urban; clean) |
| I2 | Idea 2 learned deferral | seed 0 only, aggregates only for the headline: not determinable (one seed), listed with the reason |
| I3 | Idea 3 disagreement | logits per fold for one or two seeds, clean only, no per-fold F1 table, no pick: listed as not computed with the reason |
| I4 | Idea 4 repaired router: `uniform_ctrl`, `pprior_a1.0`, `A_dense` | `2026-10-20_2039_..._idea4-stage1.../analysis/diag_esc_s0.json` (esc, seed 0, 5 folds, clean); seed-level quantities not available |
| I5 | Idea 5 precision cascade of the AST, stage 2, `n3g64 -> w8g64`, FP32 head, margin, b05 (a headline arm of the README) | `2026-10-20_2359_..._idea5-stage2.../analysis/<task>/cascades.csv` (14 cells for esc50 and esc5, 4 for urban) |
| I5b | Idea 5b hull of random mixing of two fixed bit-widths | `2026-10-22_1042_..._idea5b-hull.../outputs/*/cascade_pairs.csv.gz`; no pre-registered pick (post hoc best points only): listed, computed only if a pick-free row exists, otherwise reason given |
| I6a | Idea 6 stage 1 (EfficientNet cascade), `primary_fp32.csv` rows | `2026-10-22_0953_..._idea6-stage1.../analysis/<task>/cascades.csv` (clean) |
| I6b | Idea 6 stage 2 H1 (8-bit EfficientNet to the short 4-bit AST) | `2026-10-22_1120_..._idea6-stage2.../analysis/<task>/cascades.csv.gz`, `picks.json["H1"]` |
| I6c | Idea 6 stage 3b P2 | `2026-10-22_1652_..._stage3b.../analysis/<task>/cascades.csv.gz`, `picks.json["P2"]` (clean, field) |
| I6d | Idea 6 stage 3c P3 = the margin-routed pair | primary analysis above |
| OB | Option B (TAU) cascade pick | `2026-10-23_2133_..._optionB.../analysis/tau/cascades.csv.gz` (supplementary bed) |
| OC | Option C: stage-3c-style two-tier pick H1 and the three-tier pick E1 | `2026-10-23_2133_..._optionC.../analysis/<task>/cascades*.csv.gz`, `picks.json` |
| I7 | Idea 7 device to cloud, primary row (`n3g64`, `esc` head, rule `pre`, field) | `2026-10-22_1105_..._idea7-stage1.../outputs/<task>/pairs.csv.gz` |
| AA | Always-average of the pair's two experts | primary analysis above |
| T1 | T1 and T1c specialist R10 and its uniform twin U | `2026-10-24_2013_moe-difficulty-pilot.../t1/t1_arm_metrics_per_seed_fold.csv`, `2026-10-25_0052_moe-t1c.../t1/t1c_arm_metrics_per_seed_fold.csv`. Only whole-clip accuracy is stored (`*_all`), not macro-F1: the row says "accuracy" |
| T2b | Gate arms: hard router (A1), soft mixture (A3f, A3h), stacking and the always-average reference (A4) | `2026-10-24_2224_moe-t2b.../urban/t2b_per_seed_fold.csv`, `2026-10-25_0052_moe-t1c.../t2b/t2b_per_seed_fold.csv` (`f1`) |
| T7 | Masked experts, primary arm `CE_energy25` | `2026-10-24_2313_moe-t7.../{urban,esc50}/arms_per_seed_fold.csv` (`f1`) |
| T9 | Any-precision tier selection | per-fold differences to the hull only, no per-seed F1: listed as not computed |
| T12 | Adapter expert | micro-pilot, fold 1 only: not determinable, listed |
| T13 | Uniform mixture, the pre-selected pair per task | `2026-10-25_0145_moe-t13.../cell_fold_acc.csv` (`kind=mean`) |
| Sg | Single fixed models: every system of stage 3c `static.csv` (sets `stage3b`, `new`), plus the final-round singles | table `stability_singles.csv`; the compact table lists the cheap and AST experts of the pairs and the best singles |
| T4, IBL | RL router control (T4) and information-based routing losses | planned and never built: "not run", no number |

Reference for the variance ratio and the shift differences of every system s: the best single model at equal cost on the same task and the same cells, defined as in section 2:
the single of the same source file (so that costs share one unit) with priced cost at most the system's mean clean cost and the highest clean inner-validation F1. Where a source
file stores no inner-validation F1 (idea 5, the protocol-2 and protocol-3 runs), the reference is the single with the highest mean clean test F1 among those of cost at most the
system's, which is selected on test data and therefore favours the reference; the row is labelled `ref_selection=test_oracle`. Protocol-2 MoE arms use the protocol-3 singles
of the same task and the cost limit of their `training_flops`/bit-ops fields is not comparable, so there the reference is the best protocol-3 single on clean test F1, labelled the same way.

Per row the table reports, per task: SD of macro F1 across folds and across seeds (clean, and field where scored), the seed-by-fold interaction SD, the worst-case F1 and the average
drop over the cells in which the system was scored (noise, field, corruptions, devices; the reference is measured on the same cells), p95 over median latency where measured
(esc50 and urban pairs only), the fold-level variance ratio to the reference with the section 3 bootstrap interval, the shift differences with fold-level t tests.

Verdict per system and task, then overall. Sampling label: "more stable" if in every scored condition of {clean, field} the ratio is below 0.7 with an interval below 1; "less stable" if
the ratio is above 1/0.7 with an interval above 1; "as stable" if the whole interval lies inside (0.7, 1/0.7); otherwise "not determinable at five folds". Shift label (cells >= 3):
"more stable" if worst-case and drop differences are both at least +0.5 with Holm p (over the system's tasks) below 0.05; "less stable" if both at most -0.5 with Holm p below 0.05;
otherwise "no difference shown". Overall: "more stable" if either notion is more stable on at least two tasks (all scored tasks when the system is scored on fewer than three) and
neither is less stable; "less stable" if either notion is less stable on at least two tasks (or all scored tasks) and neither is more stable; "as stable" if every scored sampling
label is "as stable" and no shift label is more or less; "mixed" if both directions occur; else "not determinable at five folds". A system with one seed, one fold, or aggregates only
is "not determinable" or "not stored" with the reason. Holm over all rows of the extension is shown in a separate column and does not change a verdict; the pair verdicts of
sections 3 to 5 govern the paper question.

Folds and seeds of different sources coincide for the same task (the same fold files), so the same reference single is used across sources where costs are in the same unit. Costs in
different units are never compared across sources.

## 9. Amendment 2 (25 Oct 2026, coordinator instructions after "execute all"; committed before the rerun of the primary analysis)

Disclosure. Between amendment 1 and this one, `t14_pair.py`, `t14_check_inputs.py`, `t14_primary.py` and `t14_latency.py` were run once, with the best single chosen on clean inner-validation F1
(section 2). The outputs are kept unchanged under `tables/plan_v1_cleaniv/` and `latency/` of the results folder and are reported next to the amended run. The amendment is therefore
made after those results were seen. The change only concerns which stored single model is the comparator; the selection uses inner-validation clips, never a test fold.

1. Family. T14 is counted in a family of 15 experiments (T1, T1b, T1c, T2b, T7, T9, T12, T13, T14, T15, T16, T17, T18, T3, T5). The extra line is each adjusted p multiplied by 15
   (capped at 1), replacing the 9 of section 1. The decision Holm stays over the 12 primary T14 tests.
2. Comparator rule (replaces the clean-validation rule of section 2). The best single at equal cost is chosen on the inner-validation F1 of the same condition family as the test:
   sampling, clean: mean clean inner-validation F1; sampling, field: mean inner-validation F1 on the field-mix rows; shift: mean inner-validation F1 over the non-clean cells of the
   task's shift set (one comparator per task, not one per cell, which would be a hull). Candidates: stored singles of stage 3c (sets `stage3b`, `new`) with priced cost at most
   `cost_iv` of the pair that are complete (test and inner-validation logits, every seed) on every cell of the shift set. Inner-validation F1 is recomputed from the stored
   inner-validation logits and labels (`t14_select.py`); `static_iv.csv` holds clean only. Candidates the loader cannot map to a score folder are listed and excluded. The pair uses the
   comparator of its own notion: clean and field sampling tests use the clean and field comparators, both shift tests use the shift comparator. The sensitivity comparator keeps the
   larger cost limit of section 2. The v1 (clean-iv) results are reported as a sensitivity.
3. Protocols. Every row of the extension carries a `protocol` column: protocol 2 (random inner split, reshuffled folds, leaky: original MoE arms and the old uniform-routing control, ideas
   1 stage 1, 2, 3 and 4) or protocol 3 (inner split grouped by recording: protocol-3 singles, idea 1 stage 2, ideas 5, 5b, 6, 7, options B and C, T-series, T13). The references of
   amendment 1 for protocol-2 arms ("protocol-3 singles") are withdrawn. A variance ratio is only formed between systems of the same protocol and the same source. Protocol-2 MoE arms
   report the ratio of `pprior_a1.0` to `uniform_ctrl` (the original claim's own comparison, labelled leaky) and "not determinable against the best single" because no protocol-2 single
   with a comparable cost field is read. No spread of a protocol-2 system is compared with any protocol-3 number.
4. Inner-validation F1 is not stored for idea 5, the protocol-2 runs and several older stages. There the reference is chosen by clean test F1 among singles of cost at most the system's
   (`ref_selection=test_oracle`, favourable to the reference), or the row is marked not determinable if no priced singles of the same source exist.
5. Sources that hold only logits for one or two seeds (ideas 1 and 3 stage 1, idea 2) are listed as "not computed, logits only"; sources with aggregates only (T9, T12) are "not determinable".
   T1 and T1c store accuracy, not F1; they are reported as accuracy and flagged.
