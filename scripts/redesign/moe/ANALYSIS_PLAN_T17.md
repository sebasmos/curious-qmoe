# Difficulty mixture of experts, T17: diversity by training, distillation, and an all-clips adapter expert

Written and committed on 25 Oct 2026 (branch `moe/t17`, from `moe/t13`) before any head of T17 was trained and before any test label was read. The job script, the
inner-validation selection script and the scoring script are each committed after this file and before they are run (a script that reads a test label is committed first).
Machine: CPU, `nice -n 10`, two single-thread lanes, for parts (a) and (b). Part (c) needs the GPU lock and gets its own amendment (section 8) before any (c) test clip is scored.

## 1. Question and family

(a) Can the two cheap CNN experts be trained to be complementary (negative correlation learning, error decorrelation, boosting), so that the pair beats independently trained
heads, a same-architecture two-seed ensemble and the time-sharing hull at equal priced cost under shift?
(b) Can the cheap tier be made stronger by distilling the AST's soft labels into it, so that the distilled cheap head gains on the shifted conditions and its system beats the hull?
(c) Can an AST adapter expert trained on all clips (difficulty weights, SpecAugment, mixup, distillation) beat its uniform twin on the clips the cheap model flags? This is the repair
named in the T12 README (the T12 specialist was trained on the flagged subset only and lost 5.1 points to its twin).

Family. The owner and the task brief fix the family at **15** (T1, T1b, T1c, T2b, T7, T9, T12, T13, T14, T15, T16, T17, T18, T3, T5). The Table 4 text of `index.md` (step 3 of the
order of work) still says 12 for T17; 15 is used here. Holm: inside T17 over its **6 primary tests** (3 tasks x parts (a) and (b), section 6). The README also prints each Holm p
multiplied by 15 (Bonferroni over experiments, capped at 1) as the family-level line; a part is called PASS only if it clears the pass rule and the within-T17 Holm below 0.05,
and the README states separately whether it survives the family-15 line. Part (c) is a single one-fold micro-pilot with its own bar and no p value.

## 2. Data: stored embeddings only for (a) and (b)

Nothing is extracted. Clean-trained heads on stored embeddings; every embedding is read from a stored folder (read only).

| Name | Embedding | Dim | Stored in |
|---|---|---|---|
| E1 = MN10 | mn10 W8A8, clean and all nine field cells; UrbanSound8K is the padded mn10 of stage 3b, as in T1c | 960 | stage 3b `emb_cnn` (ESC-50, UrbanSound8K); option B (TAU, clean cell = every row) |
| MN20 | mn20 W8A8, same cells | 1920 | stage 3c `emb_cnn`; option B (TAU) |
| AST (teacher) | AST FP32 head logits, 5-fold cross-fitted on the training clips (`E2a_oof`) | 768 | T1c `raw_oof/s<seed>_f<fold>.npz` |
| AST (member of the system in (b)) | AST w4g128 at 1024 frames, T1c head `UA_E2b` logits on every test cell | 768 | T1c `raw_strength/s<seed>_f<fold>.npz` |

Beds, folds, seeds, inner splits, flag masks: exactly those of T1c (`fit_p`, `iv_p`, `S2`, `tau1_oof`, `z1_oof` from the T1c strength files; the assertion `tr`, `te` equal T1c's is run in every job).
UrbanSound8K 10 folds x seeds 0 to 2, ESC-50 (50 classes) 5 folds x seeds 0 to 2, TAU 5 folds x seeds 0 to 2 (training on device A rows). That is 60 seed-folds.

Conditions (T1c definitions, `t1c_common.groups`): U-field, U-0dB (bg at 0 dB; only bg 0 dB is stored for every UrbanSound8K AST clip), E-field, E-0dB (bg, pink, white at 0 dB pooled),
T-shift (non-A devices of the test fold); descriptive controls: clean for each bed. Tasks used for the pass rules pool the shifted conditions of a bed: **urban** = mean of U-field and U-0dB,
**esc50** = mean of E-field and E-0dB, **tau** = T-shift. The unit of every test is the outer fold, seeds averaged inside the fold.

Inner-validation families (no test clip; used to choose lambda, temperature, alpha and the best-single comparator "on the same condition family"):
- field family (UrbanSound8K, ESC-50): the inner-validation clips of the fold, each put into the condition cell given by `field_assignment(N, FIELD_SEEDS["val"] = 5002)` (the test assignment uses 5001);
- 0 dB family: UrbanSound8K bg 0 dB, ESC-50 bg, pink and white 0 dB, all inner-validation clips of the fold;
- TAU shift family: the stored TAU embeddings cover every device of every fold, so the non-A rows of the training folds (never trained on, none in the test fold) serve as the inner-validation shifted set;
- clean family: the clean inner-validation clips.
Selection is done per fold, summing correct counts over the three seeds of that fold, so no label of the fold's own test clips and no label of another fold's test clips enters a choice. Ties go to the first
value in the grid order.

## 3. Training: one loop, `t17_heads.py`

Heads are the repo's FP32 `esc` head (`BitNetExpert(num_bits=32)`, hidden 640/320, the protocol 3 recipe of `config/urban8k.yaml`: AdamW, class-weighted loss, label smoothing, plateau schedule,
early stopping, best checkpoint) with the selection metric = inner-validation accuracy (of the mixture, for a pair). **Deviation from the Table 4 wording "Q4 heads"**: a 4-bit hidden-layer head exists in the
repo (`cnn:mn10:w8a8:4` is priced, 2.9 percent below the `esc` head on mn10), but quantization-aware training is 2.7 times slower per epoch (probe: 28 s against 10 s, 17 epochs against 26) which would put the run
near 9 hours; the stored singles that make the hull are FP32 `esc` heads, as in T13, so FP32 heads are used for every arm and every control, and costs are the `esc` costs of T13. The head is under 3 percent of the pair's cost.

A pair is two heads, one on MN10 and one on MN20, trained in lock step on shared minibatches; the pair's output is the mean of the two softmax outputs. Reproduction check (`t17_check.py`, logged before the run):
a one-member group in mode plain equals `heads.fit_head(metric="acc")` to 0.0.

Arms (per seed-fold; `p`, plain single heads trained alone with weight 1; pairs trained together):
- `p10`: MN10 head, seed s. `p10b`: MN10 head, seed s + 100 (same split). `p20`: MN20 head, seed s.
- `indep`: pair with lambda = 0 (equals two independent fits trained on the same minibatches; stopping and checkpoint on the pair's accuracy). **Control 1.**
- `ncl<lam>`, lam in {0.1, 0.3, 0.5}: negative correlation learning, loss = CE_1 + CE_2 minus (lam / 2) times the batch mean of the squared Euclidean distance between the two softmax vectors (the K = 2 form of the penalty; gradient through both members).
  With lam = 0 it is `indep`. The primary diversity arm is `ncl` with lam chosen per condition and fold on the inner-validation family among {0 (= indep), 0.1, 0.3, 0.5}, by pair accuracy on the family.
- `dec<lam>`, lam in {0.1, 0.3, 0.5}: error decorrelation, loss = CE_1 + CE_2 plus lam times the batch cosine similarity of the two residual matrices (p_i minus the one-hot label, flattened over the batch). Secondary arm, lam chosen the same way.
- `b20`: MN20 head trained with per-clip weight 1 + 2 e, where e = 1 if the cross-fitted MN10 head (stored `z1_oof`, 5-fold out-of-fold logits on the training clips) misses the clip, else 0. Boosting arm = (`p10`, `b20`) mean. Secondary.
- Two-seed same-architecture ensemble (**Control 2**) = (`p10`, `p10b`) mean. **Deviation**: only one MN10 backbone exists per bed, so this is two head seeds on one MN10 embedding; its priced cost is one MN10 forward plus two small heads, below the pair's cost
  (MN10 + MN20). If the pair cannot beat this cheaper control, cross-architecture diversity adds nothing over seed averaging; if it can, part of the margin is the extra cost of MN20.
- Distillation, single MN10 head with the loss of section 1: per-clip alpha_i * CE(label, smoothed) + (1 - alpha_i) * T^2 * KL(teacher_T || student_T), teacher = `E2a_oof` (the FP32 AST head, cross-fitted, so it never labels a clip it was fitted on).
  `kdA_T<T>_a<alpha>`: all clips, T in {1, 2, 4}, alpha in {0.25, 0.5, 0.75} (9 heads); `kdF_T<T>_a<alpha>`: distillation only on the clips the cross-fitted cheap model flags (`S2`, margin below `tau1_oof`), alpha_i = 1 elsewhere (9 heads).
  (T, alpha) chosen per condition and fold on the inner-validation family by cheap-head accuracy, separately for `kdA` and `kdF`. **Control** = `p10`. Primary distilled arm = `kdA`.

Cost per seed-fold: 3 plain + 7 pairs (14 fits) + 1 + 18 = 36 head fits; probe 10 s per fit on two threads, about 15 s on one. 60 seed-folds, about 9 hours of single-thread work, 4 to 5 hours on two lanes.

## 4. Systems, comparators, cost

Pair cost = priced bit-ops of MN10 plus MN20 (stage 3c `costs.json`: UrbanSound8K 1.02e11 with the padded mn10, ESC-50 1.02e11, TAU option B). System of (b) = uniform mean of the softmax of [distilled or plain MN10 head, `UA_E2b` AST w4g128 head]; cost = mn10 + w4g128 at 1024 frames
(about 7.0e12). Reading chosen for "its system beats the hull by 1.0": the **two-expert system** (cheap head plus AST), the structure that already beat the hull in T13; the distilled cheap head on its own is also compared with the plain head (gain clause).

Hull: Stage 2's `hull_pair` on the stored single experts of T13 (`t13_common`: mn10, mn20, astfp32, astw4, astshort where stored for the cell) at the system's cost, on the mean accuracy over seeds and folds of the cell, built per cell and averaged over the condition's cells; the fold values are the
cell's fold accuracies (the T13 construction, `cell_fold_acc.csv` of the T13 run, which is derived from stored logits and rebuilt by `t13_score.py`; the hull is chosen on test accuracy, favourable to the hull). For the pair of (a) the hull cost is the pair cost; the UrbanSound8K hull singles use the unpadded mn10 of T13 (the stronger stored model), a stricter comparator than the padded embedding used for the new heads.

Best single at equal cost: among the T13 single experts with priced cost at most the system's cost, the one with the highest accuracy on the inner-validation family of the condition (T13 inner-validation logits, `field` with seed 5002 or the 0 dB cells, pooled over the three seeds, per fold). TAU: the stored TAU inner-validation logits cover device A only,
so the choice on TAU uses the clean device-A inner validation (stated limitation). This is a reported comparator, not a pass clause.

Metric: accuracy in percent over all rows of the condition, seeds averaged inside a fold, macro-F1 reported beside it. Nadeau-Bengio corrected resampled t-test over folds (t = mean(d) / sqrt((1/K + n_test/n_train) s_d^2), K = 10, 5, 5; n_test/n_train per fold as in T13), two-sided; 95 percent interval from the same standard error.
"Majority of folds" = at least 6 of 10 or 3 of 5.

## 5. Primary contrasts and pass rules (pre-registered)

Part (a), per task (pooled over the task's shifted conditions): C1 = `ncl`(lambda chosen) pair minus `indep` pair; C2 = `ncl` pair minus the two-seed ensemble; C3 = `ncl` pair minus the hull at the pair cost. A task passes if C1 >= +0.5, C2 >= +0.6 and C3 >= +1.0, each also in a majority of folds, and the task p (maximum of the three NB p) clears Holm.
Part (a) passes if at least two of three tasks pass. Secondary (raw p, Holm within the secondary arms of a task, not in the family): `dec` and boosting against `indep`; all three arms against the two-seed ensemble and the hull; the best-single comparator.
Part (b), per task: G = chosen `kdA` head minus `p10` on all rows of the shifted conditions (cheap head alone), S = (`kdA` head + AST w4g128) system minus hull at the system's cost. A task passes if G >= +1.5 in a majority of folds (and in the mean), S >= +1.0 in the mean and a majority of folds (>= +1.0), and the larger NB p of G and S clears Holm.
Part (b) passes if at least two of three tasks pass. Secondary: `kdF`; the distilled system minus the plain system (`p10` + the same AST head); distilled system minus best single.
Primary tests: 3 tasks x 2 parts = 6, Holm inside T17 over them (adjusted p as in T1c: sorted raw p, running maximum of (6 - i) p). Descriptive per condition and the clean controls carry raw p, outside the Holm.

## 6. Stop rule

If no arm of (a) or (b) (`ncl`, `dec`, boosting, `kdA`, `kdF`) gains +0.3 points over its control (`indep` for the pair arms, `p10` for the distilled arms) in the pooled shifted contrast of any task, parts (a) and (b) are called STOP.
Part (c) has its own bar (section 8) and runs once, as the owner's "execute all" and the task brief ask, whatever (a) and (b) say.
Verdict labels per part: PASS (rule of section 5), STOP (rule above), otherwise NO PASS; the per-task counts and every arm are printed.

## 7. Reporting

README gives, for every arm and control, the pooled and per-condition accuracies, the contrasts with 95 percent intervals, folds at the thresholds, raw and Holm p, the x15 line, costs, the chosen lambda/T/alpha frequencies, and the epochs run. Heads trained here are compared with the stored heads of T13 (mn10 and mn20 accuracy
on the same cells) as a sanity line. Everything is relative to the same stored embeddings; no claim is made about backbones trained with other seeds.

## 8. Part (c): all-clips adapter expert (amendment to be committed before any (c) clip is scored)

Design fixed now: UrbanSound8K fold 1, seeds 0 to 2, AST FP32 at 400 frames, LoRA rank 16 on the last two blocks and a new head, as in the T12 micro-pilot (its `t12_common.py`, `t12_pilot.py` code and the stored fold-1 inner split, flags and the T12 audio pipeline are reused read-only).
Both arms train on **all** training clips of fold 1 with the same SpecAugment (time and frequency masks on the fbank), mixup (beta 0.4) and distillation (teacher = the stored `E2a_oof` FP32 AST head logits, T = 2, alpha 0.5, mixed with the same mixup weights). Specialist: per-clip weight 1.0 on clips the cross-fitted cheap model misses (`S2`), 0.25 on the others; twin: uniform weight.
Unlike T12, the 10 frozen blocks are not cached (the cache blocks SpecAugment and mixup), so a step is about 1.0 s at batch 32. Bar (from the task brief): specialist minus twin on the flagged shifted test clips (mean of U-0dB and U-field, flagged by the retrained MN10 as in T12) >= +1.0 point on this fold, else no full run.
Step count, learning rate and the evaluation schedule are fixed in the amendment from inner-validation probes only. Baselines printed beside it: frozen head, T12 twin and specialist numbers.

## 9. Known limits stated now

1. Heads on frozen embeddings; no backbone retraining in (a) and (b). 2. One MN10 backbone, so the two-seed control cannot vary the backbone. 3. FP32 heads instead of Q4 (section 3). 4. U-0dB is bg only, as in T1c. 5. The hull is built on test accuracy (favourable to the hull); the UrbanSound8K hull uses the unpadded mn10.
6. ESC-50 inner validation holds about 160 clips per fold (480 in the 0 dB family), so the choices of lambda, T and alpha are noisy there. 7. TAU shift inner validation uses non-A devices of the training folds; the best-single choice on TAU falls back to device A. 8. The AST member of the system in (b) is the T1c head `UA_E2b`
(selected on the flagged inner-validation clips), not a head retrained here.

## 10. Amendment 1 to part (c), committed after two inner-validation probes and before any (c) test clip was scored

Probes (twin, seed 0, 600 steps, flagged inner-validation accuracy at steps 100 to 600): lr_body 1e-4 gave 0.665, 0.598, 0.622, 0.604, 0.601, 0.619; lr_body 3e-5 gave 0.643, 0.591, 0.616, 0.591, 0.595, 0.607.
Both peak at the first evaluation and then fall (the overfitting seen in T12 survives SpecAugment, mixup and distillation); 1.1 s per step. Fixed for both arms and all seeds: lr_body 1e-4, lr_head 1e-3, 300 steps, batch 32, evaluation every 50 steps, best checkpoint on the flagged inner-validation clips,
SpecAugment (one frequency mask up to 24 of 128 bins, one time mask up to 60 of 400 frames), mixup with probability 0.5 and Beta(0.4, 0.4), distillation T = 2, alpha 0.5, specialist weights 1.0 / 0.25. Seeds 0, 1, 2; test tokens from the T12 caches (the first ten blocks are frozen, so cached block-10 tokens are the exact input of the tail).
Bar as in section 8: mean over U-0dB and U-field of specialist minus twin on the E1-flagged test clips at least +1.0 point (scored with `t17c_score.py`, T12 bootstrap intervals); otherwise no full run.
