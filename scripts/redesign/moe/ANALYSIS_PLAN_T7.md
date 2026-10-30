# Difficulty mixture of experts, T7: a masked-view AST as the medium expert

Written and committed on 24 Oct 2026, branch `moe/t7` of curious-qmoe (from `moe/t2b`, clean tree), before any masked embedding is extracted, before any masked head is trained and before any test fold is scored by this experiment.
Source plan: `~/Documents/qmoe-results/2026-10-24_2001_difficulty-moe-plan/PLAN.md` (sha256 16b982d078f0df3974a3497b4c50d4b9192affa31806f23a5b4a913bfc215791), design D3 and experiments T7 and T8.
Earlier results this plan builds on: the pilot (T1 stopped), T1b (strength gate stopped), T2b (no gate beats always-average). Their finding: the cheap mn10 and the AST heads err on different clips (oracle 76.9 against 63.4 on the flagged clips),
and a learned gate over frozen-embedding experts recovers none of it beyond averaging. T7 asks whether an expert that sees a different part of the input is cheaper and errs more differently than the full AST.
Scripts: `scripts/redesign/moe/t7_*.py`. Extraction on the Apple GPU (MPS), at most two processes, fixed input lengths, swap watched; heads and analysis on the CPU at `nice -n 10`.

## 1. Question and status

Does an AST that reads 25 or 50 percent of the spectrogram patches, 4-bit group-wise as in the final results, (a) stay close to the full 4-bit AST and sit above the line between the small CNN and the full AST on the accuracy-cost plane, and
(b) err on different clips than the cheap model does, more so than the full AST does? The masked expert is the single new ingredient. Nothing else about the experts changes.
This is a new experiment, not an amendment of T1b or T2b. It adds no test to their families. Its own tests are the rules in section 8, applied to point estimates exactly as written.

## 2. Experts and references (UrbanSound8K, 8,732 clips, 10 official folds, seeds 0, 1, 2)

| Name | What | Source |
|---|---|---|
| E1 | mn10 W8A8, 960-d embedding, head of the pilot | stored embedding `emb_cnn/urban8k/mn10_w8a8__clean.npy` (stage 3b); logits, inner split and flagged set from the pilot's `t1/raw` |
| F | AST w4g128, 400 input frames (470 tokens), full view, head of T1b (`E2c`) | stored embedding `emb_ast_short/urban8k/w4g128__clean.npy` (stage 2); test logits from T1b `urban/raw_strength` (`E2c_A_U_te`), out-of-fold training logits from T2b `urban/raw` (`E2c_oof`) |
| M(sel, rho, head) | AST w4g128, 400 frames, kept patches only, head trained on masked embeddings | new, sections 3 to 5 |

The flagged set is the pilot's: a test clip is flagged when E1's probability margin is below tau_1, the threshold that escalates 30 percent of the inner-validation clips (clean). It is the same set as in T1, T1b and T2b.
The reference "full 4-bit AST" in every rule is F. The cheap model in every rule is E1.

## 3. The masking design

Grid. At 400 frames the AST has 12 frequency by 39 time patches of 16 x 16 (stride 10), 468 patches, plus the CLS and the distillation token, 470 tokens. The AST's pooled output is the mean of those two tokens.
A masked model keeps k patches: k = 117 for rho_keep = 0.25 (119 tokens) and k = 234 for rho_keep = 0.5 (236 tokens). Every clip keeps exactly k patches, so the input length is fixed across a batch.
Both special tokens are always kept. A kept patch keeps the position embedding it has in the 400-frame model (the table row of its grid index), so a masked model meets the same position vectors as the full one.
Nothing is retrained: the same quantized weights, `QuantAST` with `parse_config("w4g128")` (4-bit weights, group 128, 8-bit activations, 16-bit attention probabilities), run on the shorter token sequence.

Valid and padded patches. A clip of n samples has n_v = min(400, 1 + (n - 400) // 160) real fbank frames; the rest is padding. Patch (f, t) covers mel bins 10f to 10f + 15 and frames 10t to 10t + 15 and is valid when 10t < n_v.
Padding is not scored as energy (the extractor pads before normalization, so padded frames carry a middling value that a naive mean would rank as loud).

Selectors (all cheap rules, none uses an AST pass):
- `energy` (primary): score of a valid patch = mean of the AST input values (the normalized log-mel) over its 16 bins and its valid frames. Keep the k highest valid patches; ties go to the lower grid index; if fewer than k patches are valid, fill with padded patches by grid index.
  The mean of the log-mel is the patch's mean log-energy; the normalization is affine and does not change the ranking.
- `random` (control): k valid patches drawn uniformly without replacement, generator seeded by the crc32 of the clip name and rho; same fill rule. It is drawn among valid patches so that it does not waste slots on padding, which makes it the harder control.
- `crop` (control): contiguous time crop. Order the valid patches time-major (time column, then frequency); keep the k consecutive ones centred on the middle of that sequence (start = clamp((V - k) // 2, 0, V - k), V the number of valid patches); same fill rule.
- The cheap CNN's own time-frequency saliency is not run in T7. It needs a gradient pass through mn10, and its cost and the calibration of its map are not part of this plan. It stays an option for a follow-up if `energy` shows signal.

The patch embedding is computed only on kept patches (unfold, gather, then the convolution weights as a matrix product), which is numerically the convolution and is what is priced. The tokens then pass through the 12 quantized layers of `QuantAST` unchanged.

Extraction check C1 (before the masked extraction of the full set): the same code with all 468 patches kept must reproduce the stored full-view 400-frame embedding on 64 clips spread evenly over the CSV (rows 0, 136, 272, ...).
Tolerance fixed now: median per-row relative L2 error at most 0.02 and maximum at most 0.10. Kept-index checks on the same clips: exactly k unique indices, valid patches first, and `energy` index sets differ from `random` ones on more than 90 percent of clips.
If C1 fails, the cause is found and fixed in code and the fix is committed before the full extraction; no threshold above changes.

Conditions extracted for UrbanSound8K: `clean` for all 8,732 rows, and the field-mix cells (bg, pink, white at 10, 5, 0 dB, `audio_shift.FIELD_CELLS`) for the rows the test assignment (seed 5001, `audio_shift.field_assignment`) gives them,
with the noise generated by the same `NoiseMaker` as stage 2 (seeded by clip name; background clips from ESC-50), so the corrupted audio equals the audio of the stored full-view and mn10 embeddings.
Masks: 6 per condition (3 selectors x rho 0.25, 0.5). Sidecars record selector, rho, k, timing and the priced cost.

## 4. Heads

Architecture and hyper-parameters are the repo's: `BitNetExpert(num_bits=32)` with hidden sizes [640, 320], the protocol 3 recipe of `config/urban8k.yaml` (`scripts/redesign/moe/heads.py`, `fit_head`). Every head sees the AST embedding only (768-d).
The primary training and selection recipe is the one that produced F in T1b: fit part of the pilot's inner split (`fit_p`), checkpoint, schedule and early stopping on the accuracy of the inner-validation clips that E1's out-of-fold margin flags (`sel_p = iv_p[S2[iv_p]]`), uniform sample weights.
This makes F's test logits the stored T1b ones and lets the masked heads slot into T2b unchanged.

- `CE` head (primary): trained on masked embeddings of the training clips, each with its own mask (the mask is a function of the clip and is not resampled), plain label-smoothed cross-entropy of the recipe.
- `KD` head (secondary variant, loss L_mask of PLAN.md D3 without the random-mask average): recipe cross-entropy plus alpha x KL(p_F || p_M), p_F the softmax of the T2b out-of-fold logits of F on the training clips (a head that never saw the clip's recording group; detached; temperature 1), p_M the masked head's softmax.
  alpha in {0.5, 1, 2}, chosen per (seed, fold) by the best inner-validation selection score of the recipe, ties to the smaller alpha. Random-mask augmentation (several masks per training clip) is not used; one view per clip.
- Library-recipe sensitivity `LIB`: for F and the six CE masked heads, checkpoint and early stopping on macro F1 of the whole inner-validation part (as `config/urban8k.yaml` and the pilot's E1), uniform weights. Reported beside the primary results; it does not enter a rule.
- E1 and F on the field mix: the pilot's E1 head and T1b's F head are retrained with the same code, seed and inner split so that they can be applied to field embeddings. Reproduction checks before any field number is read: R1 retrained F test logits equal T1b's `E2c_A_U_te`, R2 retrained E1 logits equal the pilot's `z1_te`
  (maximum absolute difference at most 1e-3 per (seed, fold) on the clean test rows; these read logits only, not labels). If a check fails, the retrained heads are used everywhere and the difference is reported.
- Heads are fitted on clean training embeddings only. The field mix tests them on corrupted test rows, as in the earlier rounds.

Separation of scripts as in T2b: `t7_jobs.py` fits heads and writes test logits (reads no test label); `t7_score.py` is the only script that reads test labels and is run once, after the jobs finish and this file and the scripts are committed.

## 5. Cost

Priced bit-ops per clip as in the final results (stage 3c): backbone bit-ops from the token and patch counts with the repo's formula `ast_bitops_tok("w4g128", 1024, tokens, patches)` (linear MACs x 4 x 8, attention Q K^T at 8 x 8 and P V at 16 x 8, patch embedding 32 x 32,
plus one 32 x 32 rescale per token, output row and weight group), plus the Kaldi fbank front end (3.1122e9, the stored mean), plus the 768-d `esc` head (7.1631e8). Check R3: the formula for 470 tokens and 468 patches plus fbank plus head reproduces the stored `ast:w4g128@400+esc` = 2.085697e12 within 1e-6.
Masked models use tokens = k + 2 and patches = k (the convolution runs on kept patches only). The `energy` selector adds 1.3e8 bit-ops (one 32-bit add per input element per covering patch, 400 x 128 x 2.56 x 1024); `random` and `crop` add nothing; comparisons for top-k are not priced.
E1's cost is the stored `cnn:mn10:w8a8:esc` = 2.836354e10 (front end, backbone and head priced). Sensitivity: the same points without the fbank and selector terms (backbone and head only).
Measured CPU latency (`t7_latency.py`, after scoring): batch size 1, one UrbanSound8K clip, 2 torch threads, 10 warm-up and 50 timed passes, median and interquartile range, for the fbank extractor, the selector, and the AST encoder in FP32 at 470, 236 and 119 tokens including the gathered patch embedding.
No low-bit kernels exist on this machine, so a quantized AST runs at FP32 time; the latency checks that fewer tokens means less time here and says nothing about 4-bit speed.

## 6. Evaluation

Unit of every statistic: the outer fold (seeds averaged inside a fold). Per fold and arm: accuracy and macro F1 on all test clips (clean), accuracy on the flagged clips, and the same three numbers on the field mix (E1 on field rows is flagged with the clean tau_1).
Complementarity on the flagged clips, per fold: oracle(E1, X), the share of flagged clips on which E1 or X is correct, for X in {F, every masked arm}; phi coefficient between the error indicators of E1 and X on the flagged clips and on all clips; the share of E1's flagged errors that X corrects.
Oracle and phi are computed per seed and averaged over seeds inside the fold.
Contrasts are differences of fold-level values, reported with mean, Nadeau-Bengio 95 percent interval (K = 10, n_test / n_train as the mean ratio over folds) and the number of folds in each direction. The rules in section 8 use the means.
Reported for every arm, selector and rho: accuracy, F1, flagged accuracy, cost, ratio of cost to F and to E1, oracle with E1, phi with E1.

## 7. Position on the accuracy-cost plane

Points: E1 (c_E1, a_E1) and F (c_F, a_F) with priced costs (section 5) and fold-mean accuracy; masked point (c_M, a_M). The line joins E1 and F in (bit-ops, accuracy) coordinates, which is the repo's random time-sharing line (the hull joins statics linearly in cost).
Per fold, gap_M = a_M - [a_E1 + lambda (a_F - a_E1)] with lambda = (c_M - c_E1) / (c_F - c_E1). The point is above the line when the fold-mean gap is positive. The same test in log10 cost and in F1 is reported as descriptive; the repo's convention (linear cost, accuracy) decides.
The sensitivity prices without front end and selector. If the two prices disagree on the sign, the verdict is read from the primary price and the disagreement is stated.
Known property of the endpoints, from the stored results: F is not more accurate than E1 on UrbanSound8K clean (stored static table: AST w4g128 at 400 frames 86.1 percent accuracy against 86.3 to 86.5 for E1), so the line is flat or slightly falling and "above the line" is close to "no worse than E1".

## 8. Pre-registered rules, applied exactly (masked expert = `energy`, rho 0.25, `CE` head, primary recipe, UrbanSound8K clean, fold means)

(a1) Close to F: a_F - a_M at most 2.0 points.
(a2) Above the line: fold-mean gap_M greater than 0 (section 7).
(b1) Complementary by oracle: mean over folds of [oracle(E1, M) - oracle(E1, F)] on the flagged clips at least +1.0 point.
(b2) Complementary by errors: mean phi(E1, M) on the flagged clips below mean phi(E1, F), and below it in at least 7 of 10 folds.
PROOF: (a1) and (a2) and ((b1) or (b2)).
STOP: a_F - a_M greater than 4.0 points, or (a2) fails (the point is on or below the line).
Neither: the masked expert is reported without a verdict (INCONCLUSIVE) and the T8 pilot does not run.
Secondary reporting, outside the rules: rho 0.5, the controls `random` and `crop`, the `KD` head, the field mix, LIB, ESC-50. If the primary arm stops and a secondary arm would meet PROOF, that is written as an observation and is not a proof; it would need its own pre-registered run.
Reading of the controls: if `random` matches `energy` on accuracy and on the oracle gain within fold-level noise, the selector adds nothing and the report says that any effect comes from the reduced input and not from the rule.

## 9. Conditional follow-up: T8 pilot (only if PROOF)

Add the primary masked expert as a fifth expert EM to the T2b experts (E1, E2a, E2b, E2c, EM) and repeat the T2b comparison with the same arms (A1 margin E1 to E2a, A2h, A2f, A3h, A3f, A4 always-average, A5 stacking), the same hyper-parameter grids, the same budget rule, the same Holm family,
the same pass rules and the same stop rule (best gate arm beats always-average of the same expert set by less than +0.3 points: STOP), and the baselines always-average of all five and the repo's hull (random time-sharing between statics, stage 3c `front_clean.csv`).
Out-of-fold logits of EM on training clips use the `t2b_oof.py` recipe. Labelled "T8 pilot"; a T8 plan addendum and the changed scripts are committed before any T8 test fold is scored. Not run if the verdict is STOP or INCONCLUSIVE.

## 10. Secondary datasets

- ESC-50 clean (50-class `label`, 5 official folds, 512 frames, 600 patches, k = 150 and 300, stored full-view `emb_ast_short/esc50/w4g128__clean.npy`, mn10 W8A8 ESC-50 embedding): accuracy, cost, line, oracle and phi, library recipe, seeds 0 to 2, same selectors and rho. Secondary, no verdict. Any definition it needs that is not above is written as an amendment and committed before its test folds are scored.
- TAU 2020 Mobile: not run in T7 (no short-AST TAU embeddings exist and the masked extraction of its audio is a separate job).

## 11. Not decided on test folds

Selector, rho grid, k, head recipe, alpha grid, tolerance of C1, thresholds of section 8 and the price are fixed here. No test fold chooses anything. The only data-dependent choices (alpha, checkpoint) use the inner validation of the training folds.

## 12. Reporting

`~/Documents/qmoe-results/2026-10-24_<HHMM>_moe-t7_code-<commit>/` with `run_manifest.json`, `README.md`, `scripts/`, `logs/`, `inputs/`, embedding sidecars, raw logits, and every arm and ratio in CSV. The README states verdict, every number of section 6, cost and latency, caveats and next step.
Nothing under Overleaf, the AnpassenNN docs or any paper folder is touched.

## Amendment 1 (25 Oct 2026 before the full extraction, after the first run of check C1)

The first run of C1 (numbers in `logs/check_c1_first_run.txt` of the results folder, copied from the console; the log file was overwritten by the rerun) reproduced the stored full-view embedding exactly (relative L2 error 0.0 on the 64 rows, also against `QuantAST.forward`), and all kept-index checks passed, except the check that `energy` index sets differ from `random` ones on more than 90 percent of clips:
0.906 at rho 0.25 and 0.859 at rho 0.5. The cause is in the clips, not in the code: a clip with k or fewer valid patches (short clips; 6 of the 64 rows at k = 117, 9 at k = 234) has every valid patch kept by every selector, so the selectors agree on it by construction.
The check is restated for clips with more than k valid patches (differ on more than 90 percent of them). No threshold of sections 3 to 8 changes. Because the selectors coincide on clips with at most k valid patches, such clips carry no information about the selector; the report states the share of test clips with at most k valid patches.

## Amendment 2 (25 Oct 2026, ESC-50 definitions, committed before any ESC-50 head is trained)

Section 10 left the ESC-50 job open. Fixed now, all secondary and outside the verdict:
- Data: `~/Documents/DATASETS/ESC-50/ast_768/esc-50.csv`, 2,000 clips, the 50-class `label`, the 5 official folds, seeds 0, 1, 2. Inner validation split of the training folds: `inner_split(tr, y, 0.1, seed, groups)` with groups = source recording (name token 1), the protocol 3 recipe of `config/esc50.yaml` (same training recipe as UrbanSound8K).
- Experts: E1 = mn10 W8A8 ESC-50 embedding (stage 3b), F = AST w4g128 at 512 frames (stage 2 `emb_ast_short/esc50`), masked arms = the six selector and rho cells at 512 frames (600 patches, k = 150 and 300), extracted by `t7_extract.py --max-length 512`.
  C1 on ESC-50 passed (`emb/esc50_c1/check_c1.json`: relative L2 error 0.0 against the stored embedding).
- Heads: the library recipe only (`LIB`: checkpoint and early stopping on macro F1 of the whole inner validation), one head per expert. No `KD` head, no field mix, no E1 reproduction check against an earlier run (there is no ESC-50 pilot).
- Flagged clips: E1's probability margin below the threshold that escalates 30 percent of the inner-validation clips (the pilot's `thr_rate` logic).
- Cost: priced as in section 5 with the ESC-50 stage 3c numbers (`analysis/esc50/costs.json`: fbank 4.2679e9, head 7.2942e8, `cnn:mn10:w8a8:esc` 3.4816e10, `ast:w4g128@512+esc` 2.8123e12); R3 on ESC-50 is the assertion that the formula reproduces 2.8123e12 within 1e-6.
- Rules (a1), (a2), (b1), (b2) are computed for `LIB_energy25` and reported with the same thresholds, (b2) scaled to 4 of 5 folds. They carry no verdict.
- One script, `t7_esc.py`, fits and scores. Within a cell all heads and the flagged set are fixed before the test labels are read, and nothing is selected afterwards.
