# Difficulty mixture of experts, T1b: specialization with a stronger E2 and richer specialist input

Written and committed on 24 Oct 2026 before any head of T1b was trained and before any T1b score was computed. Branch `moe/t1b` of curious-qmoe
(from `moe/difficulty-pilot`, whose working tree was clean). Scripts: `scripts/redesign/moe/t1b_*.py`. Source plans: PLAN.md
(`~/Documents/qmoe-results/2026-10-24_2001_difficulty-moe-plan/`, sections 3 and 9, design D1) and the pilot's `ANALYSIS_PLAN.md`
(`~/Documents/qmoe-results/2026-10-24_2013_moe-difficulty-pilot_code-b77b98e/`), whose Part B definitions apply unchanged except where this file says otherwise.
Machine: CPU only, `nice -n 10`, 2 threads in total.

## 1. Status of this test

T1b is an amendment written after seeing T1. T1 (primary arm R10 against its uniform twin on UrbanSound8K) stopped at -2.37 points, and TAU shift stopped
at -1.71. Those numbers, and the per-arm tables of the pilot, were known when this plan was written. T1b is therefore not an independent test. It is the
second specialization test on the same data and counts in the family of tests: the family on each dataset is {T1 primary, T1b primary}, and the verdict
reports the raw Nadeau-Bengio p beside its Holm-adjusted p over that family of two (threshold 0.025 for the smaller one). The success rule below is the T1
rule, unchanged; the family adjustment is reported, and a result that passes the raw rule but not the adjusted p is called a pass of the rule with an adjusted
p above 0.05, nothing stronger.

## 2. Why T1b

In the pilot the second expert E2 was a 4-bit AST at 256 frames. On the clips E1 flags, that E2 (emb only, 59.64 percent) was not better than E1 (61.31 percent).
A head trained on a model's frozen embedding cannot learn what the embedding does not carry, and an expert no better than the cheap model on the flagged clips
leaves no room for a specialist to help. T1b asks whether specialization appears when E2 is a stronger model on the flagged clips and when the specialist also sees
the cheap model's embedding.

## 3. What does not change from the pilot

- Data, folds, seeds: UrbanSound8K, 10 official folds, seeds 0 to 2, row order of the CSV; TAU 2020 Mobile, 5 location-grouped folds, seeds 0 to 2, training on device-A
  rows of the training folds, test conditions clean (device A) and shift (the other eight devices), as in Amendment 1 of the pilot. Seeds are averaged inside a fold; the
  fold is the unit of every test.
- E1, the out-of-fold difficulty labels, the tiers, the escalated training set S_2, the thresholds tau_1 and tau_1^oof, the flagged test clips, the fit and inner-validation
  splits are the pilot's own, read from the pilot's `t1/raw/s<seed>_f<fold>.npz` and `tau/raw/s<seed>_f<fold>.npz` (arrays `tr, te, z1_oof, S2, tier, fit_p, iv_p, tau1,
  tau1_oof, z1_te, z1_iv, flag`, and `cond` for TAU). The flagged set of every test fold is thus identical to T1's, which makes the T1 and T1b contrasts comparable.
  Nothing from the pilot is retrained. The E2 cross-fit of the pilot is not reused or repeated (it served T1a and E3, neither is part of T1b).
- Training recipe: the pilot's `heads.fit_head` (protocol 3 recipe, per-sample weights, focal exponent), selection of every E2 head on accuracy of the inner-validation clips
  with m_1^oof < tau_1^oof, the same rule for the uniform twin and every specialist. Only the hidden sizes of the C input head differ (below).
- Arms and weights (pilot B6): U (all clips, weight 1), M0 (S_2 only), R05, R10, R20 (S_2 weight 1, outside 0.05, 0.10, 0.20), F (focal, gamma 2, all clips), T
  (difficulty-tier mask, easy tier removed). The primary specialist is R10. The twin of an arm is U with the same input, the same E2 and the same seed.
- Margin and flagged definition: probability margin, flagged when m_1 < tau_1.
- Metrics and tests: contrast per fold = arm minus twin accuracy on the clips E1 flags, seeds averaged inside the fold, in points. Nadeau-Bengio corrected two-sided t with
  K folds (df K-1), the ratio n_test/n_train the mean over folds (and, for TAU, per condition), 95 percent interval from the same standard error, Cohen's d_z, 10,000-draw
  bootstrap interval over folds, counts of folds at or above +0.5 and +1.0.

## 4. What is new

### 4.1 E2 candidates (stored clean embeddings, read only)

| Name | UrbanSound8K | TAU 2020 Mobile |
|---|---|---|
| E2a (primary) | AST FP32, 1024 frames: idea 5 stage 2 `emb/urban8k/fp32__clean.npy` (768-d) | AST FP32, 1024 frames: option B `emb_ast/tau/fp32__clean.npy` |
| E2b | AST w4g128 (4-bit group-wise), 1024 frames: idea 5 stage 2 `emb/urban8k/w4g128__clean.npy` | AST w4g128, 1024 frames: option B `emb_ast/tau/w4g128__clean.npy` (the pilot's E2 on TAU) |
| E2c | AST w4g128, 400 frames: idea 6 stage 2 `emb_ast_short/urban8k/w4g128__clean.npy` | none stored |

The `urban8k_400` and `emb_ast` folders of the final round and of stage 3c hold corrupted conditions only (or no AST embeddings), so the clean 400-frame embedding
comes from stage 2. There is no 400-frame TAU embedding. Row order and row count (8,732 and 23,035) are checked against the CSVs by `t1b_check_inputs.py`, together
with the sidecar json (config, frame length). For TAU, E2a is a stronger model than the pilot's E2 on the same task; for UrbanSound8K, E2a, E2b and E2c all differ from the
256-frame E2 of the pilot. E2b and E2c are secondary and are run with fewer arms (4.3).

### 4.2 Specialist inputs

| Input | Features | Head hidden sizes |
|---|---|---|
| A | AST embedding only | [640, 320] (library) |
| B | AST embedding, z_1, m_1 (the pilot's input B) | [640, 320] |
| C (new, primary) | cheap embedding (mn10 W8A8, 960-d), AST embedding (768-d), z_1, m_1 | [1024, 512] |

z_1 and m_1 are out-of-fold for training and inner-validation clips and from the final E1 for test clips, as in the pilot. The cheap embedding is the E1 input and exists at
run time because E1 runs first. The extra head cost of input C (about 2 million multiply-accumulates against 1.3e14 bit-operations of the FP32 AST) is not priced; the pipeline
cost below is E1 plus the flagged share times the E2 backbone cost.

### 4.3 Arms run

| E2 | Input | Arms |
|---|---|---|
| E2a | C | U, M0, R05, R10 (primary), R20, F, T |
| E2a | B | U, M0, R05, R10, R20, F, T |
| E2a | A | U, M0, R10 |
| E2b | C | U, M0, R10 |
| E2b | B | U, R10 |
| E2b | A | U |
| E2c (UrbanSound8K) | C | U, M0, R10 |
| E2c (UrbanSound8K) | B | U, R10 |
| E2c (UrbanSound8K) | A | U |

The primary test is R10 on input C with E2a minus U on input C with E2a. Input B with E2a isolates factor (a) (stronger E2 alone), input C isolates the added input (b).
Every arm is reported, every contrast has its own row. No arm, input or E2 is chosen with test results.

## 5. Step order and gates

1. `t1b_check_inputs.py`: shapes, finiteness, sidecars, row order against the CSV.
2. Phase 1, `t1b_jobs.py --phase strength`: for every (seed, fold), the uniform A-input head ("U-A", AST embedding only, uniform weights) for every E2 candidate. Writes
   logits on test, on inner validation.
3. Strength gate (c), `t1b_strength.py`: on the flagged test clips, accuracy of E1, of the uniform AST head U-A of E2a, and of the oracle (E1 or U-A correct), fold means over
   UrbanSound8K (TAU: the shift condition is the gate condition, clean beside it). E2a counts as stronger when the mean over folds of (U-A minus E1) on the flagged clips is at least
   +2.0 points and positive in at least 7 of 10 folds (TAU shift: at least 4 of 5). This threshold is fixed here, before the numbers exist; it asks for a gap about the size of the
   +1.0 point success margin plus the uncertainty of fold means. If the gate fails on UrbanSound8K, T1b is not informative, the report says so with these numbers, phase 2 is not run
   on either dataset and the work stops. If it fails on TAU only, TAU phase 2 is not run and TAU is reported as not informative.
4. Phase 2, `t1b_jobs.py --phase arms`: all arms of 4.3.
5. `t1b_analysis.py`: contrasts, verdicts, pipeline tables.
6. Only if T1b PASSES (section 7): T2's first comparison (`t1b_t2.py`), supervised router against the margin threshold on the frozen experts, labelled as T2 and pre-registered in a short amendment committed
   before it is run.

## 6. Pipeline analyses (secondary, reported for every arm)

- Equal share, equal cost: for every arm, the cascade prediction is E2-arm on the flagged clips and E1 elsewhere. The flagged set is the same for the specialist and for the uniform twin,
  so the flagged share and the cost c_E1 + share x c_E2 are equal. Reported: cascade accuracy and macro F1, specialist cascade minus uniform-twin cascade (fold-level NB, as in section 3).
- Cost units: the repo's cost module values (stage 3c `analysis/urban/costs.json`, option B `analysis/tau/costs.json`): `cnn:mn10:w8a8:esc` for E1, `ast:fp32+esc` for E2a, `ast:w4g128+esc` for
  E2b, `ast:w4g128@400+esc` for E2c.
- Hull: the lower convex envelope of the pilot's recomputed statics (`t0/static_recomputed.csv`, `t0_tau/static_recomputed.csv`, mean macro F1 over seeds and folds against cost; clean for UrbanSound8K, shift
  and clean for TAU), evaluated at each arm's pipeline cost, by linear interpolation of fold macro F1 between the two bracketing statics (the exact expected macro F1 of time sharing, used in T0, differs
  slightly from linear interpolation; this is stated wherever the number appears). Only the hull without the always-average point is built here; the always-average point needs the stored heads' logits and is not recomputed.
  Cascade macro F1 minus hull F1 per fold, NB test, for the primary arm and for the uniform twin.
- Comparison with T1: the primary contrast of T1b is listed beside the pilot's R10 contrast (same flagged clips).

## 7. Success and stop rules (the T1 rules, primary arm only)

Primary arm: R10 on input C with E2a minus U on input C with E2a, on the clips E1 flags.
- PASS (proof): contrast >= +1.0 point in at least 7 of 10 UrbanSound8K folds, Nadeau-Bengio two-sided p < 0.05, positive mean. TAU: at least 4 of 5 folds, flagged shift clips.
- STOP: mean contrast < +0.3 point.
- Between the two: INCONCLUSIVE.
- A GO for D1 stage 1 needs the contrast at or above +0.5 in at least 7 of 10 folds (T1a already passed on UrbanSound8K, mean test Spearman 0.61).
The verdict is declared from UrbanSound8K. TAU shift is a confirmation with its own verdict. Secondary arms (everything else in 4.3) are exploratory; they carry Holm-adjusted p within their family (input C arms
of E2a; input B arms of E2a; input A arms of E2a; E2b arms; E2c arms) and never change the verdict. Descriptive, not tested: U on C minus U on B minus U on A (what the richer input does to the uniform head).

## 8. Deviations stated in advance

1. Only E2 candidates with stored clean embeddings are used; no extraction is run.
2. E1 and the difficulty labels come from the pilot's raw files, not retrained.
3. T1a is not repeated (its strong model was the 256-frame head).
4. The hull comparison is without the always-average point and by linear interpolation of F1 (section 6).
5. The strength gate threshold (+2.0 points) is mine, set before any number of T1b exists.
