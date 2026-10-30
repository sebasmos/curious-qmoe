# Idea 6 option C: analysis plan

Written and committed on 23 Oct 2026 while the two new AST configurations were being extracted (started 21:33) and
before any head of a new system was trained or any test fold of a new system was scored. Results folder:
`~/Documents/qmoe-results/2026-10-23_2133_redesign-idea6-optionC_code-d6e7913/` (this file is copied there).

Why. Stage 3c (`2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1`) found that a margin-gated cascade from a cheap
EfficientAT CNN to a quantized AST beats the time-sharing hull on ESC-50/50 (+2.11 points, Holm 0.016) and fails on
the five groups and UrbanSound8K (Holm 0.072 each); the exact-length rerun (`2026-10-23_1918_..._unpadded_code-cbf6be6`)
re-tested Stage 3b's UrbanSound8K design with exact-length heads (+1.37, raw p 0.0014) while its own fresh pick failed
(+0.95, p 0.074). Option C asks whether a cheaper expensive tier changes this: if the AST pass costs little, the
cascade's mean cost stays near the CNN's and the hull it has to beat is made of cheap statics.

Stage 3b's plan (`scripts/redesign/idea6/stage3b/ANALYSIS_PLAN.md`) and Stage 3c's hold for everything not restated
here: tasks, audio, the priced cost view, the gate, the hull, the statistics, the head recipe.

## What the costs are before anything is measured (priced view, backbone only, T bit-ops per clip)

The priced view charges one 32 x 32 rescale per output element of every quantized layer, so a group of G input
columns costs 1024 / G bit-ops per multiply-add on top of the weight bits times 8. Per Linear multiply-add: w4g128
and n4g128 40, w3g64 and n3g64 40, w3g32 56, per-channel w4 and n4 about 33. The 3-bit group-64 tiers therefore cost
exactly what w4g128 costs at the same length, and group 32 costs 29% more. Only the per-channel 4-bit tiers and
shorter inputs are cheaper.

| length | w8g64 | w4g128 | n4, w4 | w3g64, n3g64 | w3g32 | fp32 |
|---|---|---|---|---|---|---|
| 512 (ESC-50) | 4.853 | 2.807 | 2.449 | 2.807 | 3.625 | 59.3 |
| 400 (UrbanSound8K) | 3.679 | 2.082 | 1.802 | 2.082 | 2.721 | 45.1 |
| 256 | 2.274 | 1.248 | 1.068 | 1.248 | 1.658 | 28.0 |

Unpriced (Stage 2's E3 view, group scales free): w3g64, n3g64 and w3g32 are 1.989 at 512 frames and 1.443 at 400
against 2.398 and 1.763 for every 4-bit configuration, 17% less. So the brief's premise, that 3-bit AST is a cheaper
expensive tier, holds in the unpriced view only. The primary analysis keeps the priced view, as every stage since
Stage 2; the unpriced view is a secondary family (S4). This is stated before any result exists.

Reference static F1 already known (Stage 3c `analysis/e1_front.csv`, ESC-50/50, FP32 head transferred from the
512-frame FP32 AST): w3g64@512 0.924 clean / 0.819 field, w4g128@512 0.946 / 0.855, n4@512 0.937 / 0.844, n4@256
0.916 / 0.805; mn20 W8A8 static 0.955 / 0.853 at 0.09 T. The 3-bit tier with transferred heads is below the cheap
CNN on the field mixture; whether heads trained on its own embeddings close the gap is part of what this run measures.

## New systems

AST configurations at the clips' real length (512 frames on ESC-50, 400 on UrbanSound8K), Stage 2's
`extract_ast_short.py` on Stage 2's audio (same CSVs, background clips, offsets, field assignments; sidecars record
the costs and the names checksum; `bg_plan.json` must equal Stage 2's for every clip drawn):
- `n3g64`: nested 3-bit child of the stored 8-bit group-64 parent (`QWave/backbone_quant.py`, top three bits of the
  w8g64 codes with the parent scales times 32).
- `w3g32`: direct round-to-nearest 3-bit weights with a scale per output row and 32 input columns.
- `w3g64` at 512/400 (Stage 2, `emb_ast_short/`) and `n4` at 256 frames (Stage 3b, `emb_ast/<ds>_256`) exist and
  are reused unchanged; `n3g32` is not run (same bit-ops as w3g32, and the nested rule was already weaker than direct
  RTN at 3 bits in idea 5 Stage 2).
Conditions: ESC-50 clean and the twelve noisy cells (background, pink, white x 20, 10, 5, 0 dB); UrbanSound8K clean,
background 10 and 0 dB in full, the other seven field cells for the clips the test or validation assignment gives
them. 8-bit activations as every quantized AST (per-token Linear inputs, 16-bit softmax probabilities).

Heads, two sources per 3-bit tier, kept as separate systems in every hull and family:
- transferred: Stage 2's `astshort_<task>_s<seed>/<esc|4>` heads (trained on the clean FP32 embeddings at 512/400
  frames, scaler fitted on the FP32 fit rows) applied to the new embeddings, exactly as every AST tier has been scored
  so far (`score_eff.py --dim 768`). Names `n3g64@512`, `w3g32@512` (`@400` on UrbanSound8K).
- own: heads trained on the tier's own clean embeddings with the protocol 3 recipe (`scripts/benchmark.py`, standard
  normalization, inner splits grouped by recording, CPU, 2 threads; seeds 0 to 4 on the ESC-50 tasks, 0 to 2 on
  UrbanSound8K; kinds `esc` and `4`), tags `own_<cfg>_<task>_s<seed>`, for the three 3-bit tiers w3g64, n3g64 and
  w3g32. Every head must reproduce its `summary.json` F1 when rescored, and its inner split must equal the reference
  run's (checked row by row by the analysis, as in Stage 3c). Names `w3g64o@512`, `n3g64o@512`, `w3g32o@512`.
Stage 3c's lesson (P4, P5) is that a head trained on one embedding distribution does not transfer to another; the
own heads are the honest 3-bit systems, the transferred ones keep the comparison with Stage 3c's protocol.

Costs: as Stage 3c, the fbank front end at 32 x 32 averaged over the clips, the backbone from `ast_bitops_tok` with
the sidecar's unpriced bit-ops as a check, the head's bit-ops. Priced view primary.

## Design family and selection (per task, Holm over the three tasks within each family)

Tasks: ESC-50 50 classes, ESC-50 five fixed groups, UrbanSound8K. The official categories (esc5off) are not run.

Cheap tier fixed to Stage 3c's family: `mn10` W8A8 and `mn20` W8A8 (on UrbanSound8K the exact-length mn10 with the
heads retrained at exact length, `mn10unpr` of the exact-length rerun, in place of the padded mn10 in every hull and
design, as that rerun did), four head kinds each (`esc`, `8`, `4`, `bitnet`). Expensive tier: every AST tier of
Stage 3c (19 on the ESC-50 tasks, 17 on UrbanSound8K) plus the five new systems (`n3g64`, `w3g32` transferred;
`w3g64o`, `n3g64o`, `w3g32o` own), the tier's `esc` head deciding as before. Rules `ast` and `avg`, budgets b05 to
b50. 2 backbones x 4 heads x 24 (22) tiers x 2 rules x 5 budgets = 1,920 designs per ESC-50 task, 1,760 on
UrbanSound8K.

Selection on clean inner-validation rows only (Stage 2's procedure, unchanged): each design's macro F1 minus the
expected macro F1 of the hull of every clean static system at the design's mean inner-validation cost, averaged over
(seed, fold); the largest mean gain is picked, ties to the lower cost. Hull set "new": Stage 2's, Stage 3b's and
Stage 3c's statics, the exact-length mn10 statics on UrbanSound8K (padded mn10 excluded there), and the new AST
systems with both head sources.

The test folds were used by Stage 3c and the exact-length rerun for designs that are in this family too (every
design whose tier is not new). If the pick's tier is not new, H1 is a re-test of a Stage 3c design and the README
says so; the fresh content of this run is then the fact that no new tier was picked.

## Hypotheses

- H1 (primary, Holm over 3): the pick, on the field mixture, against the time-sharing hull of every static system
  of the "new" set at the pick's mean field-mixture cost. PASS when the mean fold difference is positive and the
  Holm-adjusted fold-level t p is below 0.05. Nadeau-Bengio and the exact sign-flip p are reported next to it.
- H2 (primary, Holm over 3): the same pick on clean test folds against the clean hull.
- Option C passes when H1 passes on all three tasks; that is the question the brief asks.

## Secondary (Holm within each family, raw p alongside)

- S1, gates at the pick's escalation count on the field mixture: the random gate (exact expected F1 of the mixture),
  idea 7's `noise_lr` audio-statistic gate at the same count, the best static at no higher cost; the oracle at the
  same count reported. Families of 3 for each.
- S2, humility of the pick's cheap tier: escalation rate with the clean thresholds on every extracted condition,
  steps down in SNR where the rate does not rise; AUROC of 1 - margin for "cheap wrong" on every condition and for
  "cheap wrong, tier right" on clean and field.
- S3, static headroom per task: the best static of the "new" set on the field mixture at any cost, the best at no
  more than the pick's cost, the pick's F1, the oracle at the pick's count, the oracle over every fixable clip.
- S4, the unpriced view: the same selection and H1 test with every cost taken at group-scale price 0 (CNN backbones
  at their unpriced bit-ops, AST tiers at `ast_bitops_tok(cfg, 0.0)`), family of 3. This is where the 3-bit tiers
  are cheaper than the 4-bit ones.
- S5, the new tiers as statics: F1 on clean and the field mixture with transferred and own heads, own minus
  transferred per (seed, fold) with the fold t test, for the three 3-bit tiers on every task (descriptive table), and
  own 3-bit minus the transferred w4g128 static at the same priced cost.

## Exploratory (no verdict)

- E1, three tiers: CNN cheap tier to a 3-bit AST (own head) to the 8-bit AST w8g64 at the same length (the parent of
  n3g64; stored once in the nested case, two models otherwise). Gate 1 as before (the CNN head's margin, threshold
  at budget b1 on the clean inner-validation rows). Gate 2: the 3-bit tier's `esc`-head margin, threshold set at
  rate b2 of the clean inner-validation rows of the same (seed, fold), applied to the clips gate 1 escalated. Final
  prediction: the deciding rule of the last tier reached (`ast`: that tier's head; `avg`: mean of the probabilities
  of every tier the clip ran). Cost: cheap tier plus rate1 x 3-bit tier plus rate1 x rate2 x 8-bit tier, every
  rate measured on the test rows of the condition. Family: cheap in {mn10 (mn10unpr), mn20} W8A8 x 4 heads x 3-bit
  tier in {w3g64o, n3g64o, w3g32o} x rule x b1 in {b05 to b50} x b2 in {0.2, 0.3, 0.5, 0.7}. Selection on clean
  inner validation as H1; the pick tested on the field mixture against the "new" hull with raw p, and against the
  two-tier H1 pick of the same task (fold t, raw p). The question is whether a third tier changes anything, so the
  E1 pick is also compared with the best two-tier design that uses the same cheap tier.
- E2, every two-tier design of the family on the field mixture against the "new" hull, Holm over all designs of the
  three tasks.
- E3, reproduction: with the new AST systems removed, this run's selection must return Stage 3c's P3 pick on the
  ESC-50 tasks when the cheap family is restricted to mn20 (the Stage 3c pick's backbone) and the exact-length rerun's
  pick on UrbanSound8K, with gains equal to the last digit; the rows of those picks against the hull must equal the
  stored `selected.csv` rows.

## Rules fixed now

- Nothing above changes after a test fold of a new system is scored; any later analysis is labelled post hoc.
- Every new system must be complete (every seed, head and needed condition) before H1 is computed; a system that
  cannot be completed leaves every hull and family and the README says so.
- A failed extraction or run is rerun from its checkpoint with the same code; a code change is a deviation and is
  reported.
- Extraction on MPS, at most two processes; every CPU job at nice 10 with 2 threads; the machine is shared with
  option B.
- Real data only; every number in the README comes from a file in the results folder; nothing in any earlier
  results folder is modified.
