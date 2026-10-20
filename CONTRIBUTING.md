# Contributing and reproducing

Everything needed to rerun the paper, extend the method, or file a useful bug
report. If something here does not work as written, that is a bug worth
reporting.

## Setup

```bash
conda create -n curious-qmoe python=3.11 -y && conda activate curious-qmoe
git clone https://github.com/sebasmos/curious-qmoe.git && cd curious-qmoe
pip install -e .
PYTHONPATH=. pytest tests/ -q          # all pass, 1 script-style file skipped
```

The tests pin the routing mechanism itself: the prior is the identity at
`alpha=0`, monotone in `alpha`, and matches the closed form of the
polarization-only ablation. They catch the class of bug where a routing rule
silently degrades to a no-op, which happened once in this project's history.

## Data

Three datasets, as 1536-dimensional EfficientNet-B3 embedding CSVs:

| Dataset | Task in the paper | Source |
|---|---|---|
| ESC-50 | 5 super-categories (`class_id`) | [karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50) |
| Quinn | 5 soundscape classes | Quinn et al. 2022 |
| UrbanSound8K | 10 urban classes | [urbansounddataset](https://urbansounddataset.weebly.com/urbansound8k.html) |

Point each config at your copy:

```yaml
# config/esc50.yaml
experiment:
  datasets:
    esc:
      csv: /path/to/esc-50.csv
      label_col: class_id     # 5 super-categories, NOT the 50 fine classes
```

**On the ESC-50 label column.** Every result in the paper uses `class_id`, the
five super-categories. The 50 fine-grained classes are in `label`. This is why
our ESC-50 numbers are not comparable to published 50-class results, and the
paper says so. `experiment.datasets.esc.label_col=label` trains on the 50
classes with the same folds; the extra job list below does that.

**Second backbone.** ESC-50 and UrbanSound8K also exist as 768-dimensional AST
embedding CSVs with the same rows, folds and labels; see
[data/audio_embeddings](data/audio_embeddings/README.md). The input dimension
is always read from the CSV.

## Evaluation protocol

- **Folds.** ESC-50 and UrbanSound8K use their official predefined folds
  (`cross_validation.fold_column: folder`), leave one fold out: 5 and 10 folds.
  Random splits of UrbanSound8K put slices of one recording on both sides.
  Quinn has no official folds and uses `StratifiedKFold(5, shuffle=True,
  random_state=42)`, the same split for every seed, so every arm and seed is
  evaluated on identical test folds.
- **Model selection.** A stratified 10% inner validation split is carved out
  of the training folds (`random_state` = run seed). Early stopping, the LR
  scheduler and checkpoint selection see only that split; the held-out fold is
  scored once, with the selected checkpoint. Feature scaling is fit on the
  remaining training rows only.
- **Seeds.** `experiment.seed` seeds python, numpy, torch and the loader at the
  start of each fold (init, dropout, MC dropout, loader order, inner split).
  The matrix runs ESC-50 over seeds 0 to 4, Quinn and UrbanSound8K over 0 to 2.
- **Metric.** Macro F1 is primary (selection and reporting); accuracy and
  weighted F1 are logged. `summary.json` stores every fold's values with its
  fold id and seed, so arms can be paired on (seed, fold).
- **Training recipe.** One block, `experiment.train`, for single models and
  MoE alike: AdamW, label smoothing, class-weighted loss, batch 64, at most 100
  epochs, early stopping, plateau LR schedule and gradient clipping at 1.0. It
  is copied into every summary.
- **Uniform control.** The same Bayesian router as the prior arms with the
  prior off (`curiosity_alpha=0`), for full experts and for the shared trunk.
  Every precision-prior run also re-scores its checkpoints with the prior off
  at inference (`alpha0_eval`), and records which expert each held-out sample
  reaches, overall and by uncertainty quartile.
- **Efficiency.** Measured after training, never alongside it:
  `scripts/analysis/efficiency_benchmark.py` times batch-1 inference on
  held-out samples (50 warmup, 1000 timed calls, median and IQR, arms
  interleaved, threads pinned) and wraps CodeCarbon around a fixed number of
  batch-1 calls, reported as mJ per inference (kWh x 3.6e9 / n). On Apple
  silicon without powermetrics CodeCarbon runs in constant (TDP) mode, so
  energy is constant power times time; the mode and power are recorded. k-bit
  and BitNet layers are fake-quantized in fp32, so only FP32, dynamic int8 and
  the MoE configurations are timed; storage comes from per-layer bit-widths.

## Reproducing the paper

```bash
export ESC_CSV=/path/esc-50.csv QUINN_CSV=/path/ABGQI_embeddings_torch.csv URBAN_CSV=/path/urbansound8k.csv
bash scripts/reproduce_paper.sh --list > jobs.txt         # one training command per line
tr '\n' '\0' < jobs.txt | xargs -0 -P 5 -n 1 sh -c       # 5 jobs at a time, 2 threads each
bash scripts/reproduce_paper.sh --analysis                # efficiency and routing, alone
python scripts/collect_paper_numbers.py                   # tables and tests
```

Reviewer experiments run the same way, from their own job list (also needs
`AST_ESC_CSV` and `AST_URBAN_CSV`):

```bash
bash scripts/reproduce_paper.sh --list-extra > jobs-extra.txt   # 217 training commands
tr '\n' '\0' < jobs-extra.txt | xargs -0 -P 5 -n 1 sh -c
BASE_OUTPUTS=/path/to/main/outputs bash scripts/reproduce_paper.sh --analysis-extra   # alone
python scripts/collect_paper_numbers.py --outputs outputs /path/to/main/outputs
```

They cover a mixed-precision single model (`mp8_4`: first layer 8-bit, the
rest 4-bit, the single-model control for the shared trunk), single Q8 and FP32
models widened to the parameter count of the full-expert MoE and of the
shared-trunk MoE (`pm8_moe`, `pm32_moe`, `pm8_st`, `pm32_st`, widths computed
from the MoE count and stored as `param_match` in `summary.json`), ESC-50 at
50 classes (`esc50c_*` tags) and the AST backbone (`ast_*` tags).
`--analysis-extra` runs `scripts/analysis/cascade.py`, a post-hoc Q4 to Q8 or
FP32 cascade escalating on low confidence or high MC-dropout variance, with
thresholds chosen on each fold's inner validation split, then times the new
single models. `BASE_OUTPUTS` points at the main matrix's outputs when the
extra jobs ran in another checkout. The collector puts these in their own
tables and test families; the paper tables do not change.

`bash scripts/reproduce_paper.sh` with no flag runs the same jobs serially,
then the analysis. Each run writes `outputs/<tag>_s<seed>/<model_kind>/` with
`summary.json`, `config.yaml`, and per fold `split.json` plus the selected
checkpoint; logs go to `outputs/logs/`. The collector writes
`outputs/paper_numbers.json` and one LaTeX table per paper table to
`outputs/paper_tables/` (both paths can be changed with `--json` and
`--tex-dir`). Tests are two-sided paired t-tests on (seed, fold) with a
Wilcoxon check and Bonferroni within each declared family; variance is
compared with Levene's test on per-(seed, fold) scores.

> `paper_numbers.json` and all result directories are gitignored. Regenerate
> rather than trusting a copy you find on disk.

## Running one configuration

```bash
python scripts/benchmark.py \
    --config-path $(pwd)/config --config-name esc50 \
    experiment.seed=0 "experiment.models_to_run=[moe]" \
    experiment.router.use_curiosity=true \
    experiment.router.curiosity_strategy=precision_prior \
    experiment.router.curiosity_alpha=1.0 \
    experiment.router.shared_trunk=true \
    experiment.metadata.tag=my_run
```

Use an absolute `--config-path`. Keep the dataset key as `esc` whatever the
dataset is; that key names the config block, not the corpus. And keep
`expert_quantizations` the same length as `num_experts`, or construction fails
with an explicit error.

## Claim to command

| Paper claim | Produced by |
|---|---|
| Uniform-routing baselines | `*_uniform_ctrl` and `*_sharedtrunk_uniform_ctrl` tags (alpha 0) |
| Precision prior, main result | `*_pprior_a1.0` tags |
| Shared trunk, recommended | `*_sharedtrunk_a1.0` tags |
| Prior-strength sweep | `esc_pprior_a{0.3,0.5,1.0}` |
| Routing-strategy ablation | `esc_<strategy>_a0.3` tags, `curiosity_strategy=` in {`kl_divergence`, `entropy_regularization`, `precision_sharp`, `escalation`, `soft_escalation`} |
| Monte Carlo budget | `mc_samples=` in {1, 5, 10, 20} |
| Expert-set ablation | `expert_quantizations=` variants |
| Latency, energy and router overhead | `scripts/analysis/efficiency_benchmark.py` |
| Routing and confidence analysis (held-out folds) | `scripts/analysis/routing_analysis.py` |
| Figures | `scripts/analysis/generate_{architecture,confidence}_figure.py` |
| Reviewer experiments | `--list-extra` tags (`*_singles_mp`, `*_singles_pm`, `esc50c_*`, `ast_*`) and `scripts/analysis/cascade.py` |

## Honest ablations

Two efficiency ideas did not survive measurement under the `arxiv-v4` protocol
and are reported anyway:

- **Uncertainty gating** (`gate_threshold=0.9`) runs the Monte Carlo passes only
  on ambiguous samples to cut router overhead. It scored below the uniform
  baseline, because suppressing the uncertainty estimate removes the signal
  exactly where the prior would act. The current matrix re-runs it
  (`esc_gated_a1.0`).
- **Warm starting** (`warm_start=`) initializes each expert from a trained
  single model. A first attempt reused one fold's checkpoint for every fold,
  which leaks validation data; the corrected per-fold version is statistically
  indistinguishable from training from scratch. It is not in the current run
  matrix.

Both are in the paper's appendix for the same reason they are here: a negative
result that cost real compute is worth publishing.

## Determinism

Folds are predefined (ESC-50, UrbanSound8K) or fixed by `random_state=42`
(Quinn), and every fold is seeded from `experiment.seed`, so a rerun with the
same seed and thread count on the same machine gives identical fold scores.
Absolute timings are hardware-specific.

## Contributing changes

- Open an issue before a large change, so the design can be agreed first.
- Keep `pytest tests/` green, and add a test for any behaviour you fix.
- A result only counts when it comes from a real run on real data, not from
  tests passing alone.
- Do not commit result directories, checkpoints, or `paper_numbers.json`.
