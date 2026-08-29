# Contributing and reproducing

Everything needed to rerun the paper, extend the method, or file a useful bug
report. If something here does not work as written, that is a bug worth
reporting.

## Setup

```bash
conda create -n curious-qmoe python=3.11 -y && conda activate curious-qmoe
git clone https://github.com/sebasmos/curious-qmoe.git && cd curious-qmoe
pip install -e .
PYTHONPATH=. pytest tests/ -q          # 34 pass, 1 skipped
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
paper says so.

## Reproducing the paper

```bash
bash scripts/reproduce_paper.sh          # every run behind the tables
python scripts/collect_paper_numbers.py  # tables and significance tests
```

The first writes one directory per run under `outputs/<tag>/`, each with a
`summary.json` holding per-fold F1, timing, and carbon measurements. The second
reads those and prints every table in the paper, then writes
`paper_numbers.json` so each reported value traces to a named run.

> `paper_numbers.json` and all result directories are gitignored. Regenerate
> rather than trusting a copy you find on disk.

## Running one configuration

```bash
python scripts/benchmark.py \
    --config-path $(pwd)/config --config-name esc50 \
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
| Uniform-routing baselines | `*_uniform_ctrl` tags |
| Precision prior, main result | `*_pprior_a1.0` tags |
| Shared trunk, recommended | `*_sharedtrunk_a1.0` tags |
| Prior-strength sweep | `esc_pprior_a{0.3,0.5,1.0}` |
| Routing-strategy ablation | `curiosity_strategy=` in {`precision_prior`, `kl_divergence`, `precision_sharp`, `escalation`, `soft_escalation`} |
| Monte Carlo budget | `mc_samples=` in {1, 5, 10, 20} |
| Expert-set ablation | `expert_quantizations=` variants |
| Latency and router overhead | `scripts/analysis/latency_benchmark.py` |
| Routing and confidence analysis | `scripts/analysis/routing_analysis.py` |
| Figures | `scripts/analysis/generate_{architecture,confidence}_figure.py` |

## Honest ablations

Two efficiency ideas did not survive measurement and are reported anyway:

- **Uncertainty gating** (`gate_threshold=0.9`) runs the Monte Carlo passes only
  on ambiguous samples to cut router overhead. It scored below the uniform
  baseline, because suppressing the uncertainty estimate removes the signal
  exactly where the prior would act.
- **Warm starting** (`warm_start=`) initializes each expert from a trained
  single model. A first attempt reused one fold's checkpoint for every fold,
  which leaks validation data; the corrected per-fold version is statistically
  indistinguishable from training from scratch.

Both are in the paper's appendix for the same reason they are here: a negative
result that cost real compute is worth publishing.

## Determinism

Folds are stratified and seeded (`random_state=42`), so a rerun on the same
machine reproduces the same splits. Accuracy is device-independent. Absolute
timings are not: the paper's were measured on Apple Silicon CPUs and will
differ on other hardware.

## Contributing changes

- Open an issue before a large change, so the design can be agreed first.
- Keep `pytest tests/` green, and add a test for any behaviour you fix.
- A result only counts when it comes from a real run on real data, not from
  tests passing alone.
- Do not commit result directories, checkpoints, or `paper_numbers.json`.
