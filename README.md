[![arXiv](https://img.shields.io/badge/arXiv-2511.11743-b31b1b.svg)](https://arxiv.org/abs/2511.11743)
[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/quantaudio)

# Uncertainty-Aware Routing Makes Quantized Mixture-of-Experts Stable


**curious-qmoe** is an uncertainty-aware quantized Mixture-of-Experts framework for efficient audio classification on resource-constrained edge devices. curious-qmoe matches full-precision accuracy at 4-bit with 4× compression, and its Bayesian uncertainty-based routing polarizes expert selection to stabilize cross-fold performance and inference cost.

**Key Features:**
- **Heterogeneous Quantization**: BitNet ternary, BitLinear (1-16 bit), post-training quantization (PTQ) with bitwise operations
- **Uncertainty-Aware Routing**: a Bayesian router (Monte Carlo dropout) whose precision prior shifts uncertain samples toward higher-precision experts by construction
- **Mixture-of-Experts**: Dynamic expert selection across quantized models for adaptive precision
- **Hardware-Efficient**: Optimized for edge deployment with predictable latency (29 ms std)
- **Comprehensive Evaluation**: Energy consumption, carbon emissions, and statistical significance testing
- **Reproducible**: Hydra configuration management, cross-validation, experiment tracking

**Datasets:** ESC-50, Quinn, UrbanSound8K

---

## Setup

```bash
conda create -n curious-qmoe python=3.11 -y
conda activate curious-qmoe
git clone https://github.com/sebasmos/curious-qmoe.git
cd curious-qmoe
pip install -e .
```

---

## Quick Start

### Basic Usage

```bash
cd scripts
python benchmark.py \
  --config-path /path/to/curious-qmoe/config \
  --config-name esc50 \
  experiment.datasets.esc.csv=/path/to/esc-50.csv \
  experiment.device=cpu \
  experiment.models_to_run=[esc]
```

### MoE with Uncertainty-Aware Routing (precision prior)

```bash
python benchmark.py \
  --config-path /path/to/curious-qmoe/config \
  --config-name esc50 \
  experiment.device=cpu \
  experiment.datasets.esc.csv=/path/to/esc-50.csv \
  experiment.models_to_run=[moe] \
  experiment.router.expert_quantizations="[bitnet,'1','2','4','8','16',qesc]" \
  experiment.router.num_experts=3 \
  experiment.router.top_k=1 \
  experiment.router.use_curiosity=true \
  experiment.router.curiosity_strategy=precision_prior \
  experiment.router.curiosity_alpha=1.0 \
  experiment.metadata.tag=esc_moe_uncertainty_aware
```

**Uncertainty outputs** (saved per fold):
- `curiosity_values.json` - Raw uncertainty values
- `curiosity_histogram.png` - Distribution of epistemic uncertainty
- `curiosity_per_class.png` - Average uncertainty per class

---

## Project Structure

```text
curious-qmoe/
├── config/                    # Hydra configs
│   └── esc50.yaml             # ESC-50 configuration
├── curious_qmoe/              # Core source code
│   ├── datasets.py            # EmbeddingDataset and normalization
│   ├── models.py              # Neural architectures (MLP, ESCModel)
│   ├── bitnnet.py             # BitNet quantized layers
│   ├── qmoe_layers.py         # Quantized MoE layers
│   ├── moe.py                 # MoE training and Bayesian Router
│   ├── train_utils.py         # Training/validation utilities
│   ├── memory.py              # Model size calculation
│   ├── graphics.py            # Plotting (ROC, losses, curiosity)
│   └── utils.py               # Helpers (seeding, device, metrics)
├── scripts/
│   ├── benchmark.py           # Main benchmarking pipeline
│   └── tables/                # Results analysis scripts
│       ├── analyze-std.py           # Generate tables with mean±std
│       ├── analyze-significance.py  # Statistical testing (t-tests, Levene)
│       └── README-significance.md   # Model nomenclature reference
├── outputs/                   # Auto-generated results
└── pyproject.toml
```

---

## Results Analysis

After running experiments, analyze results with the scripts in `scripts/tables/`:

### 1. Verify Paper Numbers
Deterministic verification of all MoE-related numbers against paper tables:

```bash
python scripts/tables/verify_paper_tables.py
```

**Output:** Per-value comparison with OK/MISMATCH status for Tables 1, 3, 4, 5, and prose claims.

### 2. Generate Tables (mean±std)
Create 5 tables with mean±std from 5-fold cross-validation:

```bash
cd scripts/tables
python analyze-std.py
```

**Output:** `tables-std/` folder with 5 CSV files:
- `table1-cross_dataset_performance_std.csv` (Table 1)
- `table2-ablation_quantization_std.csv` (Table 2)
- `table3-moe_curiosity_std.csv` (Table 3)
- `table4-inference_latency_std.csv` (Table 4)
- `supplementary-carbon_emissions_std.csv` (Supplementary)

### 3. Statistical Significance Testing
Run paired t-tests and variance tests:

```bash
cd scripts/tables
python analyze-significance.py
```

**Output:** `significance-tests/` folder with 6 CSV files:
- F1-score comparisons (Tables 1-3)
- Latency speedup tests (Table 4)
- Energy efficiency tests (Table 3)
- Variance reduction analysis (Levene's test)

**Model nomenclature:** See `scripts/tables/README-significance.md` for standardized names (FP32-Base, Q8-Base-PTQ, etc.)

---

## Config Overview

Key parameters in `config/esc50.yaml`:

```yaml
experiment:
  models_to_run: [esc]  # Options: esc, bitnet, moe, qmoe, 1, 2, 4, 8, 16, qesc
  device: "cpu"  # or "cuda", "mps"

  datasets:
    esc:
      csv: "/path/to/esc-50.csv"
      normalization_type: "standard"

  model:
    batch_size: 64
    hidden_sizes: [640, 320]
    learning_rate: 0.0005793146438537801
    epochs: 10

  router:  # For MoE models
    expert_quantizations: [1, 2, 4, 16]
    num_experts: 4
    top_k: 1
    use_curiosity: false  # Enable Bayesian Router
    load_balancing: true

  cross_validation:
    n_splits: 5
    shuffle: true
    random_seed: 42
```

---


**Supported schemes:**
- **1-bit to 16-bit**: Symmetric quantization with scale factors
- **BitNet**: Ternary weights {-1, 0, 1} with per-channel scaling
- **qesc**: Bitwise popcount with 2-bit ternary encoding

---

## Reproducing ECCV 2026 Paper Results

All MoE-C (curiosity routing) experiments use corrected Eq. 8 implementation. Results are in `outputs/` and `docs-temp/eccv-paper/results/`.

### Datasets

- **ESC-50**: `/path/to/ESC-50/efficientnet_1536/esc-50.csv`
- **Quinn**: `/path/to/Quinn/efficientnet_ABGQI/ABGQI_embeddings_torch.csv`
- **UrbanSound8K**: `/path/to/Urban8k/efficientnet/urbansound8k.csv`

### 3-Expert MoE-C (BitNet-Q4/8-QMoE-C, Table 1 main config)

```bash
# Baseline (no curiosity) — per dataset
python scripts/qMoE/qmoe.py \
  --config-path /path/to/curious-qmoe/config --config-name esc50 \
  "experiment.router.expert_quantizations=['bitnet', 4, 8]" \
  experiment.router.num_experts=3 experiment.router.use_curiosity=false \
  experiment.datasets.esc.csv=/path/to/dataset.csv \
  experiment.device=cpu experiment.metadata.tag=<dataset>_baseline

# Curiosity α=0.2
python scripts/qMoE/qmoe.py \
  --config-path /path/to/curious-qmoe/config --config-name esc50 \
  "experiment.router.expert_quantizations=['bitnet', 4, 8]" \
  experiment.router.num_experts=3 experiment.router.use_curiosity=true \
  experiment.router.curiosity_strategy=kl_divergence \
  experiment.router.curiosity_alpha=0.2 experiment.router.mc_samples=10 \
  experiment.datasets.esc.csv=/path/to/dataset.csv \
  experiment.device=cpu experiment.metadata.tag=<dataset>_alpha_020

# Curiosity α=0.3
python scripts/qMoE/qmoe.py \
  --config-path /path/to/curious-qmoe/config --config-name esc50 \
  "experiment.router.expert_quantizations=['bitnet', 4, 8]" \
  experiment.router.num_experts=3 experiment.router.use_curiosity=true \
  experiment.router.curiosity_strategy=kl_divergence \
  experiment.router.curiosity_alpha=0.3 experiment.router.mc_samples=10 \
  experiment.datasets.esc.csv=/path/to/dataset.csv \
  experiment.device=cpu experiment.metadata.tag=<dataset>_alpha_030
```

### 4-Expert MoE-C (BitNet-Q4/8/16-QMoE-C)

```bash
python scripts/qMoE/qmoe.py \
  --config-path /path/to/curious-qmoe/config --config-name esc50 \
  "experiment.router.expert_quantizations=['bitnet', 4, 8, 16]" \
  experiment.router.num_experts=4 experiment.router.use_curiosity=true \
  experiment.router.curiosity_strategy=kl_divergence \
  experiment.router.curiosity_alpha=0.3 experiment.router.mc_samples=10 \
  experiment.datasets.esc.csv=/path/to/dataset.csv \
  experiment.device=cpu experiment.metadata.tag=<dataset>_bitnet_4_8_16_curiosity
```

### PTQ MoE-C Variants

```bash
# BitNet-Q8/16-PTQ-QMoE-C (4 experts: bitnet, 8, 16, qesc)
python scripts/qMoE/qmoe.py \
  --config-path /path/to/curious-qmoe/config --config-name esc50 \
  "experiment.router.expert_quantizations=['bitnet', '8', '16', 'qesc']" \
  experiment.router.num_experts=4 experiment.router.use_curiosity=true \
  experiment.router.curiosity_strategy=kl_divergence \
  experiment.router.curiosity_alpha=0.3 experiment.router.mc_samples=10 \
  experiment.datasets.esc.csv=/path/to/dataset.csv \
  experiment.device=cpu experiment.metadata.tag=<dataset>_bitnet_8_16_qesc_curiosity

# BitNet-Q8PTQ-QMoE-C (2 experts: bitnet, qesc)
python scripts/qMoE/qmoe.py \
  --config-path /path/to/curious-qmoe/config --config-name esc50 \
  "experiment.router.expert_quantizations=['bitnet', 'qesc']" \
  experiment.router.num_experts=2 experiment.router.use_curiosity=true \
  experiment.router.curiosity_strategy=kl_divergence \
  experiment.router.curiosity_alpha=0.3 experiment.router.mc_samples=10 \
  experiment.datasets.esc.csv=/path/to/dataset.csv \
  experiment.device=cpu experiment.metadata.tag=<dataset>_bitnet_qesc_curiosity
```

### Generate Rebuttal Figure

```bash
python scripts/analysis/generate_rebuttal_figure.py \
  --routing-json outputs-rebuttal/analysis/outputs-0.2/rebuttal_routing/routing_results.json \
  --output docs-temp/eccv-paper/figs/rebuttal_confidence_distribution.pdf
```

### Output Tags → Paper Model Names

| Tag pattern | Paper name |
|-------------|-----------|
| `*_baseline` | Baseline MoE (uniform routing) |
| `*_alpha_020` | BitNet-Q4/8-QMoE-C (α=0.2) |
| `*_alpha_030` | BitNet-Q4/8-QMoE-C (α=0.3) |
| `*_bitnet_4_8_16_curiosity` | BitNet-Q4/8/16-QMoE-C |
| `*_bitnet_8_16_qesc_curiosity` | BitNet-Q8/16-PTQ-QMoE-C |
| `*_bitnet_qesc_curiosity` | BitNet-Q8PTQ-QMoE-C |

Each experiment outputs `summary.json` with `f1_mean`, `f1_std`, per-fold results, carbon emissions, and timing data.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

```bibtex
@article{ordonez2025uncertainty,
  title={Uncertainty Makes It Stable: Curiosity-Driven Quantized Mixture-of-Experts},
  author={Ord{\'o}{\~n}ez, Sebasti{\'a}n Andr{\'e}s Cajas and Torres, Luis Fernando Torres and Meni, Mackenzie J and Paredes, Carlos Andr{\'e}s Duran and Arazo, Eric and Bosch, Cristian and Carbajo, Ricardo Simon and Lai, Yuan and Celi, Leo Anthony},
  journal={arXiv preprint arXiv:2511.11743},
  year={2025}
}
```
