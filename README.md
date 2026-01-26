[![arXiv](https://img.shields.io/badge/arXiv-2512.02646-b31b1b.svg)](https://arxiv.org/pdf/2511.11743)
[![LICENSE](https://img.shields.io/badge/license-CC%20BY--NC--SA-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/quantaudio)

# Uncertainty Makes It Stable: Curiosity-Driven Quantized Mixture-of-Experts


**curious-qmoe** is a curiosity-driven quantized Mixture-of-Experts framework for efficient audio classification on resource-constrained edge devices. curious-qmoe achieves 99.9% of full-precision accuracy with 4× compression and 82% latency variance reduction through Bayesian epistemic uncertainty-based routing.

**Key Features:**
- **Heterogeneous Quantization**: BitNet ternary, BitLinear (1-16 bit), post-training quantization (PTQ) with bitwise operations
- **Curiosity-Driven Routing**: Bayesian router with Monte Carlo dropout for epistemic uncertainty estimation
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
git clone https://github.com/sebasmos/QWave.git
cd QWave
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

### MoE with Curiosity Mode

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
  experiment.metadata.tag=esc_moe_curiosity
```

**Curiosity outputs** (saved per fold):
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

### 1. Generate Tables (mean±std)
Create 5 tables with mean±std from 5-fold cross-validation:

```bash
python analyze-std.py
```

**Output:** `tables-std/` folder with 4 main tables + 1 supplementary

### 2. Statistical Significance Testing
Run paired t-tests and variance tests:

```bash
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

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

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
