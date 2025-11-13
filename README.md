[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sebasmos/quantaudio/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/quantaudio)

# QWave: Learning to Route Curiously in Low-Bit Mixture-of-Experts


**QWave** is a curiosity-driven quantized Mixture-of-Experts framework for efficient audio classification on resource-constrained edge devices. QWave achieves 99.9% of full-precision accuracy with 4× compression and 82% latency variance reduction through Bayesian epistemic uncertainty-based routing.

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
conda create -n qwave python=3.11 -y
conda activate qwave
git clone https://github.com/sebasmos/qwave.git
cd qwave
pip install -e .
```

---

## Quick Start

### Basic Usage

```bash
cd scripts
python benchmark.py \
  --config-path /path/to/QWave/config \
  --config-name esc50 \
  experiment.datasets.esc.csv=/path/to/esc-50.csv \
  experiment.device=cpu \
  experiment.models_to_run=[esc]
```

### MoE with Curiosity Mode

```bash
python benchmark.py \
  --config-path /path/to/QWave/config \
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
QWave/
├── config/                    # Hydra configs
│   └── esc50.yaml             # ESC-50 configuration
├── QWave/                     # Core source code
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
│       ├── organize-results.py      # Combine CSV results
│       ├── analyze-std.py           # Generate tables with mean±std
│       ├── analyze-significance.py  # Statistical testing (t-tests, Levene)
│       └── README-significance.md   # Model nomenclature reference
├── outputs/                   # Auto-generated results
└── pyproject.toml
```

---

## Results Analysis

After running experiments, analyze results with the scripts in `scripts/tables/`:

### 1. Organize Results
Combine CSV files from multiple experiments:

```bash
cd scripts/tables
python organize-results.py  # Edit dataset path in script
```

### 2. Generate Tables (mean±std)
Create 5 tables with mean±std from 5-fold cross-validation:

```bash
python analyze-std.py
```

**Output:** `tables-std/` folder with 4 main tables + 1 supplementary

### 3. Statistical Significance Testing
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

This project is licensed under the [MIT License](https://github.com/sebasmos/QuantAudio/blob/main/LICENSE).

---

## Citation

```bibtex
@software{Cajas2025_QWave,
  author = {Cajas Ordóñez, Sebastián Andrés and Torres, Luis and Meno, Mackenzie and Lai, Yuan and Durán, Carlos and Celi, Leo Anthony},
  title = {QWave: Learning to Route Curiously in Low-Bit Mixture-of-Experts},
  year = {2025},
  url = {https://github.com/sebasmos/QWave},
  license = {MIT}
}
```
