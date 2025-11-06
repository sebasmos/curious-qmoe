[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sebasmos/quantaudio/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/quantaudio)

# QWave: Quantized Embeddings for Efficient Audio Classification

> 🚧 **This repository is under active development.**
>
> 📄 Code and models will be released upon preprint upload or journal submission.

---

## 🔍 Overview

**QWave** provides an efficient and lightweight pipeline for soundscape classification based on quantized vector embeddings derived from pre-trained models. The framework supports:

- **Multiple quantization schemes**: 1, 2, 4, 8, 16-bit, BitNet ternary, and bitwise popcount
- **Mixture-of-Experts (MoE)** with heterogeneous quantized experts
- **Bayesian Router with curiosity mode** for epistemic uncertainty estimation
- **Accurate model size calculation** for quantized models
- **Cross-validation and experiment tracking** via Hydra

**Datasets**: ESC-50 and UrbanSound8K

**Requirements**: CUDA 12.6+ (optional for GPU acceleration)

---

## 📁 Project Structure

```text
QWave/
├── config/                    # Hydra configs for experiments
│   └── esc50.yaml             # ESC-50 configuration with curiosity mode
├── QWave/                     # Core source code
│   ├── datasets.py            # EmbeddingDataset class and normalization
│   ├── models.py              # Neural network architectures (MLP, ESCModel)
│   ├── bitnnet.py             # BitNet quantized layers (1-16 bit, ternary)
│   ├── qmoe_layers.py         # Quantized MoE layers (BitwisePopcount)
│   ├── moe.py                 # MoE training and Bayesian Router with curiosity
│   ├── train_utils.py         # Training and validation utilities
│   ├── memory.py              # Model size calculation (quantization-aware)
│   ├── graphics.py            # Plotting (ROC, losses, curiosity distributions)
│   └── utils.py               # Helpers (seeding, device selection, metrics)
├── scripts/                   # Benchmark scripts
│   └── benchmark.py           # Main benchmarking pipeline with MoE support
├── outputs/                   # Auto-generated experiment results
├── pyproject.toml             # Package configuration
├── LICENSE
└── README.md
```

---

## ⚙️ Setup

### 1. Create Environment

```bash
conda create -n qwave python=3.11 -y
conda activate qwave
```

### 2. Install Requirements

```bash
git clone https://github.com/sebasmos/qwave.git
cd qwave
pip install -e .
```

---

## 🚀 Quick Start

### Basic Usage

Run benchmarks with the default configuration:

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

Enable Bayesian Router with epistemic uncertainty estimation:

```bash
python benchmark.py \
  --config-path /path/to/QWave/config \
  --config-name esc50 \
  experiment.device=cpu \
  experiment.datasets.esc.csv=/path/to/esc-50.csv \
  experiment.datasets.esc.normalization_type=standard \
  experiment.models_to_run=[moe] \
  experiment.router.expert_quantizations="[bitnet,'1','2','4','8','16',qesc]" \
  experiment.router.num_experts=3 \
  experiment.router.top_k=1 \
  experiment.router.use_curiosity=true \
  experiment.metadata.tag=esc_moe_curiosity
```

**Curiosity outputs** (saved per fold in `outputs/esc_moe_curiosity/moe/fold_X/`):
- `curiosity_values.json` - Raw uncertainty values per sample
- `curiosity_histogram.png` - Distribution of epistemic uncertainty
- `curiosity_per_class.png` - Average uncertainty per predicted class

> ✅ Results and checkpoints are saved in `outputs/<tag>/<model>/fold_*/`

---

## 🔍 Config Overview

The main configuration file is `config/esc50.yaml`. Key parameters:

```yaml
experiment:
  models_to_run: [esc]  # Options: esc, bitnet, moe, qmoe, 1, 2, 4, 8, 16, qesc
  device: "cpu"  # or "cuda", "mps"

  datasets:
    esc:
      csv: "/path/to/esc-50.csv"
      normalization_type: "standard"  # or "l2", "min_max", "raw"

  model:
    batch_size: 64
    hidden_sizes: [640, 320]
    learning_rate: 0.0005793146438537801
    dropout_prob: 0.1953403862875243
    epochs: 10

  router:  # For MoE models
    expert_quantizations: [1, 2, 4, 16]
    num_experts: 4
    top_k: 1
    use_curiosity: false  # Enable Bayesian Router with uncertainty
    load_balancing: true
    load_balancing_alpha: 1e-3

  cross_validation:
    n_splits: 5
    shuffle: true
    random_seed: 42

  metadata:
    tag: "experiment_name"
    notes: ""
```

---


## 📊 Features

- ✅ **Embedding extraction from EfficientNet / CLIP ViT**
- ✅ **Post-training quantization (1, 2, 4, 8, 16-bit)**
- ✅ **BitNet ternary quantization**
- ✅ **Mixture-of-Experts (MoE) with quantized experts**
- ✅ **Bayesian Router with curiosity-driven uncertainty estimation**
- ✅ **Accurate model size calculation for quantized models**
- ✅ **Cross-validation with reproducible config**
- ✅ **Class-imbalance handling**
- ✅ **Memory profiling & metrics logging**
- ✅ **Hydra integration for flexible experiments**

---

## 🔬 Benchmarking

### Model Size Calculation

QWave includes accurate model size calculation that accounts for quantization. Unlike traditional approaches that save `state_dict()` (which stores dequantized float32 weights), our implementation calculates the **theoretical quantized size** based on bit-width:

```python
from QWave.memory import print_size_of_model

# Automatically detects quantization and reports accurate size
print_size_of_model(model, label="quantized_model")
# Output: model: quantized_model     Size (KB): 12.345 [quantized]
```

**Supported quantization schemes:**
- **1-bit to 16-bit**: Symmetric quantization with scale factors
- **BitNet**: Ternary weights {-1, 0, 1} with per-channel alpha scaling
- **qesc**: Bitwise popcount with 2-bit ternary encoding

### Running Benchmarks

#### 1. Quantized Models Baseline (GPU)

Run all baseline quantized models (1, 2, 4, 8, 16-bit) and full-precision ESC model on GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python benchmark.py \
  --config-path /home/sebasmos/Desktop/QWave/config \
  --config-name esc50 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/home/sebasmos/Documents/DATASET/esc-50.csv \
  experiment.device=cuda \
  experiment.models_to_run="['1','2','4','8','16',esc]" \
  experiment.metadata.tag=benchmark_baselines
```

#### 2. BitNet Baseline (CPU)

Run BitNet ternary quantization and full-precision models on CPU:

```bash
python benchmark.py \
  --config-path /home/sebasmos/Desktop/QWave/config \
  --config-name esc50 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/home/sebasmos/Documents/DATASET/esc-50.csv \
  experiment.device=cpu \
  experiment.models_to_run="[bitnet,esc]" \
  experiment.metadata.tag=benchmark_baselines
```

#### 3. Mixture-of-Experts (MoE)

Run MoE with heterogeneous quantized experts (BitNet, 1-bit, 2-bit, 4-bit, 8-bit, 16-bit, qesc):

```bash
python benchmark.py \
  --config-path /home/sebasmos/Desktop/QWave/config \
  --config-name esc50 \
  experiment.device=cpu \
  experiment.datasets.esc.csv=/home/sebasmos/Documents/DATASET/esc-50.csv \
  experiment.datasets.esc.normalization_type=standard \
  experiment.models_to_run=[moe] \
  experiment.router.expert_quantizations="[bitnet,'1','2','4','8','16',qesc]" \
  experiment.router.num_experts=3 \
  experiment.router.top_k=1 \
  experiment.metadata.tag=esc_moe_bitnet_8_16_qesc
```

#### 4. MoE with Curiosity (Bayesian Router)

Enable **curiosity mode** to use a Bayesian Router with Monte Carlo Dropout for epistemic uncertainty estimation:

```bash
python benchmark.py \
  --config-path /home/sebasmos/Desktop/QWave/config \
  --config-name esc50 \
  experiment.device=cpu \
  experiment.datasets.esc.csv=/home/sebasmos/Documents/DATASET/esc-50.csv \
  experiment.datasets.esc.normalization_type=standard \
  experiment.models_to_run=[moe] \
  experiment.router.expert_quantizations="[bitnet,'1','2','4','8','16',qesc]" \
  experiment.router.num_experts=3 \
  experiment.router.top_k=1 \
  experiment.router.use_curiosity=true \
  experiment.metadata.tag=esc_moe_bitnet_8_16_qesc_curiosity
```

**Curiosity outputs** (saved per fold):
- `curiosity_values.json` - Raw uncertainty values per sample
- `curiosity_histogram.png` - Distribution of epistemic uncertainty
- `curiosity_per_class.png` - Average uncertainty per predicted class

---

## 🤝 Contributing

We welcome contributions! Fork the [repository](https://github.com/sebasmos/QuantAudio), make your improvements, and open a PR. Feature suggestions and bug reports are appreciated.

---

## 📄 License

This project is licensed under the [MIT License](https://github.com/sebasmos/QuantAudio/blob/main/LICENSE).

---

## 📙 Citation

```bibtex
@software{Cajas2025_QWave,
  author = {Cajas-Ordóñez, Sebastián Andrés and Torres, Luis and Meno, Mackenzie and Lai, Yuan and Durán, Carlos and Celi, Leo Anthony},
  title = {QWave: Quantized Audio Classification Framework},
  year = {2025},
  url = {https://github.com/sebasmos/QWave},
  license = {MIT}
}
```
