[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sebasmos/quantaudio/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/sebasmos/quantaudio)

# QWave: Quantized Embeddings for Efficient Audio Classification

> 🚧 **This repository is under active development.**
>
> 📄 Code and models will be released upon preprint upload or journal submission.

---

## 🔍 Overview

**QWave** provides an efficient and lightweight pipeline for soundscape classification based on quantized vector embeddings derived from pre-trained models. The framework supports ESC-50 and UrbanSound8K datasets and includes post-training quantization, cross-validation, and experiment tracking via Hydra.

---

## 📁 Project Structure

```text
QuantAudio/
├── configs/                   # Hydra configs for training and experiment tracking
│   └── configs.yaml           # Central configuration file
├── QWave/                     # Core source code
│   ├── datasets.py            # EmbeddingDataset class and quantization logic
│   ├── models.py              # Simple MLP classifier definition
│   ├── train_utils.py         # Training and logging utilities
│   ├── memory.py              # Memory usage profiler
│   └── utils.py               # Save, seeding, and metric helpers
├── scripts/                   # Run scripts
│   └── train_cv.py            # K-Fold cross-validation pipeline using Hydra
├── outputs/                   # Auto-generated experiment results
├── requirements.txt           # Python dependencies
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
pip install -r requirements.txt
```

---

## 🚀 Run Cross-Validation For Vector Embeddings Framework

You can run an experiment with:

```bash

python your_script.py experiment.metadata.tag=EfficientNet_esc50_imgs_1536 \
                        experiment.model.batch_size=64 \
                        experiment.model.epochs=100 \
                        experiment.device=cuda \
                        experiment.datasets.esc.csv=/path/to/your.csv \
                        experiment.datasets.esc.imgs=/path/to/your/images
```

For example: 

```bash
python run_trainer.py experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
                   experiment.cross_validation.n_splits=5 \
                   experiment.device=mps
                   experiment.model.batch_size=32 \
                   experiment.metadata.tag=experiment_name \
```

> ✅ This will save logs and checkpoints in `outputs/experiment_name/fold_*/`.

---

## 🔍 Config Overview (`configs.yaml`)

```yaml
experiment:
  datasets:
    esc:
      csv: "/absolute/path/to/esc-50.csv"

  model:
    batch_size: 32
    hidden_sizes: [256, 128, 64]
    learning_rate: 0.001

  training:
    epochs: 50
    early_stopping:
      patience: 10
      delta: 0.01

  cross_validation:
    n_splits: 5
    shuffle: true
    random_seed: 42

  logging:
    log_interval: 50
    save_checkpoint: true
    resume: true

  metadata:
    tag: "exp01"
    notes: "EfficientNet baseline on ESC-50"
```

---

## 📊 Features

- ✅ **Embedding extraction from EfficientNet / CLIP ViT**
- ✅ **Post-training quantization**
- ✅ **Cross-validation with reproducible config**
- ✅ **Class-imbalance handling**
- ✅ **Memory profiling & metrics logging**
- ✅ **Hydra integration for flexible experiments**

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
  author = {Sebastián Andrés Cajas Ordóñez and others},
  title = {QWave: Quantized Embeddings for Efficient Audio Classification},
  year = {2025},
  url = {https://github.com/sebasmos/QuantAudio},
  license = {MIT}
}
```

