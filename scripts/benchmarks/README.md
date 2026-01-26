# Benchmark Scripts

## models.sh

Replication script for curiosity-driven routing experiments on ESC-50 dataset.

This script is designed to be **shareable** - others can use it to replicate the experiments by simply updating a few configuration variables at the top of the script.

---

## Quick Start

### 1. Configure the Script

**BEFORE RUNNING**, open `models.sh` and update these 3 variables at the top:

```bash
# 1. Path to the curious-qmoe repository root
REPO_ROOT="/path/to/your/curious-qmoe"

# 2. Path to ESC-50 dataset CSV file (with EfficientNet-B0 1536-dim embeddings)
DATASET_CSV="/path/to/your/ESC-50/efficientnet_1536/esc-50.csv"

# 3. Device to use for training/inference (cpu, cuda, mps)
DEVICE="cpu"
```

The script will validate these paths exist before running and provide clear error messages if something is missing.

### 2. Run the Script

```bash
cd /path/to/curious-qmoe/scripts/benchmarks
./models.sh
```

---
