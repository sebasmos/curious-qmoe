#!/bin/bash
#
# Curiosity-Driven Routing: Full 5-Fold Cross-Validation
# Replication script for BitNet-Q4/8-QMoE-C configuration
#
# Date: 2026-01-26
# Configuration: [bitnet,'4','8'] - 3 heterogeneous experts
# Dataset: ESC-50
#

set -e  # Exit on any error

# ============================================================
# CONFIGURATION - UPDATE THESE PATHS BEFORE RUNNING
# ============================================================
#
# IMPORTANT: You MUST update these paths to match your local setup!
#

# 1. Path to the curious-qmoe repository root
REPO_ROOT="/Users/cajas.sebastian/Desktop/repositories/curious-qmoe"

# 2. Path to ESC-50 dataset CSV file (with EfficientNet-B0 1536-dim embeddings)
DATASET_CSV="/Users/cajas.sebastian/Documents/DATASETS/ESC-50/efficientnet_1536/esc-50.csv"

# 3. Device to use for training/inference (cpu, cuda, mps)
DEVICE="cpu"

# ============================================================
# DERIVED PATHS - DO NOT MODIFY
# ============================================================
SCRIPTS_DIR="${REPO_ROOT}/scripts"
CONFIG_PATH="${REPO_ROOT}/config"
OUTPUT_DIR="${REPO_ROOT}/outputs"

# Verify paths exist
if [ ! -d "$REPO_ROOT" ]; then
    echo "ERROR: Repository root not found: $REPO_ROOT"
    echo "Please update REPO_ROOT in this script to point to your curious-qmoe directory"
    exit 1
fi

if [ ! -f "$DATASET_CSV" ]; then
    echo "ERROR: Dataset CSV not found: $DATASET_CSV"
    echo "Please update DATASET_CSV in this script to point to your ESC-50 dataset"
    exit 1
fi

if [ ! -d "$CONFIG_PATH" ]; then
    echo "ERROR: Config directory not found: $CONFIG_PATH"
    echo "Please verify your repository structure"
    exit 1
fi

# Change to scripts directory
cd "$SCRIPTS_DIR"

echo "=================================="
echo "Starting Curiosity Experiments"
echo "=================================="
echo ""
echo "Configuration: BitNet-Q4/8-QMoE-C"
echo "Experts: [bitnet, 4, 8]"
echo "Folds: 5-fold cross-validation"
echo ""

# ============================================================
# Experiment 1: Baseline (No Curiosity)
# ============================================================
# echo "=================================="
# echo "Running Experiment 1/3: Baseline (No Curiosity)"
# echo "Expected F1: 0.7736 ± 0.0247"
# echo "=================================="
# echo ""

# python benchmark.py \
#   --config-path "$CONFIG_PATH" \
#   --config-name esc50 \
#   experiment.datasets.esc.csv="$DATASET_CSV" \
#   experiment.device=$DEVICE \
#   'experiment.models_to_run=[moe]' \
#   experiment.router.use_curiosity=false \
#   'experiment.router.expert_quantizations=[bitnet,4,8]' \
#   experiment.router.num_experts=3 \
#   experiment.router.top_k=1 \
#   experiment.cross_validation.n_splits=5 \
#   experiment.metadata.tag=full_baseline_final

# echo ""
# echo "Baseline experiment complete!"
# echo ""

# ============================================================
# Experiment 2: KL Divergence (Paper's Equation 8) - WINNER
# ============================================================
echo "=================================="
echo "Running Experiment 2/3: KL Divergence (Paper's Equation 8) ⭐"
echo "Expected F1: 0.7832 ± 0.0210 (+1.23% vs baseline)"
echo "=================================="
echo ""

python benchmark.py \
  --config-path "$CONFIG_PATH" \
  --config-name esc50 \
  experiment.datasets.esc.csv="$DATASET_CSV" \
  experiment.device=$DEVICE \
  'experiment.models_to_run=[moe]' \
  experiment.router.use_curiosity=true \
  experiment.router.curiosity_strategy=kl_divergence \
  experiment.router.curiosity_alpha=0.1 \
  'experiment.router.expert_quantizations=[bitnet,4,8]' \
  experiment.router.num_experts=3 \
  experiment.router.top_k=1 \
  experiment.cross_validation.n_splits=5 \
  experiment.metadata.tag=full_kl_divergence_final

echo ""
echo "KL Divergence experiment complete!"
echo ""

# # ============================================================
# # Experiment 3: Entropy Regularization
# # ============================================================
# echo "=================================="
# echo "Running Experiment 3/3: Entropy Regularization"
# echo "Expected F1: 0.7670 ± 0.0297 (-0.86% vs baseline)"
# echo "=================================="
# echo ""

# python benchmark.py \
#   --config-path "$CONFIG_PATH" \
#   --config-name esc50 \
#   experiment.datasets.esc.csv="$DATASET_CSV" \
#   experiment.device=$DEVICE \
#   'experiment.models_to_run=[moe]' \
#   experiment.router.use_curiosity=true \
#   experiment.router.curiosity_strategy=entropy_regularization \
#   experiment.router.curiosity_alpha=0.1 \
#   'experiment.router.expert_quantizations=[bitnet,4,8]' \
#   experiment.router.num_experts=3 \
#   experiment.router.top_k=1 \
#   experiment.cross_validation.n_splits=5 \
#   experiment.metadata.tag=full_entropy_regularization_final

# echo ""
# echo "=================================="
# echo "All experiments complete!"
# echo "=================================="
# echo ""
# echo "Results saved to:"
# echo "  - ${OUTPUT_DIR}/full_baseline_final/moe/summary.json"
# echo "  - ${OUTPUT_DIR}/full_kl_divergence_final/moe/summary.json"
# echo "  - ${OUTPUT_DIR}/full_entropy_regularization_final/moe/summary.json"
# echo ""
# echo "To analyze results, run:"
# echo "  cd ${SCRIPTS_DIR}/tables"
# echo "  python analyze-curiosity-results.py"
# echo ""
