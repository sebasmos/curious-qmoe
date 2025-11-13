# Model Nomenclature for CVPR Paper

## Scripts

- **organize-results.py**: Combines CSV results → `data/*.csv`
- **analyze-results.py**: Generates tables (mean only) → `tables/`
- **analyze-std.py**: Generates tables (mean±std) → `tables-std/`
- **analyze-significance.py**: Statistical tests → `significance-tests/`

---

## Model Names

The `analyze-significance.py` script uses the following standardized model names:

| Old Name | New Name | Meaning |
|----------|----------|---------|
| **Single Models** |
| ESC-Base† | **FP32-Base** | Full precision baseline (32-bit floating point) |
| QESC-Base‡ | **Q8-Base-PTQ** | 8-bit post-training quantized baseline |
| Q1-Base | **Q1-Base** | ✓ Keep (1-bit quantization-aware training) |
| Q2-Base | **Q2-Base** | ✓ Keep (2-bit quantization-aware training) |
| Q4-Base | **Q4-Base** | ✓ Keep (4-bit quantization-aware training) |
| Q8-Base | **Q8-Base** | ✓ Keep (8-bit quantization-aware training) |
| Q16-Base | **Q16-Base** | ✓ Keep (16-bit quantization-aware training) |
| BitNet-Base | **BitNet-Base** | ✓ Keep (Ternary quantization baseline) |
| **MoE Models** |
| BitNet-QESC-QMoE | **BitNet-Q8PTQ-QMoE** | MoE with 8-bit PTQ experts, uniform routing |
| BitNet-QESC-QMoE-C | **BitNet-Q8PTQ-QMoE-C** | MoE with 8-bit PTQ experts, curiosity routing |
| BitNet-Q8/16-QESC-QMoE | **BitNet-Q8/16-PTQ-QMoE** | MoE with 8-bit/16-bit PTQ experts, uniform routing |
| BitNet-Q8/16-QESC-QMoE-C | **BitNet-Q8/16-PTQ-QMoE-C** | MoE with 8-bit/16-bit PTQ experts, curiosity routing |
| BitNet-Q4/8-QMoE-C | **BitNet-Q4/8-QMoE-C** | ✓ Keep (4-bit/8-bit QAT experts, curiosity routing) |
| BitNet-Q4/8/16-QMoE | **BitNet-Q4/8/16-QMoE** | ✓ Keep (4/8/16-bit QAT experts, uniform routing) |
| BitNet-Q4/8/16-QMoE-C | **BitNet-Q4/8/16-QMoE-C** | ✓ Keep (4/8/16-bit QAT experts, curiosity routing) |

**Key Changes:**
- ESC-Base† (FP32) → **FP32-Base** (clearer naming)
- QESC-Base‡ (INT8) → **Q8-Base-PTQ** (distinguishes PTQ from QAT)
- All QESC MoE models → use `PTQ` or `Q8PTQ` prefix

---

## Usage

```bash
cd notebooks
python analyze-significance.py  # Runs statistical tests
```

**Significance levels:** `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` p≥0.05

**Output:** 6 CSV files in `significance-tests/`:
1. `table1_f1_significance.csv` - Cross-dataset F1 vs FP32-Base
2. `table2_f1_significance.csv` - Quantization ablation F1
3. `table3_f1_significance.csv` - MoE F1 vs FP32-Base
4. `table4_latency_significance.csv` - Latency speedup tests
5. `table3_energy_significance.csv` - Energy efficiency tests
6. `latency_variance_reduction.csv` - Variance reduction (Levene's test)

**Metrics from `fold_X/metrics.json`:**
- `best_f1` - F1-score
- `val_duration` - Latency (÷5 for per-sample)
- `train_energy_consumed` - Energy (joules)
- `val_duration_std` - Latency variance
