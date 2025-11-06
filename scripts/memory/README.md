# Model Size Calculator

Standalone script to calculate theoretical model sizes for all quantization schemes in QWave.

## Features

- ✅ **No training required** - just instantiates models and calculates sizes
- ✅ **All quantization schemes** - 1, 2, 4, 8, 16-bit, BitNet, qesc
- ✅ **Accurate calculations** - uses theoretical quantized size based on bit-width
- ✅ **CSV export** - save results for analysis
- ✅ **Visual comparison** - bar charts and reduction ratios

## Usage


### Save to CSV

```bash
python scripts/memory/calculate_model_sizes.py --output model_sizes.csv
```

### Custom Configuration

```bash
python scripts/memory/calculate_model_sizes.py \
  --in_dim 2048 \
  --num_classes 10 \
  --hidden_sizes 512 256 128 \
  --dropout 0.2
```

### Debug Mode

Show layer-by-layer breakdown:

```bash
python scripts/memory/calculate_model_sizes.py --debug
```

## Output

The script outputs:

1. **Per-model breakdown** with verification of quantization settings
2. **Summary table** with all models and their sizes
3. **CSV file** (if `--output` specified) with columns:
   - Model
   - Quantization type
   - Bits per weight
   - Parameter count
   - Size (KB)
   - Size (MB)
   - Reduction vs FP32

4. **Key insights** including most efficient model and size progression visualization

## Example Output

```
================================================================================
📋 SUMMARY TABLE
================================================================================
     Model    Quantization  Bits  Params  Size (KB)  Size (MB) Reduction vs FP32
ESC (FP32)            None    32 1206770   4830.805   4.830805             1.00x
     1-bit       BitLinear     1 1211122    241.616   0.241616            19.99x
     2-bit       BitLinear     2 1211122    390.096   0.390096            12.38x
     4-bit       BitLinear     4 1211122    687.056   0.687056             7.03x
     8-bit       BitLinear     8 1211122   1280.976   1.280976             3.77x
    16-bit       BitLinear    16 1211122   2468.816   2.468816             1.96x
    BitNet         Ternary     2 1211122    393.928   0.393928            12.26x
      qesc BitwisePopcount     2 1205760    308.640   0.308640            15.65x
```

## Notes

- **Theoretical vs Actual Size**: The script calculates what the model size SHOULD BE if properly quantized for deployment. PyTorch checkpoint files (.pth) will still be larger because `torch.save()` converts everything to FP32.

- **Parameter Count**: All quantized models have similar parameter counts (slightly different due to architectural variations). Size reduction comes from bits-per-parameter, not fewer parameters.

- **qesc**: Requires the `qmoe_layers` module. If not available, will skip qesc and continue with other models.

## Integration with Benchmark

This script is **separate from the main benchmark** (`scripts/benchmark.py`). The benchmark focuses on training and accuracy metrics, while this script focuses on model size analysis.

To get both accuracy AND size metrics:
1. Run `benchmark.py` to get accuracy/F1/training metrics
2. Run this script to get accurate model size comparisons

### Benchmark Test Commands

The script now includes MoE (Mixture-of-Experts) models. Here are the corresponding benchmark commands to test accuracy/F1 with training:

#### 1. MoE without Curiosity

Standard MoE with heterogeneous experts (bitnet, 1-bit, 2-bit, 4-bit, 8-bit, 16-bit, qesc):

```bash
cd scripts
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

**What it does:**
- Trains MoE with 3 experts selected from the pool of quantizations
- Router learns to select best expert per sample
- No uncertainty estimation

#### 2. MoE with Curiosity (Bayesian Router)

MoE with epistemic uncertainty estimation via Monte Carlo Dropout:

```bash
cd scripts
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

**What it does:**
- Same as above BUT with Bayesian Router
- Estimates epistemic uncertainty per prediction
- Generates curiosity plots: `curiosity_histogram.png`, `curiosity_per_class.png`
- Saves uncertainty values: `curiosity_values.json`

**Curiosity outputs** are saved per fold in: `outputs/<tag>/moe/fold_X/`

### Differences: MoE vs MoE+Curiosity

| Feature | MoE | MoE+Curiosity |
|---------|-----|---------------|
| **Router Type** | Standard | Bayesian (MC Dropout) |
| **Uncertainty Estimation** | ❌ No | ✅ Yes |
| **Inference Speed** | Faster (1 pass) | Slower (multiple passes) |
| **Model Size** | ~Same | Slightly larger (router params) |
| **Use Case** | Production inference | Research, active learning |
| **Outputs** | Predictions only | Predictions + uncertainty |

### Quick Comparison

To compare model sizes without training:

```bash
# Get theoretical sizes for all models including MoE
python scripts/memory/calculate_model_sizes.py --output comparison.csv
```

Expected output includes:
- Individual quantized models (1, 2, 4, 8, 16-bit)
- BitNet ternary
- qesc
- **MoE (bitnet,1,2)** ← Heterogeneous experts
- **MoE+Curiosity (bitnet,1,2)** ← With Bayesian Router

## Troubleshooting

**ImportError for timm**: The script handles missing `timm` by using a conditional import. ESCModel doesn't require timm for size calculation.

**ImportError for qmoe_layers**: If qesc import fails, the script will skip qesc model and continue with others. This is expected if the qmoe module has dependency issues.

**ImportError for MoE (sklearn, hydra, etc.)**: MoE models require additional dependencies (`sklearn`, `hydra-core`, `omegaconf`). If these are not installed, the script will skip MoE models and continue with single-model quantization schemes. To enable MoE calculations:

```bash
pip install scikit-learn hydra-core omegaconf
```

Or install all QWave dependencies:

```bash
cd /home/sebasmos/Desktop/QWave
pip install -e .
```
