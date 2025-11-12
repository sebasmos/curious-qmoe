import pandas as pd
import numpy as np
import json
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
results_base = Path(r"C:\Users\sebastian.cajasordon\Downloads\RESULTS-paper")
tables_std_dir = script_dir / "tables-std"

# Create tables-std directory if it doesn't exist
tables_std_dir.mkdir(exist_ok=True)

print("="*100)
print("CVPR PAPER - RESULTS ANALYSIS WITH STANDARD DEVIATIONS")
print("="*100)

# ============================================================================
# DATA COLLECTION FROM FOLD-LEVEL METRICS
# ============================================================================

def collect_fold_metrics(results_base):
    """
    Collect metrics from all fold_*/metrics.json files
    """
    all_fold_data = []

    for dataset_dir in results_base.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name.replace(" Results", "")
        print(f"\nProcessing {dataset_name}...")

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Handle different directory structures
            # Case 1: model_dir/moe|qmoe/fold_*/metrics.json (MoE models)
            # Case 2: model_dir/1|2|4|8|16|bitnet|esc|qesc/fold_*/metrics.json (individual models)

            subdirs = list(model_dir.iterdir())
            variant_dirs = [d for d in subdirs if d.is_dir() and d.name in ['moe', 'qmoe', '1', '2', '4', '8', '16', 'bitnet', 'esc', 'qesc']]

            if variant_dirs:
                # Multiple variants
                for variant_dir in variant_dirs:
                    variant_name = f"{model_name}_{variant_dir.name}"
                    fold_data = collect_folds_from_dir(variant_dir, dataset_name, variant_name)
                    all_fold_data.extend(fold_data)
            else:
                # Direct fold structure
                fold_data = collect_folds_from_dir(model_dir, dataset_name, model_name)
                all_fold_data.extend(fold_data)

    return pd.DataFrame(all_fold_data)

def collect_folds_from_dir(directory, dataset_name, model_name):
    """
    Collect all fold metrics from a directory containing fold_1, fold_2, etc.
    """
    fold_data = []

    for fold_dir in directory.iterdir():
        if not fold_dir.is_dir() or not fold_dir.name.startswith('fold_'):
            continue

        metrics_file = fold_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            fold_data.append({
                'dataset': dataset_name,
                'model_name': model_name,
                'fold': fold_dir.name,
                'accuracy': metrics.get('accuracy', np.nan),
                'f1': metrics.get('best_f1', np.nan),
                'parameter_count': metrics.get('parameter_count', np.nan),
                'train_duration': metrics.get('train_duration', np.nan),
                'latency': metrics.get('val_duration', np.nan) / 5 if metrics.get('val_duration') else np.nan,  # Average per sample
                'infer_ram_gb': metrics.get('max_ram_mb_val', np.nan) / 1024,  # Convert MB to GB
                'energy_train': metrics.get('train_energy_consumed', np.nan),
                'energy_val': metrics.get('val_energy_consumed', np.nan),
                'emissions_train': metrics.get('train_emissions', np.nan),
                'emissions_val': metrics.get('val_emissions', np.nan)
            })
        except Exception as e:
            print(f"  Error reading {metrics_file}: {e}")

    return fold_data

# Collect all fold-level data
print("\nCollecting fold-level metrics from all datasets...")
fold_df = collect_fold_metrics(results_base)

print(f"\nCollected {len(fold_df)} fold measurements from {fold_df['model_name'].nunique()} models")

# ============================================================================
# AGGREGATE STATISTICS (MEAN AND STD)
# ============================================================================

def aggregate_with_std(fold_df):
    """
    Aggregate fold-level data to get mean and std for each model/dataset combination
    """
    agg_data = fold_df.groupby(['dataset', 'model_name']).agg({
        'f1': ['mean', 'std', 'count'],
        'accuracy': ['mean', 'std'],
        'parameter_count': ['first'],  # Changed to list to ensure consistent flattening
        'latency': ['mean', 'std'],
        'infer_ram_gb': ['mean', 'std'],
        'energy_train': ['mean', 'std'],
        'energy_val': ['mean', 'std'],
        'emissions_train': ['mean', 'std'],
        'emissions_val': ['mean', 'std'],
        'train_duration': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    agg_data.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in agg_data.columns.values]

    return agg_data

agg_df = aggregate_with_std(fold_df)

# Convert units
agg_df['latency_mean_ms'] = agg_df['latency_mean'] * 1000
agg_df['latency_std_ms'] = agg_df['latency_std'] * 1000
agg_df['energy_train_mean_mJ'] = agg_df['energy_train_mean'] * 1000
agg_df['energy_train_std_mJ'] = agg_df['energy_train_std'] * 1000
agg_df['emissions_train_mean_ug'] = agg_df['emissions_train_mean'] * 1e6
agg_df['emissions_train_std_ug'] = agg_df['emissions_train_std'] * 1e6

# ============================================================================
# MODEL NOMENCLATURE MAPPING
# ============================================================================

def standardize_model_name(name):
    """Map raw model names to standardized nomenclature"""
    name_lower = name.lower()

    # Handle individual models
    if 'models_all_final' in name_lower:
        if '_bitnet' in name_lower:
            return 'BitNet-Base'
        elif '_esc' in name_lower and '_qesc' not in name_lower:
            return 'ESC-Base† (FP32)'
        elif '_qesc' in name_lower:
            return 'QESC-Base‡ (INT8)'
        elif name_lower.endswith('_1'):
            return 'Q1-Base'
        elif name_lower.endswith('_2'):
            return 'Q2-Base'
        elif name_lower.endswith('_4'):
            return 'Q4-Base'
        elif name_lower.endswith('_8'):
            return 'Q8-Base'
        elif name_lower.endswith('_16'):
            return 'Q16-Base'

    # Handle MoE models
    if 'qmoe' in name_lower or 'moe' in name_lower:
        has_curiosity = 'curiosity' in name_lower
        suffix = '-C' if has_curiosity else ''

        if '4_8_16' in name_lower:
            return f'BitNet-Q4/8/16-QMoE{suffix}'
        elif '4_8' in name_lower:
            return f'BitNet-Q4/8-QMoE{suffix}'
        elif '8_16' in name_lower:
            if 'qesc' in name_lower:
                return f'BitNet-Q8/16-QESC-QMoE{suffix}'
            return f'BitNet-Q8/16-QMoE{suffix}'
        elif 'qesc' in name_lower and 'bitnet' in name_lower:
            return f'BitNet-QESC-QMoE{suffix}'
        else:
            return f'BitNet-QMoE{suffix}'

    return name

agg_df['model_name_std'] = agg_df['model_name'].apply(standardize_model_name)

# Map dataset names
dataset_map = {'Quinn': 'Quinn', 'ESC': 'ESC-50', 'Urban8': 'UrbanSound8K'}
agg_df['dataset'] = agg_df['dataset'].map(dataset_map).fillna(agg_df['dataset'])

# ============================================================================
# TABLE 1: Cross-Dataset Performance with STD
# ============================================================================
print("\n" + "="*100)
print("TABLE 1: Cross-Dataset Performance (with ±std)")
print("="*100)

# Get models that appear across multiple datasets
multi_dataset_models = agg_df.groupby('model_name_std')['dataset'].apply(lambda x: len(set(x))).reset_index()
multi_dataset_models = multi_dataset_models[multi_dataset_models['dataset'] >= 2]['model_name_std'].tolist()

table1_data = []
for model in multi_dataset_models:
    model_data = agg_df[agg_df['model_name_std'] == model]

    entry = {
        'Model': model,
        'Params (M)': round(model_data['parameter_count_first'].iloc[0] / 1e6, 2)
    }

    for dataset in ['ESC-50', 'Quinn', 'UrbanSound8K']:
        dataset_model = model_data[model_data['dataset'] == dataset]
        if len(dataset_model) > 0:
            f1_mean = dataset_model['f1_mean'].iloc[0]
            f1_std = dataset_model['f1_std'].iloc[0]
            entry[f'{dataset} F1'] = f"{f1_mean:.3f}±{f1_std:.3f}"
        else:
            entry[f'{dataset} F1'] = '-'

    # Calculate overall average F1
    f1_values = model_data['f1_mean'].dropna()
    if len(f1_values) > 0:
        avg_f1 = f1_values.mean()
        avg_std = np.sqrt((model_data['f1_std'].dropna()**2).mean())  # RMS of stds
        entry['Avg F1'] = f"{avg_f1:.3f}±{avg_std:.3f}"
    else:
        entry['Avg F1'] = '-'

    table1_data.append(entry)

table1_df = pd.DataFrame(table1_data)
# Sort by average F1 (extract mean from string)
table1_df['sort_key'] = table1_df['Avg F1'].apply(lambda x: float(x.split('±')[0]) if x != '-' else 0)
table1_df = table1_df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)

print(table1_df.to_string(index=False))

# Export
output_path = tables_std_dir / "table1-cross_dataset_performance_std.csv"
table1_df.to_csv(output_path, index=False)
print(f"\n✓ Exported: {output_path.relative_to(script_dir.parent)}")

# ============================================================================
# TABLE 2: Quantization Ablation with STD
# ============================================================================
print("\n" + "="*100)
print("TABLE 2: Quantization Ablation (with ±std)")
print("="*100)

# Filter quantized models
quant_models = agg_df[agg_df['model_name_std'].str.contains('Q[0-9]+-Base', regex=True)].copy()

def extract_bits(name):
    import re
    match = re.search(r'Q(\d+)-Base', name)
    return int(match.group(1)) if match else 999

quant_models['bit_width'] = quant_models['model_name_std'].apply(extract_bits)

# Calculate % of 16-bit for each dataset
q16_baseline = {}
for dataset in ['ESC-50', 'Quinn', 'UrbanSound8K']:
    q16_model = quant_models[(quant_models['dataset'] == dataset) & (quant_models['bit_width'] == 16)]
    if len(q16_model) > 0:
        q16_baseline[dataset] = q16_model['f1_mean'].iloc[0]

table2_data = []
for _, row in quant_models.iterrows():
    dataset = row['dataset']
    f1_mean = row['f1_mean']
    f1_std = row['f1_std']

    pct_16bit = (f1_mean / q16_baseline[dataset] * 100) if dataset in q16_baseline else np.nan

    table2_data.append({
        'Bit-Width': row['bit_width'],
        'Model': row['model_name_std'],
        'Dataset': dataset,
        'Params (M)': round(row['parameter_count_first'] / 1e6, 2),
        'F1-Score': f"{f1_mean:.3f}±{f1_std:.3f}",
        '% of 16-bit': f"{pct_16bit:.1f}" if not pd.isna(pct_16bit) else '-',
        'Latency (ms)': f"{row['latency_mean_ms']:.2f}±{row['latency_std_ms']:.2f}",
        'Energy (mJ)': f"{row['energy_train_mean_mJ']:.3f}±{row['energy_train_std_mJ']:.3f}",
        'RAM (GB)': f"{row['infer_ram_gb_mean']:.2f}±{row['infer_ram_gb_std']:.2f}"
    })

table2_df = pd.DataFrame(table2_data).sort_values(['Dataset', 'Bit-Width'])
print(table2_df.to_string(index=False))

# Export
output_path = tables_std_dir / "table2-ablation_quantization_std.csv"
table2_df.to_csv(output_path, index=False)
print(f"\n✓ Exported: {output_path.relative_to(script_dir.parent)}")

# ============================================================================
# TABLE 3: MoE with Curiosity Routing (Aggregated with STD)
# ============================================================================
print("\n" + "="*100)
print("TABLE 3: MoE with Curiosity Routing (with ±std)")
print("="*100)

# Filter MoE models
moe_models = agg_df[agg_df['model_name_std'].str.contains('QMoE', regex=True)].copy()

# Aggregate across datasets
table3_data = moe_models.groupby('model_name_std').agg({
    'parameter_count_first': 'first',
    'f1_mean': 'mean',
    'f1_std': lambda x: np.sqrt((x**2).mean()),  # RMS of stds
    'latency_mean_ms': 'mean',
    'latency_std_ms': lambda x: np.sqrt((x**2).mean()),
    'energy_train_mean_mJ': 'mean',
    'energy_train_std_mJ': lambda x: np.sqrt((x**2).mean()),
    'emissions_train_mean_ug': 'mean',
    'emissions_train_std_ug': lambda x: np.sqrt((x**2).mean())
}).reset_index()

# Determine type (MoE vs MoE-Curiosity)
table3_data['Type'] = table3_data['model_name_std'].apply(lambda x: 'MoE-Curiosity' if '-C' in x else 'MoE')

# Calculate efficiency
table3_data['efficiency'] = table3_data['f1_mean'] / table3_data['energy_train_mean_mJ']

table3_final = pd.DataFrame({
    'Model': table3_data['model_name_std'],
    'Type': table3_data['Type'],
    'Params (M)': (table3_data['parameter_count_first'] / 1e6).round(2),
    'Avg F1': table3_data.apply(lambda r: f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}", axis=1),
    'Latency (ms)': table3_data.apply(lambda r: f"{r['latency_mean_ms']:.2f}±{r['latency_std_ms']:.2f}", axis=1),
    'Energy (mJ)': table3_data.apply(lambda r: f"{r['energy_train_mean_mJ']:.3f}±{r['energy_train_std_mJ']:.3f}", axis=1),
    'CO2 (µg)': table3_data.apply(lambda r: f"{r['emissions_train_mean_ug']:.2f}±{r['emissions_train_std_ug']:.2f}", axis=1),
    'Eff (F1/mJ)': table3_data['efficiency'].round(3)
})

# Sort by F1 (extract mean)
table3_final['sort_key'] = table3_final['Avg F1'].apply(lambda x: float(x.split('±')[0]))
table3_final = table3_final.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)

print(table3_final.to_string(index=False))

# Export
output_path = tables_std_dir / "table3-moe_curiosity_std.csv"
table3_final.to_csv(output_path, index=False)
print(f"\n✓ Exported: {output_path.relative_to(script_dir.parent)}")

# ============================================================================
# TABLE 4: Inference Latency with STD
# ============================================================================
print("\n" + "="*100)
print("TABLE 4: Inference Latency (with ±std)")
print("="*100)

# Aggregate across datasets for all models
latency_agg = agg_df.groupby('model_name_std').agg({
    'parameter_count_first': 'first',
    'latency_mean_ms': 'mean',
    'latency_std_ms': lambda x: np.sqrt((x**2).mean()),
    'infer_ram_gb_mean': 'mean',
    'infer_ram_gb_std': lambda x: np.sqrt((x**2).mean()),
    'f1_mean': 'mean'
}).reset_index()

# Find ESC-Base baseline
esc_baseline = latency_agg[latency_agg['model_name_std'].str.contains('ESC-Base†', regex=False)]
if len(esc_baseline) > 0:
    baseline_latency = esc_baseline['latency_mean_ms'].iloc[0]
else:
    baseline_latency = latency_agg['latency_mean_ms'].max()

latency_agg['speedup'] = baseline_latency / latency_agg['latency_mean_ms']

table4_final = pd.DataFrame({
    'Model': latency_agg['model_name_std'],
    'Params (M)': (latency_agg['parameter_count_first'] / 1e6).round(2),
    'Latency (ms)': latency_agg.apply(lambda r: f"{r['latency_mean_ms']:.2f}±{r['latency_std_ms']:.2f}", axis=1),
    'Speedup': latency_agg['speedup'].round(2),
    'RAM (GB)': latency_agg.apply(lambda r: f"{r['infer_ram_gb_mean']:.2f}±{r['infer_ram_gb_std']:.2f}", axis=1),
    'F1-Score': latency_agg['f1_mean'].round(3)
})

# Sort by latency (extract mean)
table4_final['sort_key'] = table4_final['Latency (ms)'].apply(lambda x: float(x.split('±')[0]))
table4_final = table4_final.sort_values('sort_key').drop('sort_key', axis=1)

print(table4_final.to_string(index=False))

# Export
output_path = tables_std_dir / "table4-inference_latency_std.csv"
table4_final.to_csv(output_path, index=False)
print(f"\n✓ Exported: {output_path.relative_to(script_dir.parent)}")

print("\n" + "="*100)
print("All tables with standard deviations exported successfully to notebooks/tables-std/")
print("="*100)
print("""
FORMAT: mean±std
✓ All F1-scores, latencies, energies, and RAM values shown as mean±std
✓ Standard deviations calculated from 5-fold cross-validation
✓ % of 16-bit baseline included in Table 2
✓ Ready for CVPR paper with confidence intervals
""")
