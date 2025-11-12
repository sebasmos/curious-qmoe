import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
data_dir = script_dir / "data"  # data folder inside notebooks
tables_dir = script_dir / "tables"

# Create directories if they don't exist
data_dir.mkdir(exist_ok=True)
tables_dir.mkdir(exist_ok=True)

# Read all three datasets from data folder
quinn_df = pd.read_csv(data_dir / "quinn-results.csv")
esc_df = pd.read_csv(data_dir / "esc-results.csv")
urban8_df = pd.read_csv(data_dir / "urban8-results.csv")

# Add dataset column to each
quinn_df['dataset'] = 'Quinn'
esc_df['dataset'] = 'ESC-50'
urban8_df['dataset'] = 'UrbanSound8K'

# Combine all datasets
all_data = pd.concat([quinn_df, esc_df, urban8_df], ignore_index=True)

# ============================================================================
# MODEL NOMENCLATURE MAPPING
# ============================================================================
def standardize_model_name(name):
    """
    Standardize model names to proper nomenclature:
    - ESC-Base† (FP32 baseline)
    - QESC-Base‡ (PTQ INT8 baseline)
    - Q2/Q4/Q8/Q16-Base (quantization levels)
    - BitNet-Q8-MoE, BitNet-Q4/8-QMoE, BitNet-QMoE (MoE variants)
    """
    name_lower = name.lower()

    # Individual/baseline models with quantization
    if 'individual_models' in name_lower or 'models_all_final' in name_lower:
        if 'bitnet' in name_lower:
            return 'BitNet-Base'
        elif name_lower.endswith('_esc'):
            return 'ESC-Base† (FP32)'  # Primary baseline
        elif name_lower.endswith('_qesc'):
            return 'QESC-Base‡ (INT8)'  # Secondary baseline
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
        else:
            return 'Full-Precision'

    # MoE models
    elif 'qmoe' in name_lower or 'moe' in name_lower:
        has_curiosity = 'curiosity' in name_lower

        if '4_8_16' in name_lower:
            return 'BitNet-Q4/8/16-QMoE' + ('-C' if has_curiosity else '')
        elif '4_8' in name_lower:
            return 'BitNet-Q4/8-QMoE' + ('-C' if has_curiosity else '')
        elif '8_16' in name_lower:
            if 'qesc' in name_lower:
                return 'BitNet-Q8/16-QESC-QMoE' + ('-C' if has_curiosity else '')
            return 'BitNet-Q8/16-QMoE' + ('-C' if has_curiosity else '')
        elif 'qesc' in name_lower and 'bitnet' in name_lower:
            return 'BitNet-QESC-QMoE' + ('-C' if has_curiosity else '')
        else:
            return 'BitNet-QMoE' + ('-C' if has_curiosity else '')

    return 'Other'

def categorize_model_type(name):
    """Categorize models into broad types for analysis"""
    std_name = standardize_model_name(name)

    if 'QMoE' in std_name:
        if '-C' in std_name:
            return 'MoE-Curiosity'
        return 'MoE'
    elif std_name.startswith('Q') and 'Base' in std_name:
        return 'Quantized-Base'
    elif 'Base' in std_name:
        return 'Baseline'
    else:
        return 'Other'

all_data['model_name_std'] = all_data['name'].apply(standardize_model_name)
all_data['model_category'] = all_data['name'].apply(categorize_model_type)

# Convert units for better readability
all_data['Energy-train (mJ)'] = all_data['Energy-train'] * 1000  # J to mJ
all_data['Energy-val (mJ)'] = all_data['Energy-val'] * 1000
all_data['Emissions-train (µg)'] = all_data['Emissions-train'] * 1e6  # g to µg
all_data['Emissions-val (µg)'] = all_data['Emissions-val'] * 1e6
all_data['Latency (ms)'] = all_data['Latency (s)'] * 1000  # s to ms

print("="*100)
print("CVPR PAPER - RESULTS ANALYSIS")
print("="*100)

# ============================================================================
# TABLE 1: Cross-Dataset Performance (MAIN RESULT)
# ============================================================================
print("\n" + "="*100)
print("TABLE 1: Cross-Dataset Performance")
print("="*100)

# Get models that appear across multiple datasets
table1_data = all_data.groupby('model_name_std').agg({
    'dataset': lambda x: list(x),
    'F1 (Mean)': 'mean',
    'Param. Count': 'first',
    'Latency (ms)': 'mean',
    'Energy-train (mJ)': 'mean'
}).reset_index()

# Only include models tested on multiple datasets
table1_data['num_datasets'] = table1_data['dataset'].apply(len)
table1_data = table1_data[table1_data['num_datasets'] >= 2].copy()

# Create detailed per-dataset F1 columns
table1_detailed = []
for model in table1_data['model_name_std'].unique():
    model_data = all_data[all_data['model_name_std'] == model]

    entry = {
        'Model': model,
        'Params (M)': round(model_data['Param. Count'].iloc[0] / 1e6, 2),
    }

    for dataset in ['ESC-50', 'Quinn', 'UrbanSound8K']:
        dataset_model = model_data[model_data['dataset'] == dataset]
        if len(dataset_model) > 0:
            entry[f'{dataset} F1'] = round(dataset_model['F1 (Mean)'].iloc[0], 3)
        else:
            entry[f'{dataset} F1'] = np.nan

    # Calculate average across available datasets
    f1_cols = [entry[f'{d} F1'] for d in ['ESC-50', 'Quinn', 'UrbanSound8K'] if not pd.isna(entry.get(f'{d} F1'))]
    entry['Avg F1'] = round(np.mean(f1_cols), 3) if f1_cols else np.nan

    table1_detailed.append(entry)

table1_df = pd.DataFrame(table1_detailed).sort_values('Avg F1', ascending=False)
print(table1_df.to_string(index=False))

# Export
output_path = tables_dir / "table1-cross_dataset_performance.csv"
table1_df.to_csv(output_path, index=False)
print(f"\n✓ Exported: {output_path.relative_to(script_dir.parent)}")
print("Description: Cross-dataset generalization - F1-scores across ESC-50, Quinn, and UrbanSound8K")

# ============================================================================
# TABLE 2: Ablation Study - Quantization Bit-Width Impact
# ============================================================================
print("\n" + "="*100)
print("TABLE 2: Ablation Study - Quantization Bit-Width Impact")
print("="*100)

# Focus on quantized base models across bit-widths
quant_models = all_data[all_data['model_name_std'].str.contains('Q[0-9]+-Base', regex=True)].copy()

table2_data = quant_models.groupby(['model_name_std', 'dataset']).agg({
    'F1 (Mean)': 'mean',
    'Latency (ms)': 'mean',
    'Energy-train (mJ)': 'mean',
    'Param. Count': 'first',
    'Infer. RAM (GB)': 'mean'
}).reset_index()

# Extract bit-width for sorting
def extract_bits(name):
    import re
    match = re.search(r'Q(\d+)-Base', name)
    return int(match.group(1)) if match else 999

table2_data['Bit-Width'] = table2_data['model_name_std'].apply(extract_bits)

# Calculate percentage of 16-bit performance for each dataset
q16_baseline = {}
for dataset in ['ESC-50', 'Quinn', 'UrbanSound8K']:
    q16_model = table2_data[(table2_data['dataset'] == dataset) & (table2_data['Bit-Width'] == 16)]
    if len(q16_model) > 0:
        q16_baseline[dataset] = q16_model['F1 (Mean)'].iloc[0]
    else:
        q16_baseline[dataset] = None

# Add % of 16-bit column
table2_data['% of 16-bit'] = table2_data.apply(
    lambda row: round((row['F1 (Mean)'] / q16_baseline[row['dataset']]) * 100, 1)
    if q16_baseline[row['dataset']] else np.nan,
    axis=1
)

table2_final = pd.DataFrame({
    'Bit-Width': table2_data['Bit-Width'],
    'Model': table2_data['model_name_std'],
    'Dataset': table2_data['dataset'],
    'Params (M)': (table2_data['Param. Count'] / 1e6).round(2),
    'F1-Score': table2_data['F1 (Mean)'].round(3),
    '% of 16-bit': table2_data['% of 16-bit'],
    'Latency (ms)': table2_data['Latency (ms)'].round(2),
    'Energy (mJ)': table2_data['Energy-train (mJ)'].round(3),
    'RAM (GB)': table2_data['Infer. RAM (GB)'].round(2)
})

# Sort by bit-width and dataset
table2_final = table2_final.sort_values(['Dataset', 'Bit-Width'])

print(table2_final.to_string(index=False))

# Export
output_path = tables_dir / "table2-ablation_quantization_impact.csv"
table2_final.to_csv(output_path, index=False)
print(f"\n✓ Exported: {output_path.relative_to(script_dir.parent)}")
print("Description: Quantization bit-width (1, 2, 4, 8, 16-bit) impact on performance and efficiency")

# Summary statistics
print("\n--- Quantization Bit-Width Summary ---")
if len(table2_final) > 0:
    bit_summary = table2_final.groupby('Bit-Width').agg({
        'F1-Score': 'mean',
        'Latency (ms)': 'mean',
        'Energy (mJ)': 'mean',
        'RAM (GB)': 'mean'
    }).round(3)
    print(bit_summary.to_string())

# ============================================================================
# TABLE 3: Mixture of Experts with Curiosity-Driven Routing
# ============================================================================
print("\n" + "="*100)
print("TABLE 3: Mixture of Experts with Curiosity-Driven Routing")
print("="*100)

# Filter MoE models
moe_models = all_data[all_data['model_category'].isin(['MoE', 'MoE-Curiosity'])].copy()

# Aggregate across datasets to get ~7 rows
table3_data = moe_models.groupby('model_name_std').agg({
    'F1 (Mean)': 'mean',
    'Latency (ms)': 'mean',
    'Energy-train (mJ)': 'mean',
    'Emissions-train (µg)': 'mean',
    'Param. Count': 'first',
    'model_category': 'first'
}).reset_index()

# Calculate efficiency metrics
table3_data['Energy Eff (F1/mJ)'] = table3_data['F1 (Mean)'] / table3_data['Energy-train (mJ)']

table3_final = pd.DataFrame({
    'Model': table3_data['model_name_std'],
    'Type': table3_data['model_category'],
    'Params (M)': (table3_data['Param. Count'] / 1e6).round(2),
    'Avg F1': table3_data['F1 (Mean)'].round(3),
    'Latency (ms)': table3_data['Latency (ms)'].round(2),
    'Energy (mJ)': table3_data['Energy-train (mJ)'].round(3),
    'CO2 (µg)': table3_data['Emissions-train (µg)'].round(2),
    'Eff (F1/mJ)': table3_data['Energy Eff (F1/mJ)'].round(3)
})

# Sort by F1 descending
table3_final = table3_final.sort_values('Avg F1', ascending=False)

print(table3_final.to_string(index=False))

# Export
output_path = tables_dir / "table3-moe_curiosity_routing.csv"
table3_final.to_csv(output_path, index=False)
print(f"\n✓ Exported: {output_path.relative_to(script_dir.parent)}")
print("Description: MoE architectures with energy metrics (C = Curiosity-driven routing)")

# Summarize curiosity impact
print("\n--- Curiosity-Driven Routing Impact ---")
curiosity_comparison = table3_final.groupby('Type').agg({
    'Avg F1': 'mean',
    'Energy (mJ)': 'mean',
    'Latency (ms)': 'mean',
    'Eff (F1/mJ)': 'mean'
}).round(3)
print(curiosity_comparison.to_string())

# ============================================================================
# TABLE 4: Inference Latency for Edge Deployment
# ============================================================================
print("\n" + "="*100)
print("TABLE 4: Inference Latency for Edge Deployment")
print("="*100)

table4_data = all_data.groupby('model_name_std').agg({
    'Latency (ms)': 'mean',
    'F1 (Mean)': 'mean',
    'Param. Count': 'first',
    'Infer. RAM (GB)': 'mean'
}).reset_index()

# Find ESC-Base (FP32) as baseline for speedup calculation
esc_baseline = table4_data[table4_data['model_name_std'].str.contains('ESC-Base†', regex=False)]
if len(esc_baseline) > 0:
    baseline_latency = esc_baseline['Latency (ms)'].iloc[0]
else:
    # Fallback to slowest model
    baseline_latency = table4_data['Latency (ms)'].max()

table4_data['Speedup vs ESC-Base†'] = baseline_latency / table4_data['Latency (ms)']

table4_final = pd.DataFrame({
    'Model': table4_data['model_name_std'],
    'Params (M)': (table4_data['Param. Count'] / 1e6).round(2),
    'Latency (ms)': table4_data['Latency (ms)'].round(2),
    'Speedup': table4_data['Speedup vs ESC-Base†'].round(2),
    'RAM (GB)': table4_data['Infer. RAM (GB)'].round(2),
    'F1-Score': table4_data['F1 (Mean)'].round(3)
})

# Sort by latency
table4_final = table4_final.sort_values('Latency (ms)')

print(table4_final.to_string(index=False))

# Export
output_path = tables_dir / "table4-inference_latency.csv"
table4_final.to_csv(output_path, index=False)
print(f"\n✓ Exported: {output_path.relative_to(script_dir.parent)}")
print("Description: Inference latency (ms), speedup vs ESC-Base† (FP32), and memory for edge deployment")

# ============================================================================
# SUPPLEMENTARY TABLE: Carbon Emissions Analysis
# ============================================================================
print("\n" + "="*100)
print("SUPPLEMENTARY TABLE: Carbon Emissions Analysis")
print("="*100)

table_supp_data = all_data.groupby('model_name_std').agg({
    'Emissions-train (µg)': 'mean',
    'Emissions-val (µg)': 'mean',
    'Train. Runtime (s)': 'mean',
    'F1 (Mean)': 'mean',
    'Param. Count': 'first'
}).reset_index()

# Calculate total emissions
table_supp_data['Total CO2 (µg)'] = table_supp_data['Emissions-train (µg)'] + table_supp_data['Emissions-val (µg)']

# Calculate emissions per sample (assuming validation set size)
# Emission rate per second of training
table_supp_data['CO2 Rate (µg/s)'] = table_supp_data['Emissions-train (µg)'] / table_supp_data['Train. Runtime (s)']

table_supp_final = pd.DataFrame({
    'Model': table_supp_data['model_name_std'],
    'Params (M)': (table_supp_data['Param. Count'] / 1e6).round(2),
    'Train CO2 (µg)': table_supp_data['Emissions-train (µg)'].round(2),
    'Val CO2 (µg)': table_supp_data['Emissions-val (µg)'].round(2),
    'Total CO2 (µg)': table_supp_data['Total CO2 (µg)'].round(2),
    'CO2 Rate (µg/s)': table_supp_data['CO2 Rate (µg/s)'].round(3),
    'F1-Score': table_supp_data['F1 (Mean)'].round(3)
})

# Sort by total emissions (lowest first)
table_supp_final = table_supp_final.sort_values('Total CO2 (µg)')

print(table_supp_final.to_string(index=False))

# Export
output_path = tables_dir / "supplementary-carbon_emissions.csv"
table_supp_final.to_csv(output_path, index=False)
print(f"\n✓ Exported: {output_path.relative_to(script_dir.parent)}")
print("Description: [SUPPLEMENTARY] Detailed CO2 emissions in micrograms (µg) during training and validation")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*100)
print("SUMMARY - KEY FINDINGS")
print("="*100)

print("\n1. Best Overall Models (by F1-Score):")
for dataset in ['ESC-50', 'Quinn', 'UrbanSound8K']:
    best = all_data[all_data['dataset'] == dataset].nlargest(1, 'F1 (Mean)').iloc[0]
    print(f"   {dataset:15s}: {best['model_name_std']:30s} (F1: {best['F1 (Mean)']:.3f})")

print("\n2. Most Energy-Efficient Models (Top 3 by F1/mJ):")
all_data['energy_eff'] = all_data['F1 (Mean)'] / all_data['Energy-train (mJ)']
top_efficient = all_data.nlargest(3, 'energy_eff')
for _, row in top_efficient.iterrows():
    print(f"   {row['model_name_std']:30s} on {row['dataset']:15s} (Eff: {row['energy_eff']:.3f} F1/mJ)")

print("\n3. Fastest Inference Models (Top 3):")
top_fast = all_data.nsmallest(3, 'Latency (ms)')
for _, row in top_fast.iterrows():
    print(f"   {row['model_name_std']:30s} on {row['dataset']:15s} (Latency: {row['Latency (ms)']:.2f} ms)")

print("\n4. Lowest Carbon Footprint (Top 3):")
all_data['total_co2'] = all_data['Emissions-train (µg)'] + all_data['Emissions-val (µg)']
top_green = all_data.nsmallest(3, 'total_co2')
for _, row in top_green.iterrows():
    print(f"   {row['model_name_std']:30s} on {row['dataset']:15s} (CO2: {row['total_co2']:.2f} µg)")

print("\n" + "="*100)
print("TABLE ORGANIZATION FOR CVPR PAPER")
print("="*100)
print("""
MAIN PAPER (4 Tables):
  Table 1: Cross-Dataset Performance - Robustness across ESC-50, Quinn, UrbanSound8K
  Table 2: Quantization Ablation - Bit-width impact with % of 16-bit baseline
  Table 3: MoE with Curiosity Routing - Aggregated view (~7 rows) with energy metrics
  Table 4: Inference Latency - Speed benchmarks (Speedup vs ESC-Base† FP32)

SUPPLEMENTARY MATERIAL:
  Supplementary: Carbon Emissions - Detailed CO2 analysis

BASELINES:
† ESC-Base† (FP32) - Primary baseline (unquantized full precision)
‡ QESC-Base‡ (INT8) - Secondary baseline (standard PTQ)

FORMATTING:
✓ All metrics use 3 decimal places maximum (CVPR standard)
✓ Only F1-Score reported (more informative for imbalanced datasets)
✓ No scientific notation - proper units:
  • Energy: millijoules (mJ)
  • Emissions: micrograms (µg)
  • Latency: milliseconds (ms)
✓ Parameters rounded to 2 decimals
✓ Curiosity models marked with '-C' suffix
✓ Speedup calculated relative to ESC-Base† (FP32)
✓ All tables ready for LaTeX import

KEY IMPROVEMENTS:
✓ Table 3 now aggregated (~7 rows instead of 21)
✓ Energy metrics merged into Table 3 (MoE table)
✓ Table 2 includes '% of 16-bit' column
✓ Clear baseline references with † and ‡ symbols
""")

print("\n" + "="*100)
print("All tables exported successfully to notebooks/tables/")
print("="*100)
