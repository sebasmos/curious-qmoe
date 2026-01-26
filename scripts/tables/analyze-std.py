"""
Analyze results across all datasets
"""


import pandas as pd
import numpy as np
import json
from pathlib import Path

script_dir = Path(__file__).parent
# results_base = Path(r"/Users/cajas.sebastian/Desktop/repositories/curious-qmoe/RESULTS")
results_base = Path(r"/Users/cajas.sebastian/Desktop/repositories/curious-qmoe/RESULTS")

tables_std_dir = script_dir / "tables-std"
tables_std_dir.mkdir(exist_ok=True)

def collect_fold_metrics(results_base):
    all_fold_data = []
    for dataset_dir in results_base.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name.replace(" Results", "")
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            subdirs = list(model_dir.iterdir())
            variant_dirs = [d for d in subdirs if d.is_dir() and d.name in ['moe', 'qmoe', '1', '2', '4', '8', '16', 'bitnet', 'esc', 'qesc']]
            if variant_dirs:
                for variant_dir in variant_dirs:
                    variant_name = f"{model_name}_{variant_dir.name}"
                    fold_data = collect_folds_from_dir(variant_dir, dataset_name, variant_name)
                    all_fold_data.extend(fold_data)
            else:
                fold_data = collect_folds_from_dir(model_dir, dataset_name, model_name)
                all_fold_data.extend(fold_data)
    return pd.DataFrame(all_fold_data)

def collect_folds_from_dir(directory, dataset_name, model_name):
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
                'latency': metrics.get('val_duration', np.nan) / 5 if metrics.get('val_duration') else np.nan,
                'infer_ram_gb': metrics.get('max_ram_mb_val', np.nan) / 1024,
                'energy_train': metrics.get('train_energy_consumed', np.nan),
                'energy_val': metrics.get('val_energy_consumed', np.nan),
                'emissions_train': metrics.get('train_emissions', np.nan),
                'emissions_val': metrics.get('val_emissions', np.nan)
            })
        except:
            pass
    return fold_data

fold_df = collect_fold_metrics(results_base)

def aggregate_with_std(fold_df):
    agg_data = fold_df.groupby(['dataset', 'model_name']).agg({
        'f1': ['mean', 'std', 'count'],
        'accuracy': ['mean', 'std'],
        'parameter_count': ['first'],
        'latency': ['mean', 'std'],
        'infer_ram_gb': ['mean', 'std'],
        'energy_train': ['mean', 'std'],
        'energy_val': ['mean', 'std'],
        'emissions_train': ['mean', 'std'],
        'emissions_val': ['mean', 'std'],
        'train_duration': ['mean', 'std']
    }).reset_index()
    agg_data.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in agg_data.columns.values]
    return agg_data

agg_df = aggregate_with_std(fold_df)
agg_df['latency_mean_ms'] = agg_df['latency_mean'] * 1000
agg_df['latency_std_ms'] = agg_df['latency_std'] * 1000
agg_df['energy_train_mean_mJ'] = agg_df['energy_train_mean'] * 1000
agg_df['energy_train_std_mJ'] = agg_df['energy_train_std'] * 1000
agg_df['emissions_train_mean_ug'] = agg_df['emissions_train_mean'] * 1e6
agg_df['emissions_train_std_ug'] = agg_df['emissions_train_std'] * 1e6

def standardize_model_name(name):
    name_lower = name.lower()
    if 'models_all_final' in name_lower or 'individual_models' in name_lower:
        if '_bitnet' in name_lower:
            return 'BitNet-Base'
        elif '_esc' in name_lower and '_qesc' not in name_lower:
            return 'FP32-Base'
        elif '_qesc' in name_lower:
            return 'Q8-Base-PTQ'
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
    if 'qmoe' in name_lower or 'moe' in name_lower:
        has_curiosity = 'curiosity' in name_lower
        suffix = '-C' if has_curiosity else ''
        if '4_8_16' in name_lower:
            return f'BitNet-Q4/8/16-QMoE{suffix}'
        elif '4_8' in name_lower:
            return f'BitNet-Q4/8-QMoE{suffix}'
        elif '8_16' in name_lower:
            if 'qesc' in name_lower:
                return f'BitNet-Q8/16-PTQ-QMoE{suffix}'
            return f'BitNet-Q8/16-QMoE{suffix}'
        elif 'qesc' in name_lower and 'bitnet' in name_lower:
            return f'BitNet-Q8PTQ-QMoE{suffix}'
        else:
            return f'BitNet-QMoE{suffix}'
    return name

agg_df['model_name_std'] = agg_df['model_name'].apply(standardize_model_name)
dataset_map = {'Quinn': 'Quinn', 'ESC': 'ESC-50', 'Urban8': 'UrbanSound8K'}
agg_df['dataset'] = agg_df['dataset'].map(dataset_map).fillna(agg_df['dataset'])

multi_dataset_models = agg_df.groupby('model_name_std')['dataset'].apply(lambda x: len(set(x))).reset_index()
multi_dataset_models = multi_dataset_models[multi_dataset_models['dataset'] >= 2]['model_name_std'].tolist()

table1_data = []
for model in multi_dataset_models:
    model_data = agg_df[agg_df['model_name_std'] == model]
    entry = {'Model': model, 'Params (M)': round(model_data['parameter_count_first'].iloc[0] / 1e6, 2)}
    for dataset in ['ESC-50', 'Quinn', 'UrbanSound8K']:
        dataset_model = model_data[model_data['dataset'] == dataset]
        if len(dataset_model) > 0:
            f1_mean = dataset_model['f1_mean'].iloc[0]
            f1_std = dataset_model['f1_std'].iloc[0]
            entry[f'{dataset} F1'] = f"{f1_mean:.3f}±{f1_std:.3f}"
        else:
            entry[f'{dataset} F1'] = '-'
    f1_values = model_data['f1_mean'].dropna()
    if len(f1_values) > 0:
        avg_f1 = f1_values.mean()
        avg_std = np.sqrt((model_data['f1_std'].dropna()**2).mean())
        entry['Avg F1'] = f"{avg_f1:.3f}±{avg_std:.3f}"
    else:
        entry['Avg F1'] = '-'
    table1_data.append(entry)

table1_df = pd.DataFrame(table1_data)
table1_df['sort_key'] = table1_df['Avg F1'].apply(lambda x: float(x.split('±')[0]) if x != '-' else 0)
table1_df = table1_df.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
table1_df.to_csv(tables_std_dir / "table1-cross_dataset_performance_std.csv", index=False)

quant_models = agg_df[agg_df['model_name_std'].str.contains('Q[0-9]+-Base', regex=True)].copy()

def extract_bits(name):
    import re
    match = re.search(r'Q(\d+)-Base', name)
    return int(match.group(1)) if match else 999

quant_models['bit_width'] = quant_models['model_name_std'].apply(extract_bits)
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
table2_df.to_csv(tables_std_dir / "table2-ablation_quantization_std.csv", index=False)

moe_models = agg_df[agg_df['model_name_std'].str.contains('QMoE', regex=True)].copy()
table3_data = moe_models.groupby('model_name_std').agg({
    'parameter_count_first': 'first',
    'f1_mean': 'mean',
    'f1_std': lambda x: np.sqrt((x**2).mean()),
    'latency_mean_ms': 'mean',
    'latency_std_ms': lambda x: np.sqrt((x**2).mean()),
    'energy_train_mean_mJ': 'mean',
    'energy_train_std_mJ': lambda x: np.sqrt((x**2).mean()),
    'emissions_train_mean_ug': 'mean',
    'emissions_train_std_ug': lambda x: np.sqrt((x**2).mean())
}).reset_index()

table3_data['Type'] = table3_data['model_name_std'].apply(lambda x: 'MoE-Curiosity' if '-C' in x else 'MoE')
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

table3_final['sort_key'] = table3_final['Avg F1'].apply(lambda x: float(x.split('±')[0]))
table3_final = table3_final.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
table3_final.to_csv(tables_std_dir / "table3-moe_curiosity_std.csv", index=False)

latency_agg = agg_df.groupby('model_name_std').agg({
    'parameter_count_first': 'first',
    'latency_mean_ms': 'mean',
    'latency_std_ms': lambda x: np.sqrt((x**2).mean()),
    'infer_ram_gb_mean': 'mean',
    'infer_ram_gb_std': lambda x: np.sqrt((x**2).mean()),
    'f1_mean': 'mean'
}).reset_index()

fp32_baseline = latency_agg[latency_agg['model_name_std'].str.contains('FP32-Base', regex=False)]
baseline_latency = fp32_baseline['latency_mean_ms'].iloc[0] if len(fp32_baseline) > 0 else latency_agg['latency_mean_ms'].max()
latency_agg['speedup'] = baseline_latency / latency_agg['latency_mean_ms']

table4_final = pd.DataFrame({
    'Model': latency_agg['model_name_std'],
    'Params (M)': (latency_agg['parameter_count_first'] / 1e6).round(2),
    'Latency (ms)': latency_agg.apply(lambda r: f"{r['latency_mean_ms']:.2f}±{r['latency_std_ms']:.2f}", axis=1),
    'Speedup': latency_agg['speedup'].round(2),
    'RAM (GB)': latency_agg.apply(lambda r: f"{r['infer_ram_gb_mean']:.2f}±{r['infer_ram_gb_std']:.2f}", axis=1),
    'F1-Score': latency_agg['f1_mean'].round(3)
})

table4_final['sort_key'] = table4_final['Latency (ms)'].apply(lambda x: float(x.split('±')[0]))
table4_final = table4_final.sort_values('sort_key').drop('sort_key', axis=1)
table4_final.to_csv(tables_std_dir / "table4-inference_latency_std.csv", index=False)

emissions_agg = agg_df.groupby('model_name_std').agg({
    'parameter_count_first': 'first',
    'emissions_train_mean_ug': 'mean',
    'emissions_train_std_ug': lambda x: np.sqrt((x**2).mean()),
    'emissions_val_mean': 'mean',
    'emissions_val_std': lambda x: np.sqrt((x**2).mean()),
    'train_duration_mean': 'mean',
    'train_duration_std': lambda x: np.sqrt((x**2).mean()),
    'f1_mean': 'mean'
}).reset_index()

emissions_agg['emissions_val_mean_ug'] = emissions_agg['emissions_val_mean'] * 1e6
emissions_agg['emissions_val_std_ug'] = emissions_agg['emissions_val_std'] * 1e6
emissions_agg['total_co2_mean'] = emissions_agg['emissions_train_mean_ug'] + emissions_agg['emissions_val_mean_ug']
emissions_agg['total_co2_std'] = np.sqrt(emissions_agg['emissions_train_std_ug']**2 + emissions_agg['emissions_val_std_ug']**2)
emissions_agg['co2_rate_mean'] = emissions_agg['emissions_train_mean_ug'] / emissions_agg['train_duration_mean']
emissions_agg['co2_rate_std'] = emissions_agg['co2_rate_mean'] * np.sqrt(
    (emissions_agg['emissions_train_std_ug'] / emissions_agg['emissions_train_mean_ug'])**2 +
    (emissions_agg['train_duration_std'] / emissions_agg['train_duration_mean'])**2
)

table_supp_final = pd.DataFrame({
    'Model': emissions_agg['model_name_std'],
    'Params (M)': (emissions_agg['parameter_count_first'] / 1e6).round(2),
    'Train CO2 (µg)': emissions_agg.apply(lambda r: f"{r['emissions_train_mean_ug']:.2f}±{r['emissions_train_std_ug']:.2f}", axis=1),
    'Val CO2 (µg)': emissions_agg.apply(lambda r: f"{r['emissions_val_mean_ug']:.2f}±{r['emissions_val_std_ug']:.2f}", axis=1),
    'Total CO2 (µg)': emissions_agg.apply(lambda r: f"{r['total_co2_mean']:.2f}±{r['total_co2_std']:.2f}", axis=1),
    'CO2 Rate (µg/s)': emissions_agg.apply(lambda r: f"{r['co2_rate_mean']:.3f}±{r['co2_rate_std']:.3f}", axis=1),
    'F1-Score': emissions_agg['f1_mean'].round(3)
})

table_supp_final['sort_key'] = table_supp_final['Total CO2 (µg)'].apply(lambda x: float(x.split('±')[0]))
table_supp_final = table_supp_final.sort_values('sort_key').drop('sort_key', axis=1)
table_supp_final.to_csv(tables_std_dir / "supplementary-carbon_emissions_std.csv", index=False)
