import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats

script_dir = Path(__file__).parent
results_dir = Path(r"C:\Users\sebastian.cajasordon\Downloads\RESULTS-paper")
output_dir = script_dir / "significance-tests"
output_dir.mkdir(exist_ok=True)

def collect_folds_from_dir(directory, dataset_name, model_name):
    fold_data = []
    for fold_dir in directory.iterdir():
        if not fold_dir.is_dir() or not fold_dir.name.startswith('fold_'):
            continue
        metrics_file = fold_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            fold_data.append({
                'dataset': dataset_name,
                'model_name': model_name,
                'fold': fold_dir.name,
                'f1': metrics.get('best_f1', np.nan),
                'latency': metrics.get('val_duration', np.nan) / 5,
                'energy_train': metrics.get('train_energy_consumed', np.nan),
                'latency_variance': metrics.get('val_duration_std', np.nan) if 'val_duration_std' in metrics else np.nan
            })
    return fold_data

all_folds = []

for dataset_name, dataset_folder in [('ESC-50', 'ESC Results'), ('Quinn', 'Quinn Results'), ('UrbanSound8K', 'Urban8 Results')]:
    dataset_path = results_dir / dataset_folder
    if not dataset_path.exists():
        continue
    for model_dir in dataset_path.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if (model_dir / 'qmoe').exists():
            qmoe_dir = model_dir / 'qmoe'
            if list(qmoe_dir.glob('fold_*')):
                all_folds.extend(collect_folds_from_dir(qmoe_dir, dataset_name, model_name + '/qmoe'))
        elif (model_dir / 'moe').exists():
            moe_dir = model_dir / 'moe'
            if list(moe_dir.glob('fold_*')):
                all_folds.extend(collect_folds_from_dir(moe_dir, dataset_name, model_name + '/moe'))
        for bit_width in ['1', '2', '4', '8', '16']:
            bit_dir = model_dir / bit_width
            if bit_dir.exists() and list(bit_dir.glob('fold_*')):
                all_folds.extend(collect_folds_from_dir(bit_dir, dataset_name, f"{model_name}/{bit_width}"))
        for baseline in ['esc', 'qesc']:
            baseline_dir = model_dir / baseline
            if baseline_dir.exists() and list(baseline_dir.glob('fold_*')):
                all_folds.extend(collect_folds_from_dir(baseline_dir, dataset_name, f"{model_name}/{baseline}"))
        if list(model_dir.glob('fold_*')):
            all_folds.extend(collect_folds_from_dir(model_dir, dataset_name, model_name))

fold_df = pd.DataFrame(all_folds)

def standardize_model_name(name):
    name_lower = name.lower()
    if 'models_all_final' in name_lower or 'individual_models' in name_lower:
        if name_lower.endswith('/1') or name_lower.endswith('\\1'):
            return 'Q1-Base'
        elif name_lower.endswith('/2') or name_lower.endswith('\\2'):
            return 'Q2-Base'
        elif name_lower.endswith('/4') or name_lower.endswith('\\4'):
            return 'Q4-Base'
        elif name_lower.endswith('/8') or name_lower.endswith('\\8'):
            return 'Q8-Base'
        elif name_lower.endswith('/16') or name_lower.endswith('\\16'):
            return 'Q16-Base'
        elif name_lower.endswith('/esc') or name_lower.endswith('\\esc'):
            return 'FP32-Base'
        elif name_lower.endswith('/qesc') or name_lower.endswith('\\qesc'):
            return 'Q8-Base-PTQ'
        elif '_esc' in name_lower and '_qesc' not in name_lower:
            return 'FP32-Base'
        elif '_qesc' in name_lower:
            return 'Q8-Base-PTQ'
        elif '_bitnet' in name_lower:
            return 'BitNet-Base'
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
        elif 'qesc' in name_lower and ('bitnet' in name_lower or 'moe' in name_lower):
            return f'BitNet-Q8PTQ-QMoE{suffix}'
        else:
            return f'BitNet-QMoE{suffix}'
    return name

fold_df['model_name_std'] = fold_df['model_name'].apply(standardize_model_name)

def paired_ttest(group1_scores, group2_scores, alternative='two-sided'):
    if len(group1_scores) != len(group2_scores) or len(group1_scores) < 2:
        return np.nan, np.nan, 'N/A'
    t_stat, p_value = stats.ttest_rel(group1_scores, group2_scores, alternative=alternative)
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    return t_stat, p_value, sig

def format_pvalue(p):
    if pd.isna(p):
        return 'N/A'
    elif p < 0.001:
        return '<0.001***'
    elif p < 0.01:
        return f'{p:.3f}**'
    elif p < 0.05:
        return f'{p:.3f}*'
    else:
        return f'{p:.3f}'

baseline_fp32 = 'FP32-Base'
baseline_int8 = 'Q8-Base-PTQ'

multi_dataset_models = fold_df.groupby('model_name_std')['dataset'].nunique()
multi_dataset_models = multi_dataset_models[multi_dataset_models >= 2].index.tolist()

table1_significance = []
for dataset in ['ESC-50', 'Quinn', 'UrbanSound8K']:
    baseline_scores = fold_df[(fold_df['model_name_std'] == baseline_fp32) & (fold_df['dataset'] == dataset)]['f1'].values
    for model in multi_dataset_models:
        if model in [baseline_fp32, baseline_int8]:
            continue
        model_scores = fold_df[(fold_df['model_name_std'] == model) & (fold_df['dataset'] == dataset)]['f1'].values
        if len(model_scores) == 0 or len(baseline_scores) == 0:
            continue
        t_stat, p_value, sig = paired_ttest(model_scores, baseline_scores, alternative='greater')
        table1_significance.append({
            'Dataset': dataset,
            'Model': model,
            'Model F1 (mean±std)': f"{model_scores.mean():.3f}±{model_scores.std():.3f}",
            'Baseline F1 (mean±std)': f"{baseline_scores.mean():.3f}±{baseline_scores.std():.3f}",
            'Difference': f"{(model_scores.mean() - baseline_scores.mean()):.3f}",
            't-statistic': f"{t_stat:.3f}" if not pd.isna(t_stat) else 'N/A',
            'p-value': format_pvalue(p_value),
            'Significance': sig
        })

if table1_significance:
    pd.DataFrame(table1_significance).to_csv(output_dir / "table1_f1_significance.csv", index=False)

quant_models = ['Q1-Base', 'Q2-Base', 'Q4-Base', 'Q8-Base', 'Q16-Base']
table2_significance = []
for dataset in ['ESC-50', 'Quinn', 'UrbanSound8K']:
    baseline_16bit = fold_df[(fold_df['model_name_std'] == 'Q16-Base') & (fold_df['dataset'] == dataset)]['f1'].values
    if len(baseline_16bit) == 0:
        continue
    for model in quant_models:
        if model == 'Q16-Base':
            continue
        model_scores = fold_df[(fold_df['model_name_std'] == model) & (fold_df['dataset'] == dataset)]['f1'].values
        if len(model_scores) == 0:
            continue
        t_stat, p_value, sig = paired_ttest(model_scores, baseline_16bit, alternative='two-sided')
        pct_of_16bit = (model_scores.mean() / baseline_16bit.mean() * 100)
        table2_significance.append({
            'Dataset': dataset,
            'Model': model,
            'F1 (mean±std)': f"{model_scores.mean():.3f}±{model_scores.std():.3f}",
            '16-bit F1 (mean±std)': f"{baseline_16bit.mean():.3f}±{baseline_16bit.std():.3f}",
            '% of 16-bit': f"{pct_of_16bit:.1f}%",
            'Difference': f"{(model_scores.mean() - baseline_16bit.mean()):.3f}",
            't-statistic': f"{t_stat:.3f}" if not pd.isna(t_stat) else 'N/A',
            'p-value': format_pvalue(p_value),
            'Significance': sig
        })

if table2_significance:
    pd.DataFrame(table2_significance).to_csv(output_dir / "table2_f1_significance.csv", index=False)

moe_models = fold_df[fold_df['model_name_std'].str.contains('QMoE', regex=True)]['model_name_std'].unique()
table3_significance = []
for model in moe_models:
    model_folds = fold_df[fold_df['model_name_std'] == model]
    baseline_folds = fold_df[fold_df['model_name_std'] == baseline_fp32]
    common_datasets = set(model_folds['dataset'].unique()) & set(baseline_folds['dataset'].unique())
    for dataset in common_datasets:
        model_scores = model_folds[model_folds['dataset'] == dataset]['f1'].values
        baseline_scores = baseline_folds[baseline_folds['dataset'] == dataset]['f1'].values
        if len(model_scores) == 0 or len(baseline_scores) == 0:
            continue
        t_stat, p_value, sig = paired_ttest(model_scores, baseline_scores, alternative='greater')
        table3_significance.append({
            'Model': model,
            'Dataset': dataset,
            'Model F1 (mean±std)': f"{model_scores.mean():.3f}±{model_scores.std():.3f}",
            'Baseline F1 (mean±std)': f"{baseline_scores.mean():.3f}±{baseline_scores.std():.3f}",
            'Difference': f"{(model_scores.mean() - baseline_scores.mean()):.3f}",
            't-statistic': f"{t_stat:.3f}" if not pd.isna(t_stat) else 'N/A',
            'p-value': format_pvalue(p_value),
            'Significance': sig
        })

if table3_significance:
    pd.DataFrame(table3_significance).to_csv(output_dir / "table3_f1_significance.csv", index=False)

table4_significance = []
baseline_latency = fold_df[fold_df['model_name_std'] == baseline_fp32]['latency'].values
q1_latency = fold_df[fold_df['model_name_std'] == 'Q1-Base']['latency'].values

if len(q1_latency) > 0 and len(baseline_latency) > 0:
    t_stat, p_value, sig = paired_ttest(baseline_latency, q1_latency, alternative='greater')
    table4_significance.append({
        'Comparison': '1-bit fastest claim',
        'Model A': baseline_fp32,
        'Model B': 'Q1-Base',
        'A Latency (ms)': f"{baseline_latency.mean()*1000:.2f}±{baseline_latency.std()*1000:.2f}",
        'B Latency (ms)': f"{q1_latency.mean()*1000:.2f}±{q1_latency.std()*1000:.2f}",
        'Speedup': f"{baseline_latency.mean() / q1_latency.mean():.2f}×",
        't-statistic': f"{t_stat:.3f}" if not pd.isna(t_stat) else 'N/A',
        'p-value': format_pvalue(p_value),
        'Significance': sig
    })

for model in quant_models:
    model_latency = fold_df[fold_df['model_name_std'] == model]['latency'].values
    if len(model_latency) == 0 or len(baseline_latency) == 0:
        continue
    t_stat, p_value, sig = paired_ttest(baseline_latency, model_latency, alternative='greater')
    table4_significance.append({
        'Comparison': 'Speedup vs FP32 baseline',
        'Model A': baseline_fp32,
        'Model B': model,
        'A Latency (ms)': f"{baseline_latency.mean()*1000:.2f}±{baseline_latency.std()*1000:.2f}",
        'B Latency (ms)': f"{model_latency.mean()*1000:.2f}±{model_latency.std()*1000:.2f}",
        'Speedup': f"{baseline_latency.mean() / model_latency.mean():.2f}×",
        't-statistic': f"{t_stat:.3f}" if not pd.isna(t_stat) else 'N/A',
        'p-value': format_pvalue(p_value),
        'Significance': sig
    })

q8_latency = fold_df[fold_df['model_name_std'] == 'Q8-Base']['latency'].values
for moe_model in moe_models:
    moe_latency = fold_df[fold_df['model_name_std'] == moe_model]['latency'].values
    if len(moe_latency) == 0 or len(q8_latency) == 0:
        continue
    t_stat, p_value, sig = paired_ttest(moe_latency, q8_latency, alternative='greater')
    table4_significance.append({
        'Comparison': 'MoE overhead vs Q8-Base',
        'Model A': moe_model,
        'Model B': 'Q8-Base',
        'A Latency (ms)': f"{moe_latency.mean()*1000:.2f}±{moe_latency.std()*1000:.2f}",
        'B Latency (ms)': f"{q8_latency.mean()*1000:.2f}±{q8_latency.std()*1000:.2f}",
        'Speedup': f"{moe_latency.mean() / q8_latency.mean():.2f}×",
        't-statistic': f"{t_stat:.3f}" if not pd.isna(t_stat) else 'N/A',
        'p-value': format_pvalue(p_value),
        'Significance': sig
    })

if table4_significance:
    pd.DataFrame(table4_significance).to_csv(output_dir / "table4_latency_significance.csv", index=False)

table3_energy_significance = []
baseline_energy = fold_df[fold_df['model_name_std'] == baseline_fp32]['energy_train'].values

for moe_model in moe_models:
    moe_energy = fold_df[fold_df['model_name_std'] == moe_model]['energy_train'].values
    if len(moe_energy) == 0 or len(baseline_energy) == 0:
        continue
    t_stat, p_value, sig = paired_ttest(baseline_energy, moe_energy, alternative='greater')
    table3_energy_significance.append({
        'Model': moe_model,
        'Model Energy (mJ)': f"{moe_energy.mean()*1000:.3f}±{moe_energy.std()*1000:.3f}",
        'Baseline Energy (mJ)': f"{baseline_energy.mean()*1000:.3f}±{baseline_energy.std()*1000:.3f}",
        'Energy Reduction': f"{(1 - moe_energy.mean() / baseline_energy.mean()) * 100:.1f}%",
        'Ratio': f"{baseline_energy.mean() / moe_energy.mean():.2f}×",
        't-statistic': f"{t_stat:.3f}" if not pd.isna(t_stat) else 'N/A',
        'p-value': format_pvalue(p_value),
        'Significance': sig
    })

for moe_model in moe_models:
    if '-C' not in moe_model:
        continue
    base_model = moe_model.replace('-C', '')
    curiosity_energy = fold_df[fold_df['model_name_std'] == moe_model]['energy_train'].values
    base_energy = fold_df[fold_df['model_name_std'] == base_model]['energy_train'].values
    if len(curiosity_energy) == 0 or len(base_energy) == 0:
        continue
    t_stat, p_value, sig = paired_ttest(curiosity_energy, base_energy, alternative='two-sided')
    table3_energy_significance.append({
        'Model': f'{moe_model} vs {base_model}',
        'Model Energy (mJ)': f"{curiosity_energy.mean()*1000:.3f}±{curiosity_energy.std()*1000:.3f}",
        'Baseline Energy (mJ)': f"{base_energy.mean()*1000:.3f}±{base_energy.std()*1000:.3f}",
        'Energy Overhead': f"{(curiosity_energy.mean() / base_energy.mean() - 1) * 100:.1f}%",
        'Ratio': f"{curiosity_energy.mean() / base_energy.mean():.2f}×",
        't-statistic': f"{t_stat:.3f}" if not pd.isna(t_stat) else 'N/A',
        'p-value': format_pvalue(p_value),
        'Significance': sig
    })

if table3_energy_significance:
    pd.DataFrame(table3_energy_significance).to_csv(output_dir / "table3_energy_significance.csv", index=False)

variance_comparison = []
baseline_latency_std = fold_df[fold_df['model_name_std'] == baseline_fp32]['latency'].std()

for model in quant_models + list(moe_models):
    model_latencies = fold_df[fold_df['model_name_std'] == model]['latency'].values
    if len(model_latencies) < 2:
        continue
    model_std = model_latencies.std()
    variance_reduction = (1 - model_std / baseline_latency_std) * 100
    variance_ratio = baseline_latency_std / model_std
    baseline_latencies = fold_df[fold_df['model_name_std'] == baseline_fp32]['latency'].values
    levene_stat, p_value = stats.levene(baseline_latencies, model_latencies)
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    variance_comparison.append({
        'Model': model,
        'Model Latency Std (ms)': f"{model_std*1000:.3f}",
        'Baseline Latency Std (ms)': f"{baseline_latency_std*1000:.3f}",
        'Variance Reduction': f"{variance_reduction:.1f}%",
        'Variance Ratio': f"{variance_ratio:.2f}×",
        'Levene Statistic': f"{levene_stat:.3f}",
        'p-value': format_pvalue(p_value),
        'Significance': sig
    })

if variance_comparison:
    pd.DataFrame(variance_comparison).to_csv(output_dir / "latency_variance_reduction.csv", index=False)
