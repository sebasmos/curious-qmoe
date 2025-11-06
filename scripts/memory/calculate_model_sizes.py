#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch
import pandas as pd
from QWave.bitnnet import BitNetExpert
from QWave.models import ESCModel
from QWave.memory import print_size_of_model

try:
    from QWave.qmoe_layers import BitNetPopcountExpert
    HAS_QESC = True
except ImportError:
    HAS_QESC = False
    print("Warning: Could not import BitNetPopcountExpert (qesc), skipping that model")

try:
    from QWave.moe import qMoEModelBatched
    from omegaconf import OmegaConf
    HAS_MOE = True
except ImportError:
    HAS_MOE = False
    print("Warning: Could not import MoE components, skipping MoE models")


def calculate_all_model_sizes(in_dim=1536, num_classes=50, hidden_sizes=[640, 320], dropout_prob=0.1953):
    results = []
    print("="*80)
    print(f"Calculating model sizes for:")
    print(f"  Input dim: {in_dim}")
    print(f"  Classes: {num_classes}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Dropout: {dropout_prob}")
    print("="*80)
    print()
    print("📊 ESC (Full Precision FP32)")
    print("-" * 80)
    model_esc = ESCModel(in_dim, num_classes, hidden_sizes, dropout_prob)
    size_esc = print_size_of_model(model_esc, "ESC_FP32", debug=False)
    params_esc = sum(p.numel() for p in model_esc.parameters())
    results.append({
        'Model': 'ESC (FP32)',
        'Quantization': 'None',
        'Bits': 32,
        'Params': params_esc,
        'Size (KB)': size_esc / 1e3,
        'Size (MB)': size_esc / 1e6,
        'Reduction vs FP32': '1.00x'
    })
    print()
    for bits in [1, 2, 4, 8, 16]:
        print(f"📊 {bits}-bit Quantization")
        print("-" * 80)
        model = BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=bits)
        first_bitlinear = next((m for m in model.modules() if hasattr(m, 'num_bits')), None)
        if first_bitlinear:
            print(f"✓ Verified: First BitLinear has num_bits = {first_bitlinear.num_bits}")
        size = print_size_of_model(model, f"{bits}-bit", debug=False)
        params = sum(p.numel() for p in model.parameters())
        results.append({
            'Model': f'{bits}-bit',
            'Quantization': 'BitLinear',
            'Bits': bits,
            'Params': params,
            'Size (KB)': size / 1e3,
            'Size (MB)': size / 1e6,
            'Reduction vs FP32': f'{size_esc/size:.2f}x'
        })
        print()
    print("📊 BitNet (Ternary {-1, 0, 1})")
    print("-" * 80)
    model_bitnet = BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits="bitnet")
    first_bitlinear = next((m for m in model_bitnet.modules() if hasattr(m, 'num_bits')), None)
    if first_bitlinear:
        print(f"✓ Verified: First BitLinear has num_bits = {first_bitlinear.num_bits}")
    size_bitnet = print_size_of_model(model_bitnet, "BitNet_ternary", debug=False)
    params_bitnet = sum(p.numel() for p in model_bitnet.parameters())
    results.append({
        'Model': 'BitNet',
        'Quantization': 'Ternary',
        'Bits': 2,
        'Params': params_bitnet,
        'Size (KB)': size_bitnet / 1e3,
        'Size (MB)': size_bitnet / 1e6,
        'Reduction vs FP32': f'{size_esc/size_bitnet:.2f}x'
    })
    print()
    if HAS_QESC:
        print("📊 qesc (Bitwise Popcount)")
        print("-" * 80)
        model_qesc = BitNetPopcountExpert(in_dim, num_classes, hidden_sizes, dropout_prob)
        size_qesc = print_size_of_model(model_qesc, "qesc", debug=False)
        params_qesc = sum(p.numel() for p in model_qesc.parameters())
        results.append({
            'Model': 'qesc',
            'Quantization': 'BitwisePopcount',
            'Bits': 2,
            'Params': params_qesc,
            'Size (KB)': size_qesc / 1e3,
            'Size (MB)': size_qesc / 1e6,
            'Reduction vs FP32': f'{size_esc/size_qesc:.2f}x'
        })
        print()
    if HAS_MOE:
        print("📊 MoE (Mixture-of-Experts with Heterogeneous Quantization)")
        print("-" * 80)
        cfg = OmegaConf.create({
            'experiment': {
                'model': {
                    'hidden_sizes': hidden_sizes,
                    'dropout_prob': dropout_prob
                },
                'router': {
                    'expert_quantizations': ['bitnet', '1', '2', '4', '8', '16', 'qesc'],
                    'num_experts': 3,
                    'top_k': 1,
                    'hidden_dim': 128,
                    'dropout_prob': 0.2,
                    'use_curiosity': False,
                    'load_balancing': True,
                    'load_balancing_alpha': 1e-3,
                    'diversity_loss_enabled': False,
                    'diversity_loss_alpha': 1e-3
                }
            }
        })
        model_moe = qMoEModelBatched(cfg, in_dim, num_classes, num_experts=3, top_k=1)
        router = model_moe.router
        router_size = print_size_of_model(router, "Router", debug=False)
        expert_sizes = []
        for i, expert in enumerate(model_moe.experts):
            s = print_size_of_model(expert, f"Expert_{i}", debug=False)
            expert_sizes.append(s)
        avg_expert_size = sum(expert_sizes) / len(expert_sizes)
        effective_size = router_size + avg_expert_size
        params_router = sum(p.numel() for p in router.parameters())
        params_expert_avg = sum(sum(p.numel() for p in e.parameters()) for e in model_moe.experts) // len(model_moe.experts)
        params_moe = params_router + params_expert_avg
        expert_quants = cfg.experiment.router.expert_quantizations[:3]
        expert_info = f"{expert_quants[0]}-{expert_quants[1]}-{expert_quants[2]}"
        results.append({
            'Model': f'MoE ({expert_info})',
            'Quantization': 'Heterogeneous',
            'Bits': 'Mixed',
            'Params': params_moe,
            'Size (KB)': effective_size / 1e3,
            'Size (MB)': effective_size / 1e6,
            'Reduction vs FP32': f'{size_esc/effective_size:.2f}x'
        })
        print()
        print("📊 MoE with Curiosity (Bayesian Router)")
        print("-" * 80)
        cfg_curiosity = OmegaConf.create({
            'experiment': {
                'model': {
                    'hidden_sizes': hidden_sizes,
                    'dropout_prob': dropout_prob
                },
                'router': {
                    'expert_quantizations': ['bitnet', '1', '2', '4', '8', '16', 'qesc'],
                    'num_experts': 3,
                    'top_k': 1,
                    'hidden_dim': 128,
                    'dropout_prob': 0.2,
                    'use_curiosity': True,
                    'load_balancing': True,
                    'load_balancing_alpha': 1e-3,
                    'diversity_loss_enabled': False,
                    'diversity_loss_alpha': 1e-3
                }
            }
        })
        model_moe_curiosity = qMoEModelBatched(cfg_curiosity, in_dim, num_classes, num_experts=3, top_k=1)
        router_c = model_moe_curiosity.router
        router_size_c = print_size_of_model(router_c, "Router_curiosity", debug=False)
        expert_sizes_c = []
        for i, expert in enumerate(model_moe_curiosity.experts):
            s = print_size_of_model(expert, f"ExpertC_{i}", debug=False)
            expert_sizes_c.append(s)
        avg_expert_size_c = sum(expert_sizes_c) / len(expert_sizes_c)
        effective_size_c = router_size_c + avg_expert_size_c
        params_router_c = sum(p.numel() for p in router_c.parameters())
        params_expert_avg_c = sum(sum(p.numel() for p in e.parameters()) for e in model_moe_curiosity.experts) // len(model_moe_curiosity.experts)
        params_moe_curiosity = params_router_c + params_expert_avg_c
        results.append({
            'Model': f'MoE+Curiosity ({expert_info})',
            'Quantization': 'Heterogeneous+Bayesian',
            'Bits': 'Mixed',
            'Params': params_moe_curiosity,
            'Size (KB)': effective_size_c / 1e3,
            'Size (MB)': effective_size_c / 1e6,
            'Reduction vs FP32': f'{size_esc/effective_size_c:.2f}x'
        })
        print()
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Calculate theoretical model sizes for all quantization schemes')
    parser.add_argument('--in_dim', type=int, default=1536, help='Input dimension (default: 1536)')
    parser.add_argument('--num_classes', type=int, default=50, help='Number of classes (default: 50)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[640, 320], help='Hidden layer sizes (default: 640 320)')
    parser.add_argument('--dropout', type=float, default=0.1953403862875243, help='Dropout probability (default: 0.1953)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path (optional)')
    parser.add_argument('--debug', action='store_true', help='Show layer-by-layer breakdown for each model')
    args = parser.parse_args()
    df = calculate_all_model_sizes(
        in_dim=args.in_dim,
        num_classes=args.num_classes,
        hidden_sizes=args.hidden_sizes,
        dropout_prob=args.dropout
    )
    print("\n" + "="*80)
    print("📋 SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to: {args.output}")
    print("\n" + "="*80)
    print("🔍 KEY INSIGHTS")
    print("="*80)
    df_quant = df[df['Model'] != 'ESC (FP32)']
    if not df_quant.empty:
        most_efficient = df_quant.loc[df_quant['Size (MB)'].idxmin()]
        print(f"Most space-efficient: {most_efficient['Model']} ({most_efficient['Size (MB)']:.3f} MB)")
        print(f"                      {most_efficient['Reduction vs FP32']} smaller than FP32")
    print(f"\nSize progression (smallest to largest):")
    df_sorted = df.sort_values('Size (MB)')
    for _, row in df_sorted.iterrows():
        bar_length = int(row['Size (MB)'] * 10)
        bar = '█' * bar_length
        print(f"  {row['Model']:15s} {row['Size (MB)']:6.3f} MB {bar}")
    print("\n" + "="*80)
    print("✅ Calculation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
