import os
import torch
import argparse
import pandas as pd
from calflops import calculate_flops
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from qnet.embeddings import *
from qnet import *

def main(args):
    device = torch.device(args.device)
    val_csv = os.path.join(args.data_path, 'val_embeddings.csv')
    
    experiment_folder = f"{args.model}_{args.data_path.split('/')[-1]}"
    output_folder = os.path.join(os.getcwd(), args.output_dir, experiment_folder)
    seed_dir = os.path.join(output_folder, f"seed_{args.seed_folder}")
    
    os.makedirs(seed_dir, exist_ok=True)

    model_path = os.path.join(seed_dir, "best_model.pth")
        
    val_dataset = EmbeddingDataset(val_csv, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_features = val_dataset.features.shape[1]
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    dummy_input = torch.randn((1, num_features)).to(device)
    model = SClassifier(num_features, args.num_classes, hidden_sizes=[256, 128, 64]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Compute FLOPs, MACs, and parameters
    flops, macs, params = calculate_flops(model=model, 
                                          input_shape=(1, num_features),
                                          output_as_string=False,
                                          output_precision=4)

    results = {
        'KFLOPS': flops / 1e3,
        'KMACS': macs / 1e3,
        'KPARAMS': params / 1e3
    }
    
    print(results)
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_path = os.path.join(seed_dir, "flops_results.csv")
    results_df.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")
    print(results_df)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FLOPs, MACs, and Params for a trained model")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory containing model folders')
    parser.add_argument('--seed_folder', type=int, required=True, help='Seed folder index where the model is stored')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes for the model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g., cpu or cuda)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--model', type=str, choices=['mlp', 'svc'], default='mlp', help='Model type used during training')
    parser.add_argument('--output_dir', type=str, default='./results_jan_29_test', help='Directory to load models and save results')
    args = parser.parse_args()
    main(args)

    