# get_memory_usage()import os
import sys
import random
import pickle
import numpy as np 
import torch
import pandas as pd
import os
# def save_predictions(args, model_name, seed, y_test, y_pred, model, metrics):
def save_predictions(pred_folder, model_name, model, metrics, y_test, y_pred,seed):
    """
    Save predicted and true values as CSV files in a specified folder.

    Args:
    - pred_folder (str): The folder path where the files will be saved.
    - model_name (str): The name of the model.
    - seed (int): The seed used for randomization.
    - y_test (numpy array): True values.
    - y_pred (numpy array): Predicted values.
    """
    # pred_folder = os.path.join(f"{args.output_dir}", model_name, f"seed_{seed}")

    os.makedirs(pred_folder, exist_ok=True)
    np.savetxt(os.path.join(pred_folder, 'y_test.csv'), y_test, delimiter=',', header="y_test", fmt='%d', comments='')
    np.savetxt(os.path.join(pred_folder, 'y_pred.csv'), y_pred, delimiter=',', header="y_pred", fmt='%d', comments='')

    df = pd.DataFrame(metrics, index=[0])
    df.to_csv(os.path.join(pred_folder, f'metrics_seed_{seed}.csv'), index=False)
    store_pickle_model(model, model_name, seed, pred_folder)

def set_seed(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Set to False for reproducibility, True for performance


def store_pickle_model(model, model_name, seed, model_path):
    """
    Store a trained model as a pickle file.

    Args:
    - model: The trained model object to be stored.
    - model_name (str): The name of the model.
    - window_size (int): The window size used in the model.
    - model_path (str): The directory path where the model will be stored.
    """
    os.makedirs(model_path, exist_ok=True)

    model_name = f"{model_name}_{seed}"

    model_filename = os.path.join(model_path, model_name + '.pkl')

    print("Model saved in:", model_filename)
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(f"Failed to pickle the model due to: {e}")

def merge_and_split_data(train_csv, val_csv, output_train_csv, output_val_csv, test_size=0.2, random_state=42):
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    merged_df = pd.concat([train_df, val_df], ignore_index=True)
    
    X = merged_df.iloc[:, :-1].values 
    y = merged_df.iloc[:, -1].values  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    train_df_split = pd.DataFrame(X_train)
    train_df_split['label'] = y_train
    
    test_df_split = pd.DataFrame(X_test)
    test_df_split['label'] = y_test
    
    train_df_split.to_csv(output_train_csv, index=False)
    test_df_split.to_csv(output_val_csv, index=False)

    print("Data Randomly Shuffled")
    
    
def consolidate_and_average_metrics(args, output_folder):
    """
    Read the metrics for all seeds, consolidate them into a single CSV file,
    and then compute and save the average metrics along with standard deviation.

    Args:
    - args: Command-line arguments.
    """
    all_metrics = []
    num_seeds = args.total_num_seed
    
    for seed in range(num_seeds):
        seed_path_full = os.path.join(output_folder, f"seed_{seed}")
        seed_metrics_path = os.path.join(seed_path_full, f"metrics_seed_{seed}.csv")
        if os.path.exists(seed_metrics_path):
            metrics_df = pd.read_csv(seed_metrics_path)
            all_metrics.append(metrics_df)
        else:
            print(f"Metrics for seed {seed} not found at {seed_path_full}. Skipping this seed.")

    if all_metrics:
        consolidated_metrics = pd.concat(all_metrics, axis=0)
        consolidated_metrics_path = os.path.join(output_folder, f'consolidated_metrics_{args.model}.csv')
        print(f"Metrics for all seeds: \n{consolidated_metrics}".center(100, "-"))
        print(f"Consolidated metrics saved at {consolidated_metrics_path}")
        
        consolidated_metrics.to_csv(consolidated_metrics_path, index=False)
        
        average_metrics = consolidated_metrics.mean(axis=0)
        std_metrics = consolidated_metrics.std(axis=0)
        
        combined_metrics = pd.DataFrame()
        for col in average_metrics.index:
            combined_metrics[f"{col}_mean"] = [average_metrics[col]]
            combined_metrics[f"{col}_std"] = [std_metrics[col]]

        average_metrics_path = os.path.join(output_folder, f'average_metrics_{args.model}.csv')
        combined_metrics.to_csv(average_metrics_path, index=False)
        
        print(f"Average and standard deviation metrics saved at {average_metrics_path}")
        
        columns_to_print = [
            "F1 Score_mean", "F1 Score_std",
            "Training time_mean", "Training time_std",
            "Inference time_mean", "Inference time_std",
            "Mem. Infer (MB)_mean", "Mem. Infer (MB)_std"
        ]

        filtered_metrics = combined_metrics[columns_to_print]
        print(filtered_metrics)
    else:
        print("No metrics were found for any seed. Cannot calculate average metrics.")

def get_device(cfg):
    preferred = cfg.experiment.get("device", "auto").lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif preferred == "cpu":
        return torch.device("cpu")
    elif preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device option: {preferred}")