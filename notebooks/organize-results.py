import pandas as pd
import os
from pathlib import Path

# Setup paths
script_dir = Path(__file__).parent
data_dir = script_dir / "data"  # data folder inside notebooks

# Create data directory if it doesn't exist
data_dir.mkdir(exist_ok=True)

# Path to the results directory
# dataset = r"C:\Users\sebastian.cajasordon\Downloads\RESULTS-paper\Quinn Results"
# OUTPUT_PATH = "quinn-results"
# dataset = r"C:\Users\sebastian.cajasordon\Downloads\RESULTS-paper\ESC Results"
# OUTPUT_PATH = "esc-results"
dataset = r"C:\Users\sebastian.cajasordon\Downloads\RESULTS-paper\Urban8 Results"
OUTPUT_PATH = "urban8-results"

# Expected columns
expected_columns = [
    "name", "Param. Count", "Acc (Mean)", "F1 (Mean)", "Infer. RAM (GB)",
    "Train. Runtime (s)", "Latency (s)", "Energy-train", "Energy-val",
    "Emissions-train", "Emissions-val", "GPU-Power-train", "GPU-Power-val",
    "FLOPs-train", "FLOPs-val"
]

# Find all CSV files with 'summary_filtered' in the name
all_dataframes = []
processed_files = set()

for root, dirs, files in os.walk(dataset):
    for file in files:
        if 'summary_filtered' in file and file.endswith('.csv'):
            file_path = os.path.join(root, file)

            # Skip if already processed (avoid duplicates)
            if file_path in processed_files:
                continue

            processed_files.add(file_path)
            print(f"Reading: {file_path}")
            try:
                df = pd.read_csv(file_path)
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Combine all dataframes
if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Remove duplicate rows if any
    initial_row_count = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    final_row_count = len(combined_df)

    # Save to data folder
    output_path = data_dir / f"{OUTPUT_PATH}.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined {len(all_dataframes)} CSV files into {output_path}")
    print(f"Total rows: {final_row_count}")
    if initial_row_count != final_row_count:
        print(f"Removed {initial_row_count - final_row_count} duplicate rows")
else:
    print("No summary_filtered CSV files found!")
