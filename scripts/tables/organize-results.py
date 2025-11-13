import pandas as pd
import os
from pathlib import Path

script_dir = Path(__file__).parent
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)

dataset = r"C:\Users\sebastian.cajasordon\Downloads\RESULTS-paper\Urban8 Results"
OUTPUT_PATH = "urban8-results"

all_dataframes = []
processed_files = set()

for root, dirs, files in os.walk(dataset):
    for file in files:
        if 'summary_filtered' in file and file.endswith('.csv'):
            file_path = os.path.join(root, file)
            if file_path in processed_files:
                continue
            processed_files.add(file_path)
            try:
                df = pd.read_csv(file_path)
                all_dataframes.append(df)
            except:
                pass

if all_dataframes:
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    output_path = data_dir / f"{OUTPUT_PATH}.csv"
    combined_df.to_csv(output_path, index=False)
