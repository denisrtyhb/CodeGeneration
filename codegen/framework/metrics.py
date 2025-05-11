import os
import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
from Levenshtein import distance

def calculate_levenshtein_distance(source_code: str, generated_code: str) -> float:
    # Calculate BLEU score between source_code and generated_code
    if type(source_code) != str or type(generated_code) != str:
        return None
    res = distance(source_code, generated_code)
    return res

from codebleu import calc_codebleu
def calculate_code_bleu_score(source_code: str, generated_code: str) -> float:
    # Calculate BLEU score between source_code and generated_code
    if type(source_code) != str or type(generated_code) != str:
        return None
    res = calc_codebleu([source_code], [generated_code], weights=(0.25, 0.25, 0.25, 0.25), lang='python')
    return res

def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    # All rows have source_code column and generated_code column. Calculate BLEU score between them.
    
    metrics = {}
    for index, row in df.iterrows():
        score = calculate_levenshtein_distance(row['source code'], row['context_generation'])
        if score is not None:
            if 'levenshtein_distance' not in metrics:
                metrics['levenshtein_distance'] = []
            metrics['levenshtein_distance'].append(score)

        score = calculate_code_bleu_score(row['source code'], row['context_generation'])
        if score is not None:
            for key, value in score.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

    for key in list(metrics.keys()):
        metrics[key], metrics[key + '_std'] = np.mean(metrics[key]), np.std(metrics[key])
    return metrics


def process_csv_file(csv_path: str) -> Dict[str, Any]:
    """
    Process a single CSV file and calculate its metrics.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        Dict[str, Any]: Dictionary containing the metrics
    """
    # try:
    df = pd.read_csv(csv_path)
    return calculate_metrics(df)
    # except Exception as e:
    #     return {
    #         'error': str(e),
    #         'file': csv_path
    #     }

def save_metrics(metrics: Dict[str, Any], output_path: str):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics (Dict[str, Any]): Metrics to save
        output_path (str): Path where to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def process_directory(dataset_path: str):
    """
    Process all CSV files in a directory and save their metrics.
    
    Args:
        dataset_path (str): Path to the directory containing CSV files
    """
    dataset_path = Path(dataset_path)
    
    # Create output directory if it doesn't exist
    output_dir = dataset_path / 'metrics'
    output_dir.mkdir(exist_ok=True)
    
    # Process each CSV file
    for csv_file in dataset_path.glob('*.csv'):
        print(f"Processing {csv_file.name}...")
        
        # Calculate metrics
        metrics = process_csv_file(str(csv_file))
        
        # Save metrics to JSON
        output_file = output_dir / f"{csv_file.stem}.json"
        save_metrics(metrics, str(output_file))
        
        print(f"Saved metrics to {output_file}")

def process_all_files(dataset_path: str):
    # Call process_directory for each subdirectory in the dataset_path
    for subdir in os.listdir(dataset_path):
        process_directory(os.path.join(dataset_path, subdir))

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate metrics for CSV files in a directory')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the directory containing CSV files')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_all_files(args.dataset_path)
