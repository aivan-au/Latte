#!/usr/bin/env python3
"""
Script to randomly sample 1000 rows from all CSV files in the data directory.
Each sampled file will be saved with a "_sample.csv" suffix.
"""

import os
import pandas as pd
import glob
from pathlib import Path

def find_data_dir():
    """Find the data directory relative to the current working directory"""
    # Try current directory first (if running from scripts/)
    if os.path.exists("../data"):
        return "../data"
    # Try data in current directory (if running from root)
    elif os.path.exists("data"):
        return "data"
    else:
        return None

def sample_csv_files(data_dir=None, sample_size=1000):
    """
    Sample random rows from all CSV files in the specified directory.
    
    Args:
        data_dir (str): Directory containing CSV files (auto-detected if None)
        sample_size (int): Number of rows to sample from each file
    """
    # Auto-detect data directory if not specified
    if data_dir is None:
        data_dir = find_data_dir()
        if data_dir is None:
            print("Could not find 'data' directory. Please ensure it exists.")
            return
    
    # Get all CSV files in the data directory
    csv_pattern = os.path.join(data_dir, "*.csv")
    all_csv_files = glob.glob(csv_pattern)
    
    # Filter out existing sample files to avoid processing them again
    csv_files = [f for f in all_csv_files if "_sample" not in os.path.basename(f)]
    
    if not csv_files:
        if all_csv_files:
            print(f"Only sample files found in '{data_dir}' directory. Use original CSV files for sampling.")
        else:
            print(f"No CSV files found in '{data_dir}' directory.")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) in '{data_dir}' directory (excluding existing samples):")
    
    for csv_file in csv_files:
        try:
            print(f"\nProcessing: {csv_file}")
            
            # Load the CSV file
            df = pd.read_csv(csv_file)
            print(f"  Original shape: {df.shape}")
            
            # Check if the file has enough rows to sample
            if len(df) < sample_size:
                print(f"  Warning: File has only {len(df)} rows, sampling all available rows.")
                sampled_df = df.copy()
            else:
                # Randomly sample rows
                sampled_df = df.sample(n=sample_size, random_state=42)
                print(f"  Sampled {sample_size} rows")
            
            # Create output filename
            file_path = Path(csv_file)
            output_filename = file_path.stem + "_sample.csv"
            output_path = file_path.parent / output_filename
            
            # Save the sampled data
            sampled_df.to_csv(output_path, index=False)
            print(f"  Saved to: {output_path}")
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {str(e)}")
    
    print("\nSampling complete!")

if __name__ == "__main__":
    # Run the sampling function
    sample_csv_files() 