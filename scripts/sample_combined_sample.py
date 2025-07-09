#!/usr/bin/env python3
"""
Script to create sampled versions of sample.csv.
For each project-tag pair, selects X entries and saves to sample_X.csv.
Creates files for X = 3, 5, 10, 20, 40, 100, 150, 200.
"""

import pandas as pd
import os

def create_sampled_datasets(input_file: str = "sample.csv", sample_sizes: list = None):
    """
    Create sampled datasets with X entries per project-tag pair.
    
    Args:
        input_file: Path to the input CSV file
        sample_sizes: List of sample sizes to create
    """
    
    if sample_sizes is None:
        sample_sizes = [3, 5, 10, 20, 40, 100, 150, 200]
    
    try:
        # Read the input file
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Total entries in input: {len(df)}")
        
        # Group by project and tag
        grouped = df.groupby(['project', 'tags'])
        
        # Show initial statistics
        print(f"Total project-tag pairs: {len(grouped)}")
        print("\nProject-tag pair sizes:")
        pair_sizes = grouped.size().sort_values(ascending=False)
        for (project, tag), size in pair_sizes.head(10).items():
            print(f"  {project}:{tag} = {size} entries")
        print("  ...")
        
        # Create sampled datasets for each sample size
        for sample_size in sample_sizes:
            print(f"\nCreating sample_{sample_size}.csv...")
            
            sampled_groups = []
            total_sampled = 0
            pairs_with_full_sample = 0
            pairs_with_partial_sample = 0
            
            for (project, tag), group in grouped:
                group_size = len(group)
                
                if group_size >= sample_size:
                    # Sample exactly sample_size entries
                    sampled_group = group.sample(n=sample_size, random_state=42)
                    pairs_with_full_sample += 1
                else:
                    # Take all entries if group is smaller than sample_size
                    sampled_group = group
                    pairs_with_partial_sample += 1
                
                sampled_groups.append(sampled_group)
                total_sampled += len(sampled_group)
            
            # Combine all sampled groups
            sampled_df = pd.concat(sampled_groups, ignore_index=True)
            
            # Save to file
            output_file = f"sample_{sample_size}.csv"
            sampled_df.to_csv(output_file, index=False)
            
            print(f"  Saved {len(sampled_df)} entries to {output_file}")
            print(f"  Project-tag pairs with full sample ({sample_size} entries): {pairs_with_full_sample}")
            print(f"  Project-tag pairs with partial sample (< {sample_size} entries): {pairs_with_partial_sample}")
            
            # Show breakdown by project for this sample size
            project_counts = sampled_df['project'].value_counts()
            print(f"  Entries per project:")
            for project, count in project_counts.items():
                print(f"    {project}: {count}")
        
        print(f"\nAll sampled files created successfully!")
        print("Files created:")
        for sample_size in sample_sizes:
            output_file = f"sample_{sample_size}.csv"
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"  {output_file} ({file_size:,} bytes)")
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run process_csv_files.py first.")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def show_sampling_statistics(input_file: str = "sample.csv", sample_sizes: list = None):
    """Show statistics about what would be sampled for each sample size."""
    
    if sample_sizes is None:
        sample_sizes = [3, 5, 10, 20, 40, 100, 150, 200]
    
    try:
        df = pd.read_csv(input_file)
        grouped = df.groupby(['project', 'tags'])
        pair_sizes = grouped.size()
        
        print("Sampling Statistics Preview:")
        print("=" * 60)
        
        for sample_size in sample_sizes:
            full_sample_pairs = sum(1 for size in pair_sizes if size >= sample_size)
            partial_sample_pairs = sum(1 for size in pair_sizes if size < sample_size)
            total_entries = sum(min(size, sample_size) for size in pair_sizes)
            
            print(f"Sample size {sample_size}:")
            print(f"  Pairs with full sample: {full_sample_pairs}")
            print(f"  Pairs with partial sample: {partial_sample_pairs}")
            print(f"  Total entries: {total_entries}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Show preview of what will be sampled
    show_sampling_statistics()
    
    # Create the sampled datasets
    create_sampled_datasets() 