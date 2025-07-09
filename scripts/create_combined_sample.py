#!/usr/bin/env python3
"""
Script to process CSV files in data folder:
1. Iterate over all CSV files excluding those ending with "_sample"
2. For each file, identify the 3 most common tags
3. Select entries with one of these 3 tags (but not multiple of these 3)
4. Replace tags field with single tag value
5. Combine all processed files into sample.csv
"""

import pandas as pd
import os
import re
from collections import Counter
from typing import List, Tuple, Set

def parse_tags(tags_string: str) -> List[str]:
    """Parse semicolon-separated tags string into list of individual tags."""
    if pd.isna(tags_string) or not tags_string:
        return []
    
    # Split by semicolon and clean up whitespace
    tags = [tag.strip() for tag in str(tags_string).split(';')]
    # Remove empty tags
    tags = [tag for tag in tags if tag]
    return tags

def get_most_common_tags(df: pd.DataFrame, n: int = 3) -> List[str]:
    """Get the n most common tags from a dataframe."""
    all_tags = []
    
    for tags_string in df['tags']:
        tags = parse_tags(tags_string)
        all_tags.extend(tags)
    
    # Count tag frequency and get top n
    tag_counter = Counter(all_tags)
    most_common = tag_counter.most_common(n)
    
    return [tag for tag, count in most_common]

def filter_entries_by_tags(df: pd.DataFrame, target_tags: List[str]) -> pd.DataFrame:
    """
    Filter entries to keep only those with exactly one of the target tags.
    Also replace the tags field with the single matching tag.
    """
    filtered_rows = []
    
    for idx, row in df.iterrows():
        tags = parse_tags(row['tags'])
        
        # Find which target tags are present
        matching_tags = [tag for tag in tags if tag in target_tags]
        
        # Keep only entries with exactly one matching tag
        if len(matching_tags) == 1:
            # Create new row with single tag
            new_row = row.copy()
            new_row['tags'] = matching_tags[0]
            filtered_rows.append(new_row)
    
    return pd.DataFrame(filtered_rows) if filtered_rows else pd.DataFrame()

def process_single_file(filepath: str) -> pd.DataFrame:
    """Process a single CSV file according to requirements."""
    print(f"Processing {filepath}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Ensure required columns exist
        if not all(col in df.columns for col in ['title', 'text', 'project', 'tags']):
            print(f"Warning: {filepath} missing required columns. Skipping.")
            return pd.DataFrame()
        
        # Get 3 most common tags
        most_common_tags = get_most_common_tags(df, 3)
        print(f"  Most common tags: {most_common_tags}")
        
        if not most_common_tags:
            print(f"  No tags found in {filepath}. Skipping.")
            return pd.DataFrame()
        
        # Filter entries
        filtered_df = filter_entries_by_tags(df, most_common_tags)
        print(f"  Filtered from {len(df)} to {len(filtered_df)} entries")
        
        return filtered_df
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return pd.DataFrame()

def main():
    """Main function to process all CSV files and combine results."""
    data_folder = "data"
    output_file = "sample.csv"
    
    # Find all CSV files that don't end with "_sample"
    csv_files = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv') and not filename.endswith('_sample.csv'):
            csv_files.append(os.path.join(data_folder, filename))
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    print()
    
    # Process each file
    all_processed_data = []
    
    for csv_file in csv_files:
        processed_df = process_single_file(csv_file)
        if not processed_df.empty:
            all_processed_data.append(processed_df)
    
    # Combine all processed data
    if all_processed_data:
        combined_df = pd.concat(all_processed_data, ignore_index=True)
        
        # Save to output file
        combined_df.to_csv(output_file, index=False)
        print(f"\nProcessing complete!")
        print(f"Combined {len(all_processed_data)} files into {output_file}")
        print(f"Total entries in final file: {len(combined_df)}")
        
        # Show summary by project
        print("\nSummary by project:")
        project_counts = combined_df['project'].value_counts()
        for project, count in project_counts.items():
            print(f"  {project}: {count} entries")
        
        # Show summary by tags
        print("\nSummary by tags:")
        tag_counts = combined_df['tags'].value_counts()
        for tag, count in tag_counts.items():
            print(f"  {tag}: {count} entries")
            
    else:
        print("No data was processed successfully.")

if __name__ == "__main__":
    main() 