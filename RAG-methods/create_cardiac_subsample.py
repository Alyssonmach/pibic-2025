import pandas as pd
import json
import os
from typing import List, Dict, Set

def create_full_dataset(train_path: str, test_path: str, output_path: str) -> pd.DataFrame:
    """
    Checks if a combined dataset file exists. If not, it creates one by concatenating
    training and test data files.
    """
    if os.path.exists(output_path):
        print(f"'{output_path}' already exists. Loading it directly.")
        return pd.read_csv(output_path)
    
    print(f"'{output_path}' not found. Creating it by merging '{train_path}' and '{test_path}'.")
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        df_all.to_csv(output_path, index=False)
        print(f"Successfully created and saved '{output_path}'.")
        return df_all
    except FileNotFoundError as e:
        print(f"Error: Could not find required data files to create the full dataset. {e}")
        return None

def main():
    """
    Main function to create a balanced subsample of reports with cardiac findings.
    """
    # --- CONFIGURATION ---
    FINDINGS_FILE = '../resources/common-findings.json'
    TRAIN_DATA_FILE = '../resources/train_data.csv'
    TEST_DATA_FILE = '../resources/test_data.csv'
    COMBINED_DATA_FILE = '../resources/all_data.csv'
    OUTPUT_FILE = '../resources/cardiac_subsample_150.csv'
    
    TARGET_TOTAL_SAMPLES = 150

    # --- SCRIPT EXECUTION ---

    # 1. Load or create the full dataset
    df_all = create_full_dataset(TRAIN_DATA_FILE, TEST_DATA_FILE, COMBINED_DATA_FILE)
    if df_all is None:
        return # Stop execution if data files are missing

    df_all.dropna(subset=['all_texts'], inplace=True)
    df_all = df_all.reset_index(drop=True)

    # 2. Load cardiac findings and keywords
    try:
        with open(FINDINGS_FILE, 'r') as f:
            findings_data = json.load(f)['findings']
    except FileNotFoundError:
        print(f"Error: The file '{FINDINGS_FILE}' was not found.")
        return

    num_findings = len(findings_data)
    samples_per_finding = TARGET_TOTAL_SAMPLES // num_findings
    print(f"\nTargeting ~{samples_per_finding} samples for each of the {num_findings} findings.")

    selected_indices: Set[int] = set()

    # 3. Search for reports for each finding in a balanced way
    for finding in findings_data:
        disease_name = finding['disease']
        keywords = finding['related-terms']
        # Add the main disease name to the keywords for a more comprehensive search
        keywords.append(disease_name.split('/')[0].strip())

        # Create a case-insensitive regex pattern: "keyword1|keyword2|..."
        regex_pattern = '|'.join([kw.replace('(', r'\(').replace(')', r'\)') for kw in keywords])
        
        print(f"\nSearching for '{disease_name}'...")
        
        # Find all reports matching the keywords
        matches = df_all['all_texts'].str.contains(regex_pattern, case=False, na=False)
        matching_df = df_all[matches]
        
        # Exclude reports that have already been selected for other findings
        unseen_matches_df = matching_df[~matching_df.index.isin(selected_indices)]
        
        print(f"Found {len(matching_df)} total matches, {len(unseen_matches_df)} of which are new.")

        # Sample from the unseen reports to meet the quota
        if len(unseen_matches_df) >= samples_per_finding:
            sampled_df = unseen_matches_df.sample(n=samples_per_finding, random_state=42)
        else:
            # If there aren't enough new reports, take all of them
            sampled_df = unseen_matches_df
        
        print(f"Adding {len(sampled_df)} unique reports for '{disease_name}' to the subsample.")
        selected_indices.update(sampled_df.index)

    # 4. Create and save the final subsample DataFrame
    print("\n--------------------------------------------------")
    final_subsample_df = df_all.loc[list(selected_indices)].copy()
    
    # Optional: shuffle the final dataset
    final_subsample_df = final_subsample_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Created a final subsample with {len(final_subsample_df)} unique reports.")
    
    final_subsample_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved the subsample to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()