import pandas as pd
import json
import argparse
from typing import Dict, Any

def load_data(generated_path: str, ground_truth_path: str) -> pd.DataFrame:
    """
    Loads the generated and ground truth reports and merges them into a single DataFrame.
    """
    try:
        # Assuming the generated reports are in a standard JSON list format
        df_generated = pd.read_json(generated_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading generated reports from '{generated_path}': {e}")
        return None

    try:
        df_truth = pd.read_csv(ground_truth_path)
        # Use the 'findings_text' column for a more focused clinical comparison
        df_truth = df_truth[['study_id', 'findings_text']]
        df_truth.rename(columns={'findings_text': 'ground_truth_text'}, inplace=True)
        df_truth.dropna(subset=['ground_truth_text'], inplace=True) # Ensure no empty findings
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading ground truth data from '{ground_truth_path}': {e}")
        return None
        
    # Merge both data sources to align the reports by study_id
    df_merged = pd.merge(df_generated, df_truth, on='study_id')
    return df_merged

def fill_prompt(template_path: str, data_row: pd.Series, model_name: str) -> str:
    """
    Fills the prompt template with data from a specific row.
    """
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    except FileNotFoundError:
        return f"Error: Template file not found at '{template_path}'"

    # Get the specific column name for the generated text (e.g., 'generated_text_moondream')
    generated_text_col = f'generated_text_{model_name}'
    
    # Extract the texts from the data row
    original_report = data_row['ground_truth_text']
    generated_report = data_row[generated_text_col]

    # Replace the placeholders in the template
    # Note: The new template uses different placeholder text.
    filled_prompt = template.replace(
        "[HERE YOU INSERT THE TEXT OF THE ORIGINAL REPORT]", 
        str(original_report)
    )
    filled_prompt = filled_prompt.replace(
        "[HERE YOU INSERT THE TEXT OF THE GENERATED REPORT]", 
        str(generated_report)
    )

    return filled_prompt

def main():
    """
    Main function to parse arguments and generate the filled prompt for clinical evaluation.
    """
    # --- Argument Parsing ---
    # Sets up the command-line interface to accept model and row number.
    parser = argparse.ArgumentParser(description="Fill a prompt template with report data for evaluation.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=['moondream', 'medgemma'], 
        help="The model whose generated data you want to use ('moondream' or 'medgemma')."
    )
    parser.add_argument(
        "--row", 
        type=int, 
        required=True, 
        help="The row number (0-indexed) of the data to use for the prompt."
    )
    args = parser.parse_args()
    
    # --- File Path Configuration ---
    model_name = args.model.lower()
    row_index = args.row
    
    GENERATED_REPORTS_PATH = f'../resources/{model_name}_generated_reports.json'
    GROUND_TRUTH_PATH = '../resources/cardiac_subsample_150.csv'
    TEMPLATE_PATH = 'prompt_template.txt' # Assumes the new template is in this file

    # --- Script Execution ---
    df_merged = load_data(GENERATED_REPORTS_PATH, GROUND_TRUTH_PATH)
    
    if df_merged is None:
        return # Stop if data loading failed

    if not (0 <= row_index < len(df_merged)):
        print(f"Error: Row index {row_index} is out of bounds. Please choose an index between 0 and {len(df_merged) - 1}.")
        return

    selected_data = df_merged.iloc[row_index]
    
    final_prompt = fill_prompt(TEMPLATE_PATH, selected_data, model_name)
    
    # Print the final, ready-to-use prompt to the console.
    print("--- Your Filled Prompt for Evaluation ---")
    print(final_prompt)


if __name__ == "__main__":
    main()