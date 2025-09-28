import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import nltk
from nltk.translate.meteor_score import meteor_score
from typing import List, Dict

# Download necessary NLTK data for METEOR calculation.
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading 'wordnet' for METEOR calculation...")
    nltk.download('wordnet', quiet=True)


def load_generated_reports(json_path: str) -> pd.DataFrame:
    """Loads generated reports from a JSON file into a pandas DataFrame."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Generated reports file not found at '{json_path}'.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_path}'.")
        return None


def calculate_individual_metrics(df_merged: pd.DataFrame, generated_text_col: str, ground_truth_col: str) -> pd.DataFrame:
    """
    Calculates a comprehensive set of metrics for each report and returns them as a DataFrame.
    """
    generated_list = df_merged[generated_text_col].astype(str).tolist()
    truth_list = df_merged[ground_truth_col].astype(str).tolist()

    # --- Initialize metric calculators ---
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    similarity_model = SentenceTransformer('all-mpnet-base-v2')

    # --- Calculate scores row by row ---
    print("\nCalculating individual scores for each report...")
    rouge_l_scores = [rouge_l_scorer.score(truth, gen)['rougeL'].fmeasure for gen, truth in tqdm(zip(generated_list, truth_list), total=len(generated_list), desc="ROUGE-L")]
    meteor_scores = [meteor_score([truth.split()], gen.split()) for gen, truth in tqdm(zip(generated_list, truth_list), total=len(generated_list), desc="METEOR")]
    
    _, _, f1_bert = bert_scorer.score(generated_list, truth_list, verbose=True)
    
    embeddings_generated = similarity_model.encode(generated_list, show_progress_bar=True, normalize_embeddings=True)
    embeddings_truth = similarity_model.encode(truth_list, show_progress_bar=True, normalize_embeddings=True)
    semantic_scores = util.pytorch_cos_sim(embeddings_generated, embeddings_truth).diag().tolist()
    
    # --- Assemble results into a DataFrame ---
    individual_scores_df = pd.DataFrame({
        'study_id': df_merged['study_id'],
        'ROUGE-L (F1)': rouge_l_scores,
        'METEOR': meteor_scores,
        'BERTScore (F1)': f1_bert.tolist(),
        'Doc Semantic Sim.': semantic_scores
    })
    
    return individual_scores_df


def main():
    """Main function to run the evaluation of generated medical reports."""
    GROUND_TRUTH_CSV = '../resources/cardiac_subsample_150.csv'
    generated_files = {
        "Moondream": "../resources/moondream_generated_reports.json",
        "MedGemma": "../resources/medgemma_generated_reports.json"
    }
    
    try:
        df_truth = pd.read_csv(GROUND_TRUTH_CSV)
        df_truth = df_truth[['study_id', 'all_texts']]
        df_truth.rename(columns={'all_texts': 'ground_truth_text'}, inplace=True)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at '{GROUND_TRUTH_CSV}'. Aborting.")
        return

    summary_results = {}

    for model_name, json_path in generated_files.items():
        print("\n" + "="*50)
        print(f"             EVALUATING MODEL: {model_name}")
        print("="*50)
        
        df_generated = load_generated_reports(json_path)
        if df_generated is None: continue
        
        generated_text_col = f'generated_text_{model_name.lower()}'
        if generated_text_col not in df_generated.columns:
            print(f"Error: Column '{generated_text_col}' not in '{json_path}'. Skipping.")
            continue
            
        df_merged = pd.merge(df_generated, df_truth, on='study_id')
        if len(df_merged) == 0:
            print("Warning: No matching reports found.")
            continue
            
        print(f"Found {len(df_merged)} matching reports for evaluation.")
        
        # This now returns a DataFrame with scores for each report
        individual_scores_df = calculate_individual_metrics(df_merged, generated_text_col, 'ground_truth_text')
        
        # Save the individual scores to a CSV file
        csv_output_path = f"{model_name.lower()}_individual_scores.csv"
        individual_scores_df.to_csv(csv_output_path, index=False, float_format='%.4f')
        print(f"\nIndividual scores for {model_name} saved to '{csv_output_path}'")
        
        # Calculate the average scores for the final summary
        average_scores = individual_scores_df.mean(numeric_only=True).to_dict()
        summary_results[model_name] = average_scores

    # --- Final Summary Generation ---
    if not summary_results:
        print("\nNo results were generated. Exiting.")
        return
        
    # Save the final aggregated results to a JSON file
    summary_output_path = 'final_summary_results.json'
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4)
    print(f"\nAggregated summary results saved to '{summary_output_path}'")

    print("\n" + "="*70)
    print("                        FINAL EVALUATION SUMMARY")
    print("="*70)
    df_summary_display = pd.DataFrame(summary_results).T
    print(df_summary_display.to_string(float_format="%.4f"))
    print("="*70)


if __name__ == "__main__":
    main()