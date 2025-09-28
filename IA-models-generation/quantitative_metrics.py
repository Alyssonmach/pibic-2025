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

# Baixa o recurso 'wordnet' para o NLTK, necessário para o METEOR.
try:
    nltk.data.find('corpora/wordnet')
except:
    print("Downloading 'wordnet' for METEOR calculation...")
    nltk.download('wordnet', quiet=True)


def load_generated_reports(json_path: str) -> pd.DataFrame:
    """Loads the generated reports from a JSON file into a pandas DataFrame."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: Generated reports file not found at '{json_path}'.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_path}'. It might be a .jsonl file or corrupted.")
        return None


def calculate_evaluation_metrics(df_merged: pd.DataFrame, generated_text_col: str, ground_truth_col: str) -> Dict[str, float]:
    """
    Calculates a comprehensive set of metrics for a set of generated reports.
    """
    generated_list = df_merged[generated_text_col].astype(str).tolist()
    truth_list = df_merged[ground_truth_col].astype(str).tolist()

    # --- Metric 1: ROUGE-L (Lexical Overlap) ---
    print("\nCalculating ROUGE-L scores...")
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [rouge_l_scorer.score(truth, gen)['rougeL'].fmeasure for gen, truth in tqdm(zip(generated_list, truth_list), total=len(generated_list), desc="ROUGE-L")]

    # --- Metric 2: METEOR (Synonym & Order Aware) ---
    print("\nCalculating METEOR scores...")
    meteor_scores = [meteor_score([truth.split()], gen.split()) for gen, truth in tqdm(zip(generated_list, truth_list), total=len(generated_list), desc="METEOR")]

    # --- Metric 3: BERTScore (Token-level Semantic Similarity) ---
    print("\nCalculating BERTScore...")
    # lang="en" is suitable for MIMIC-CXR. rescale_with_baseline=True makes scores more intuitive.
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    P, R, F1 = bert_scorer.score(generated_list, truth_list, verbose=True)
    avg_bertscore_f1 = F1.mean().item()

    # --- Metric 4: Document-level Semantic Similarity ---
    print("\nCalculating Document Semantic Similarity scores...")
    similarity_model = SentenceTransformer('all-mpnet-base-v2')
    embeddings_generated = similarity_model.encode(generated_list, show_progress_bar=True, normalize_embeddings=True)
    embeddings_truth = similarity_model.encode(truth_list, show_progress_bar=True, normalize_embeddings=True)
    semantic_scores = util.pytorch_cos_sim(embeddings_generated, embeddings_truth).diag().tolist()
    
    return {
        "Avg. ROUGE-L (F1)": np.mean(rouge_l_scores),
        "Avg. METEOR": np.mean(meteor_scores),
        "Avg. BERTScore (F1)": avg_bertscore_f1,
        "Avg. Doc Semantic Sim.": np.mean(semantic_scores)
    }


def main():
    """
    Main function to run the evaluation of generated medical reports.
    """
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

    final_results = {}

    for model_name, json_path in generated_files.items():
        print("\n" + "="*50)
        print(f"             EVALUATING MODEL: {model_name}")
        print("="*50)
        
        df_generated = load_generated_reports(json_path)
        if df_generated is None:
            continue
        
        generated_text_col = f'generated_text_{model_name.lower()}'
        if generated_text_col not in df_generated.columns:
            print(f"Error: Could not find column '{generated_text_col}' in '{json_path}'. Skipping model.")
            continue
            
        df_merged = pd.merge(df_generated, df_truth, on='study_id')
        
        if len(df_merged) == 0:
            print(f"Warning: No matching study_ids found between '{json_path}' and '{GROUND_TRUTH_CSV}'.")
            continue
            
        print(f"Found {len(df_merged)} matching reports for evaluation.")
        
        scores = calculate_evaluation_metrics(df_merged, generated_text_col, 'ground_truth_text')
        final_results[model_name] = scores

    print("\n\n" + "="*70)
    print("                        FINAL EVALUATION RESULTS")
    print("="*70)
    
    if not final_results:
        print("No results to display. Please check file paths and column names.")
        return
        
    df_results = pd.DataFrame(final_results).T
    df_results = df_results.applymap(lambda x: f"{x:.2%}")
    
    print(df_results)
    print("\n" + "="*70)

if __name__ == "__main__":
    main()