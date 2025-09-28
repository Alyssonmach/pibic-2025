import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import time
from typing import List, Dict

def load_data(file_path: str, sample_size: int = None) -> pd.DataFrame:
    """Loads data from a CSV, drops rows with missing text, and optionally creates a sample."""
    df = pd.read_csv(file_path)
    df.dropna(subset=['all_texts'], inplace=True)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df.reset_index(drop=True)

def calculate_similarity_metrics(distances: np.ndarray, k_values: List[int]) -> Dict[str, float]:
    """
    Calculates MAS@K and Median Top-1 similarity scores using vectorized NumPy operations.
    Cosine Similarity = 1 - Cosine Distance.
    """
    similarities = 1 - distances
    
    # Calculate MAS@K for each k value in a vectorized way
    mas_scores = {}
    for k in k_values:
        # Get the mean of the top-k similarities for each query, then average across all queries
        mas_scores[f"MAS@{k}"] = np.mean(similarities[:, :k])
        
    # Calculate the median of all top-1 similarity scores
    median_top_1 = np.median(similarities[:, 0])
    mas_scores["Median_Top1"] = median_top_1
    
    return mas_scores

def evaluate_retriever(model_name: str, df_train: pd.DataFrame, df_test: pd.DataFrame, k_search: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a model, encodes texts, and retrieves the top-k nearest neighbors,
    returning both distances and indices.
    """
    print(f"\n--- Evaluating model: {model_name} ---")
    
    print("Loading model...")
    model = SentenceTransformer(model_name)

    print("Encoding training data...")
    train_embeddings = model.encode(df_train['all_texts'].tolist(), show_progress_bar=True)
    
    print("Encoding test data...")
    test_embeddings = model.encode(df_test['all_texts'].tolist(), show_progress_bar=True)

    print("Building search index...")
    index = NearestNeighbors(n_neighbors=k_search, metric='cosine', algorithm='brute')
    index.fit(train_embeddings)

    print(f"Searching for top {k_search} neighbors for each test report...")
    distances, indices = index.kneighbors(test_embeddings)
    
    return distances, indices

def main():
    """Main function to run the RAG retriever evaluation."""
    # --- CONFIGURATION ---
    TRAIN_SAMPLE_SIZE = None 
    TEST_SAMPLE_SIZE = None   

    MODELS_TO_COMPARE = {
        "MiniLM": "all-MiniLM-L6-v2", # High-performance model for English
        "BioBERT": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    }
    
    K_NEIGHBORS_TO_SEARCH = 10
    K_VALUES_FOR_MAS = [1, 5, 10]

    # --- EXECUTION ---
    print("Loading datasets...")
    df_train = load_data("../resources/train_data.csv", sample_size=TRAIN_SAMPLE_SIZE)
    df_test = load_data("../resources/test_data.csv", sample_size=TEST_SAMPLE_SIZE)
    print(f"Data loaded: {len(df_train)} training reports, {len(df_test)} test reports.")

    results = {}
    for model_alias, model_name in MODELS_TO_COMPARE.items():
        start_time = time.time()
        
        distances, _ = evaluate_retriever(model_name, df_train, df_test, K_NEIGHBORS_TO_SEARCH)
        
        similarity_scores = calculate_similarity_metrics(distances, K_VALUES_FOR_MAS)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results[model_alias] = similarity_scores
        results[model_alias]["Total Time (s)"] = total_time
        
    # --- RESULTS PRESENTATION ---
    print("\n\n" + "="*60)
    print("          FINAL COMPARISON RESULTS (SIMILARITY METRICS)")
    print("="*60)

    df_results = pd.DataFrame(results).T
    
    # Format metric columns to percentage
    metric_cols = [f"MAS@{k}" for k in K_VALUES_FOR_MAS] + ["Median_Top1"]
    for col_name in metric_cols:
        if col_name in df_results.columns:
            df_results[col_name] = df_results[col_name].apply(lambda x: f"{x:.2%}")

    df_results["Total Time (s)"] = df_results["Total Time (s)"].apply(lambda x: f"{x:.2f}s")
    
    print(df_results)
    print("\n" + "="*60)
    print("\nNote: MAS@K (Mean Average Similarity @ K) is the mean similarity of the top-K results.")
    print("Median_Top1 is the typical similarity score of the single best result.")
    print("Higher values indicate better semantic retrieval performance.")

if __name__ == "__main__":
    main()