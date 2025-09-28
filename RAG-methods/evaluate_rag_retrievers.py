import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline
from tqdm import tqdm
import time
import warnings

# Suppress user warnings from the Hugging Face pipeline
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")


def load_data(file_path: str, sample_size: int = None):
    """Loads data from a CSV file, drops rows with missing text, and optionally samples the data."""
    df = pd.read_csv(file_path)
    df.dropna(subset=['all_texts'], inplace=True)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df.reset_index(drop=True)

def evaluate_retrieval_model(model_name: str, df_train, df_test, k_search: int):
    """Encodes texts and retrieves the top-k nearest neighbors for each test report."""
    print(f"\n--- Phase 1: Retrieving with model: {model_name} ---")
    model = SentenceTransformer(model_name)
    
    train_embeddings = model.encode(df_train['all_texts'].tolist(), show_progress_bar=True)
    test_embeddings = model.encode(df_test['all_texts'].tolist(), show_progress_bar=True)
    
    index = NearestNeighbors(n_neighbors=k_search, metric='cosine', algorithm='brute')
    index.fit(train_embeddings)
    
    _, indices = index.kneighbors(test_embeddings)
    return indices

def extract_entities(text: str, ner_pipeline) -> set:
    """Extracts named entities from a text using the NER pipeline and returns them as a set."""
    try:
        entities = ner_pipeline(text)
        return {entity['word'] for entity in entities}
    except Exception:
        return set()

def jaccard_similarity(set1, set2):
    """Calculates the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # To run on the full dataset, set the sample sizes to None.
    # WARNING: NER extraction is slow. A small sample is recommended for the first run.
    TRAIN_SAMPLE_SIZE = None 
    TEST_SAMPLE_SIZE = None   # Recommended to start with a low value (e.g., 100-200)

    MODELS_TO_COMPARE = {
        # Using a standard, high-performance model for English text
        "MiniLM": "all-MiniLM-L6-v2",
        "BioBERT": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    }
    
    # NER model to extract clinical entities
    NER_MODEL = "d4data/biomedical-ner-all"

    # --- PHASE 1: RETRIEVAL ---
    print("Loading datasets...")
    df_train = load_data("../resources/train_data.csv", sample_size=TRAIN_SAMPLE_SIZE)
    df_test = load_data("../resources/test_data.csv", sample_size=TEST_SAMPLE_SIZE)
    print(f"Data loaded: {len(df_train)} training reports, {len(df_test)} test reports.")

    top_1_results = {}
    for model_alias, model_name in MODELS_TO_COMPARE.items():
        indices = evaluate_retrieval_model(model_name, df_train, df_test, k_search=1)
        top_1_results[model_alias] = {
            'queries': df_test['all_texts'].tolist(),
            'retrieved': df_train.iloc[indices.flatten()]['all_texts'].tolist()
        }

    # --- PHASE 2: EVALUATION WITH NER AND JACCARD SIMILARITY ---
    print(f"\n--- Phase 2: Evaluating with NER model ({NER_MODEL}) ---")
    print("Loading NER pipeline... (this may take a moment on the first run)")
    ner_pipeline = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")

    final_results = {}

    for model_alias, results in top_1_results.items():
        print(f"\nCalculating Jaccard Similarity for {model_alias} results...")
        jaccard_scores = []
        
        for i in tqdm(range(len(results['queries']))):
            query_text = results['queries'][i]
            retrieved_text = results['retrieved'][i]

            # Extract entities from both the query and the retrieved report
            entities_query = extract_entities(query_text, ner_pipeline)
            entities_retrieved = extract_entities(retrieved_text, ner_pipeline)

            # Calculate the Jaccard similarity between the two sets of entities
            score = jaccard_similarity(entities_query, entities_retrieved)
            jaccard_scores.append(score)
            
        # Calculate the average Jaccard score for the current model
        final_results[model_alias] = {"Average Jaccard Similarity": np.mean(jaccard_scores)}

    # --- RESULTS PRESENTATION ---
    print("\n\n" + "="*60)
    print("          FINAL RESULTS (ENTITY EXTRACTION + JACCARD)")
    print("="*60)
    
    df_results = pd.DataFrame(final_results).T
    df_results["Average Jaccard Similarity"] = df_results["Average Jaccard Similarity"].apply(lambda x: f"{x:.2%}")
    print(df_results)

    print("\n" + "="*60)
    print("\nNote: 'Average Jaccard Similarity' measures the mean overlap of clinical entities")
    print("between the query report and the retrieved report. Higher values are better.")