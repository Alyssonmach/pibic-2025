import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from tqdm import tqdm
import time
import json
import os
from typing import Dict, Any, Set

# --- 1. Model Initialization ---
# The model and processor are loaded once using memory-efficient settings.
MODEL_ID = "google/medgemma-4b-pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model '{MODEL_ID}'...")

try:
    # Use device_map="auto" to allow accelerate to handle device placement,
    # and bfloat16 for memory efficiency.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)

    print(f"Model loaded successfully on device '{DEVICE}'.")

except Exception as e:
    print(f"Failed to load the model. Error: {e}")
    exit()


def generate_report_for_image(image_path: str, prompt: str) -> str | None:
    """
    Generates a report for a single image using the pre-loaded MedGemma model.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs for the model using the processor.
        inputs = processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(DEVICE, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Use torch.inference_mode() for memory efficiency during generation.
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generation = generation[0][input_len:]
        
        decoded_text = processor.decode(generation, skip_special_tokens=True)
        return decoded_text.strip()
        
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        return None

def main():
    """
    Main function to orchestrate the report generation process, with support for resuming.
    """
    INPUT_CSV = '../resources/cardiac_subsample_150.csv'
    OUTPUT_FILE = '../resources/medgemma_generated_reports.jsonl'
    GENERATION_PROMPT = "<start_of_image> findings:"

    # --- Resume Logic ---
    # Check for an existing results file to determine which reports to skip.
    processed_study_ids: Set[int] = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing results file: '{OUTPUT_FILE}'. Reading to determine progress...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_study_ids.add(json.loads(line)['study_id'])
                except (json.JSONDecodeError, KeyError):
                    # Ignore corrupted lines or lines without a study_id
                    continue 
        print(f"Found {len(processed_study_ids)} already processed reports. They will be skipped.")

    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"\nLoaded '{INPUT_CSV}' with {len(df)} total exams.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV}' not found. Please ensure it is in the correct directory.")
        return

    # Filter out already processed reports to avoid redundant work.
    df_to_process = df[~df['study_id'].isin(processed_study_ids)]
    
    if len(df_to_process) == 0:
        print("All reports have already been processed. Nothing to do.")
        return
        
    print(f"Resuming generation for the remaining {len(df_to_process)} reports.")

    # --- Generation Loop ---
    # Open the output file in append mode ('a') to continue adding results.
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for _, row in tqdm(df_to_process.iterrows(), total=df_to_process.shape[0], desc="Generating Reports"):
            image_path = "../" + row['image_path']
            
            if not os.path.exists(image_path):
                continue

            start_time = time.time()
            generated_text = generate_report_for_image(image_path, GENERATION_PROMPT)
            end_time = time.time()
            
            if generated_text:
                processing_time = end_time - start_time
                
                result_entry: Dict[str, Any] = {
                    "study_id": int(row['study_id']),
                    "subject_id": int(row['subject_id']),
                    "image_path": image_path,
                    "generated_text_medgemma": generated_text,
                    "processing_time_seconds": round(processing_time, 4)
                }
                
                # Write the new result as a single line and flush to disk immediately.
                f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                f.flush()

            # Clear GPU cache to prevent memory fragmentation during the long run.
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print("\nProcess finished successfully.")


if __name__ == "__main__":
    main()