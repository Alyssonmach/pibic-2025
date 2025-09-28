import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from tqdm import tqdm
import time
import json
import os
from typing import Dict, Any

# --- 1. Model Initialization ---
# The model is loaded once following the stable app.py pattern, then moved to the target device.
MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model '{MODEL_ID}' (revision: {REVISION})...")

try:
    # First, load model weights to CPU RAM, which is often more stable.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        revision=REVISION
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    
    # Run on CPU always
    print(f"Model loaded successfully on device '{DEVICE}'.")

except Exception as e:
    print(f"Failed to load the model. Error: {e}")
    exit()


def generate_report_for_image(image_path: str, prompt: str) -> str | None:
    """
    Generates a report for a single image using the pre-loaded model.
    Includes memory management best practices for inference.
    """
    try:
        image = Image.open(image_path)
        
        # Use torch.no_grad() to reduce memory consumption and speed up inference.
        with torch.no_grad():
            enc_image = model.encode_image(image)
            generated_text = model.answer_question(enc_image, prompt, tokenizer)
        return generated_text
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        return None

def main():
    """
    Main function to orchestrate the report generation process.
    """
    INPUT_CSV = '../resources/cardiac_subsample_150.csv'
    OUTPUT_FILE = '../resources/moondream_generated_reports.jsonl'
    GENERATION_PROMPT = "You are a radiologist. Describe the radiological findings in this chest X-ray. Focus on the heart, lungs, and vasculature."

    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"\nLoaded '{INPUT_CSV}' with {len(df)} exams to process.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV}' not found. Please ensure it is in the correct directory.")
        return

    # Open the output file in append mode ('a') to support incremental writing.
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        print(f"Starting report generation. Results will be saved to '{OUTPUT_FILE}' incrementally.")
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Reports"):
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
                    "generated_text_moondream": generated_text,
                    "processing_time_seconds": round(processing_time, 4)
                }
                
                # Write the JSON object as a single line.
                f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                
                # --- CRUCIAL FIX ---
                # Force the buffer to write to the disk immediately.
                f.flush()

            # Clear GPU cache to prevent memory buildup during the loop.
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print("\nProcess finished successfully.")


if __name__ == "__main__":
    main()