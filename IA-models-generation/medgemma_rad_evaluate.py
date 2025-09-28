import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from tqdm import tqdm
import time
import json
import os
from typing import Dict, Any

# --- 1. Model Initialization ---
# The model and processor are loaded once, following the user's medgemma.py script.
MODEL_ID = "google/medgemma-4b-pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model '{MODEL_ID}'...")

try:
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

        # Use torch.inference_mode() for memory efficiency, similar to torch.no_grad().
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            # Slice the generation to get only the newly generated tokens.
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
    Main function to orchestrate the report generation process.
    """
    INPUT_CSV = '../resources/cardiac_subsample_150.csv'
    # Use a different output file for this model's results.
    OUTPUT_FILE = '../resources/medgemma_generated_reports.jsonl'
    # Use the exact prompt structure from the user's working script.
    GENERATION_PROMPT = "<start_of_image> findings:"

    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"\nLoaded '{INPUT_CSV}' with {len(df)} exams to process.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV}' not found. Please ensure it is in the correct directory.")
        return

    # Open the output file in append mode ('a') for incremental writing.
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        print(f"Starting report generation. Results will be saved to '{OUTPUT_FILE}' incrementally.")
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Reports"):
            image_path =  "../" + row['image_path']
            
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
                
                # Write the JSON object as a single line to the file.
                f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                f.flush() # Ensure the result is written to disk immediately.

            # Clear GPU cache to prevent memory buildup.
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print("\nProcess finished successfully.")


if __name__ == "__main__":
    main()