import json
import os

def convert_jsonl_to_json(jsonl_filepath: str, json_filepath: str):
    """
    Reads a JSON Lines file (.jsonl) and converts it into a standard,
    pretty-printed JSON file (.json).

    Args:
        jsonl_filepath (str): The path to the input .jsonl file.
        json_filepath (str): The path for the output .json file.
    """
    print(f"Starting conversion of '{jsonl_filepath}'...")
    
    # Check if the input file exists
    if not os.path.exists(jsonl_filepath):
        print(f"Error: Input file not found at '{jsonl_filepath}'. Aborting.")
        return

    # Read all JSON objects from the .jsonl file
    all_data = []
    with open(jsonl_filepath, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            # Strip whitespace and check if the line is not empty
            line = line.strip()
            if line:
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode a line in '{jsonl_filepath}': {line}")

    # Write the entire list of objects to a standard .json file
    with open(json_filepath, 'w', encoding='utf-8') as f_out:
        # Use indent=4 for pretty-printing with clear line breaks and indentation
        json.dump(all_data, f_out, indent=4, ensure_ascii=False)

    print(f"Successfully converted {len(all_data)} records.")
    print(f"Formatted JSON file saved to '{json_filepath}'.")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # List of files you want to convert
    files_to_convert = [
        # "../resources/moondream_generated_reports.jsonl",
        "../resources/medgemma_generated_reports.jsonl"
    ]

    for input_file in files_to_convert:
        # Create the output filename by replacing the extension
        output_file = input_file.replace('.jsonl', '.json')
        
        convert_jsonl_to_json(input_file, output_file)
        print("-" * 30)