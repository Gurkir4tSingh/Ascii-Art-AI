import os
import json

# Resolve project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DATASET_FILE = os.path.join(DATA_DIR, "dataset.json")


def clean_ascii_art(raw_file: str, processed_file: str):
    """
    Cleans raw ASCII art text file:
      - Removes unnecessary blank lines
      - Splits ASCII arts on blank lines
      - Saves cleaned versions to 'data/processed/'
    """

    # Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Full input/output file paths
    raw_path = os.path.join(RAW_DIR, raw_file)
    processed_path = os.path.join(PROCESSED_DIR, processed_file)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"❌ Raw file not found: {raw_path}")

    cleaned_arts = []
    buffer = []

    # Read and clean the raw ASCII file
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # If a blank line, finish one ASCII art block
            if line.strip() == "":
                if buffer:
                    cleaned_arts.append("\n".join(buffer))
                    buffer = []
            else:
                buffer.append(line.rstrip("\n"))

    # Add any remaining buffered ASCII art
    if buffer:
        cleaned_arts.append("\n".join(buffer))

    # Save cleaned ASCII arts
    with open(processed_path, "w", encoding="utf-8") as f:
        for art in cleaned_arts:
            f.write(art + "\n\n")

    print(f"[OK] Cleaned {len(cleaned_arts)} ASCII arts → {processed_path}")
    return cleaned_arts


def build_dataset(processed_files):
    """
    Builds a single dataset.json file combining all processed ASCII text files.
    Output format:
    [
      {"prompt": "ASCII art #1", "ascii": "...."},
      {"prompt": "ASCII art #2", "ascii": "...."}
    ]
    """

    dataset = []
    idx = 1

    # Loop through all processed files
    for filename in processed_files:
        processed_path = os.path.join(PROCESSED_DIR, filename)

        if not os.path.exists(processed_path):
            print(f"⚠️ Skipping missing file: {processed_path}")
            continue

        with open(processed_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            arts = content.split("\n\n")  # Split on double newlines

        for art in arts:
            art = art.strip()
            if not art:
                continue
            dataset.append({
                "prompt": f"ASCII art #{idx}",
                "ascii": art
            })
            idx += 1

    # Save dataset.json in /data
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"[OK] Built dataset with {len(dataset)} samples → {DATASET_FILE}")
    return dataset


if __name__ == "__main__":
    # Default filenames for quick execution
    raw_filename = "adelfaure.txt"
    processed_filename = "adelfaure_clean.txt"

    # Run cleaning + dataset build in one go
    cleaned = clean_ascii_art(raw_filename, processed_filename)
    build_dataset([processed_filename])
