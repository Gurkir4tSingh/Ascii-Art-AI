import os
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
DATASET_FILE = os.path.join(ROOT_DIR, "data", "dataset.json")


def clean_ascii_art(raw_file: str, processed_file: str):
    """
    Reads a raw ASCII art file and cleans it:
      - Strips whitespace
      - Splits on blank lines into separate ASCII arts
      - Saves into processed folder
    """
    raw_path = os.path.join(RAW_DIR, raw_file)
    processed_path = os.path.join(PROCESSED_DIR, processed_file)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    cleaned_arts = []
    buffer = []

    # Read raw ASCII file
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if buffer:  # save buffered ASCII art
                    cleaned_arts.append("\n".join(buffer))
                    buffer = []
            else:
                buffer.append(line.rstrip("\n"))

    # Add last art if exists
    if buffer:
        cleaned_arts.append("\n".join(buffer))

    # Save cleaned arts
    with open(processed_path, "w", encoding="utf-8") as f:
        for art in cleaned_arts:
            f.write(art + "\n\n")

    print(f"[OK] Cleaned {len(cleaned_arts)} ASCII arts → {processed_path}")
    return cleaned_arts


def build_dataset(processed_files):
    """
    Builds dataset.json from processed ASCII text files.
    Format: [{"prompt": "...", "ascii": "..."}]
    """
    dataset = []
    idx = 1

    for filename in processed_files:
        processed_path = os.path.join(PROCESSED_DIR, filename)

        with open(processed_path, "r", encoding="utf-8") as f:
            arts = f.read().split("\n\n")  # split by blank lines

        for art in arts:
            art = art.strip()
            if not art:
                continue
            dataset.append({
                "prompt": f"ASCII art #{idx}",
                "ascii": art
            })
            idx += 1

    # Save dataset.json
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"[OK] Built dataset with {len(dataset)} samples → {DATASET_FILE}")
    return dataset


if __name__ == "__main__":
    raw_filename = "adelfaure.txt"
    processed_filename = "adelfaure_clean.txt"
    clean_ascii_art(raw_filename, processed_filename)
    build_dataset([processed_filename])
