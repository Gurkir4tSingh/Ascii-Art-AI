import os
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASCII_DIR = os.path.join(ROOT_DIR, "data", "ascii_data")
OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "dataset.json")


def build_dataset():
    dataset = []

    for category in os.listdir(ASCII_DIR):
        category_path = os.path.join(ASCII_DIR, category)
        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(category_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                art = f.read().strip()

            if len(art) < 10:
                continue  # skip tiny entries

            dataset.append({
                "label": category,
                "filename": filename,
                "ascii": art
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(dataset, out, indent=2)

    print(f"Built dataset with {len(dataset)} samples → {OUTPUT_FILE}")

if __name__ == "__main__":
    build_dataset()
