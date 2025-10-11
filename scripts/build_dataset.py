import os
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASCII_DIR = os.path.join(ROOT_DIR, "data", "ascii_data")
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "dataset.json")

def build_dataset():
    data = []
    for category in os.listdir(ASCII_DIR):
        category_path = os.path.join(ASCII_DIR, category)
        if not os.path.isdir(category_path):
            continue

        for fname in os.listdir(category_path):
            if fname.endswith(".txt"):
                fpath = os.path.join(category_path, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        data.append({
                            "label": category,
                            "ascii": content
                        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Built dataset with {len(data)} samples → {OUTPUT_PATH}")

if __name__ == "__main__":
    build_dataset()