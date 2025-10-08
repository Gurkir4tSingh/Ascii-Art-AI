import os
import re

RAW_FILE = os.path.join("data", "raw", "adelfaure.txt")
OUTPUT_DIR = os.path.join("data", "ascii_data")

CATEGORY_MAP = {
    "bar": "architecture",
    "moulin": "architecture",
    "ecole": "architecture",
    "maison": "architecture",
    "hall": "architecture",
    "cafe": "text_and_logos",
    "renoir": "text_and_logos",
    "logo": "text_and_logos",
    "car": "vehicles",
    "velo": "vehicles",
    "bateau": "vehicles",
    "landreville": "nature_and_scenes",
    "foret": "nature_and_scenes",
    "tree": "nature_and_scenes",
    "fleur": "nature_and_scenes",
    "birthday": "objects",
    "cake": "objects",
    "cnc": "objects",
    "outil": "objects",
}


def split_ascii_data():
    if not os.path.exists(RAW_FILE):
        print(f"File not found: {RAW_FILE}")
        return

    with open(RAW_FILE, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # More flexible pattern: captures things like "\ 2024 > filename.txt"
    pattern = r"[\\/ ]*\d{4}[^A-Za-z0-9]*([A-Za-z0-9_\-]+\.txt)"
    matches = list(re.finditer(pattern, content))

    if not matches:
        print("No recognizable .txt markers found in the raw file.")
        return

    total_saved = 0
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        filename = match.group(1).strip()
        art_content = content[start:end].strip()

        if len(art_content) < 10:
            continue

        lower_name = filename.lower()
        category = "misc"
        for key, cat in CATEGORY_MAP.items():
            if key in lower_name:
                category = cat
                break

        out_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as out_file:
            out_file.write(art_content)

        total_saved += 1

    print(f"Saved {total_saved} ASCII art files into {OUTPUT_DIR}")


if __name__ == "__main__":
    split_ascii_data()
