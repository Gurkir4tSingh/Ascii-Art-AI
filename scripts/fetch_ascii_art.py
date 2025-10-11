import os
import re
import time
import requests
from bs4 import BeautifulSoup

# Setup paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "ascii_data")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://www.asciiart.eu"

SOURCES = {
    "animals": f"{BASE_URL}/animals",
    "vehicles": f"{BASE_URL}/vehicles",
    "nature": f"{BASE_URL}/nature",
    "architecture": f"{BASE_URL}/buildings-and-places",
    "objects": f"{BASE_URL}/objects",
    "people": f"{BASE_URL}/people",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

def clean_ascii_art(art: str) -> str:
    """Clean ASCII art formatting."""
    art = art.replace("\r", "")
    lines = [line.rstrip() for line in art.split("\n")]
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines)

def fetch_ascii_from_page(url: str):
    """Scrape ASCII art <pre> blocks from a page."""
    res = requests.get(url, headers=HEADERS, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    arts = soup.find_all("pre")
    return [clean_ascii_art(a.get_text()) for a in arts if a.get_text().strip()]

def get_subpages(url: str):
    """Find all subpage links on a category page."""
    res = requests.get(url, headers=HEADERS, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = BASE_URL + href
        if BASE_URL in href and "index" not in href and href != url:
            links.append(href)
    return sorted(set(links))

def save_ascii_art(category: str, arts: list):
    """Save ASCII arts to text files grouped by category."""
    cat_dir = os.path.join(DATA_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)

    existing = len(os.listdir(cat_dir))
    for i, art in enumerate(arts, start=existing + 1):
        filename = os.path.join(cat_dir, f"{category}_{i}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(art)

def main():
    print(f"Fetching ASCII art from {len(SOURCES)} main categories...\n")

    for category, url in SOURCES.items():
        print(f"Category: {category}")
        all_arts = []

        try:
            # Fetch ASCII directly from main page
            arts = fetch_ascii_from_page(url)
            all_arts.extend(arts)

            # Then fetch from subpages
            subpages = get_subpages(url)
            print(f"  Found {len(subpages)} subpages.")
            for sub_url in subpages:
                try:
                    sub_arts = fetch_ascii_from_page(sub_url)
                    if sub_arts:
                        all_arts.extend(sub_arts)
                    time.sleep(0.5)  # gentle on server
                except Exception as e:
                    print(f"  [Error] Subpage failed: {sub_url} ({e})")

            print(f"  Total {len(all_arts)} ASCII pieces collected.")
            save_ascii_art(category, all_arts)
        except Exception as e:
            print(f"  [Error] Could not process {category}: {e}")

        print("")

    print(f"\nAll ASCII art saved under: {DATA_DIR}")

if __name__ == "__main__":
    main()
