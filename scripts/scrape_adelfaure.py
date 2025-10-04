import os
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

def scrape_ascii_art(url, output_file):
    os.makedirs(RAW_DIR, exist_ok=True)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    ascii_art = []
    for pre in soup.find_all("pre"):  
        ascii_art.append(pre.get_text())

    output_path = os.path.join(RAW_DIR, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(ascii_art))

    print(f"Saved {len(ascii_art)} ASCII arts to {output_path}")

if __name__ == "__main__":
    scrape_ascii_art("https://adelfaure.net/ascii/", "adelfaure.txt")
