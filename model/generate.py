import torch
import json
import os
from train import ASCIIGenerator

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "model", "ascii_model.pth")
VOCAB_PATH = os.path.join(ROOT_DIR, "model", "vocab.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    char_to_idx = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in vocab["char_to_idx"].items()}
    idx_to_char = {int(k): v for k, v in vocab["idx_to_char"].items()}
    return char_to_idx, idx_to_char

def generate_ascii(prompt="A", max_length=400, temperature=0.8):
    char_to_idx, idx_to_char = load_vocab()
    vocab_size = len(char_to_idx)

    model = ASCIIGenerator(vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)

    input_ids = [char_to_idx.get(ch, 0) for ch in prompt]
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    hidden = None
    generated = prompt

    print(f"Generating ASCII Art from prompt: '{prompt}'")

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)
            next_id = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_id]
            generated += next_char
            input_tensor = torch.tensor([[next_id]], device=DEVICE)

            if generated.endswith("\n\n") or len(generated) > max_length:
                break

    print("\nGenerated ASCII Art:\n")
    print(generated)
    return generated

if __name__ == "__main__":
    start = input("Enter a starting character or phrase: ") or "A"
    temp = input("Temperature (0.5 = safe, 1.0 = creative): ")
    temp = float(temp) if temp.strip() else 0.8
    generate_ascii(start, temperature=temp)
