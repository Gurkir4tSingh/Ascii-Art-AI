import torch
import torch.nn as nn
import json
import os
import torch.nn.functional as F
from train import ASCIIGenerator

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "model", "ascii_model.pth")
VOCAB_PATH = os.path.join(ROOT_DIR, "model", "vocab.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    char_to_idx = vocab_data["char_to_idx"]
    idx_to_char = {int(i): ch for i, ch in vocab_data["idx_to_char"].items()}
    return char_to_idx, idx_to_char

def sample_next_char(logits, temperature=0.8):
    """Applies temperature sampling to diversify output."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id

def generate_ascii(prompt="@", max_length=500, temperature=0.8):
    char_to_idx, idx_to_char = load_vocab()
    vocab_size = len(char_to_idx)

    model = ASCIIGenerator(vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    input_ids = [char_to_idx.get(ch, 0) for ch in prompt]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    generated = prompt
    hidden = None

    print(f"Generating ASCII Art from prompt: '{prompt}'\n")

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            next_logits = output[0, -1, :]
            next_id = sample_next_char(next_logits, temperature)
            next_char = idx_to_char[next_id]
            generated += next_char
            input_tensor = torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)

            if generated.endswith("\n\n"):
                break

    print("Generated ASCII Art:\n")
    print(generated)
    return generated

if __name__ == "__main__":
    user_input = input("Enter a starting character or phrase: ") or "@"
    temp = float(input("Temperature (0.5 = safe, 1.0 = creative): ") or 0.8)
    generate_ascii(user_input, temperature=temp)
