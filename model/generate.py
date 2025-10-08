import torch
import torch.nn.functional as F
import json
import os
from train import ASCIIGenerator

# PATHS
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "model", "ascii_model.pth")
DATASET_PATH = os.path.join(ROOT_DIR, "data", "dataset.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD VOCAB
def load_vocab():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = [item["ascii"] for item in data if "ascii" in item]
    chars = sorted(list(set("".join(samples))))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char

# GENERATION FUNCTION
def generate_ascii(prompt="A", max_length=500, temperature=0.8):
    char_to_idx, idx_to_char = load_vocab()
    vocab_size = len(char_to_idx)

    # Load trained model
    model = ASCIIGenerator(vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Prepare input
    input_ids = [char_to_idx.get(ch, 0) for ch in prompt]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    generated = prompt
    hidden = None

    print(f"Generating ASCII Art from prompt: '{prompt}'\n")

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            logits = output[0, -1, :] / temperature  # focus on last char logits
            probs = F.softmax(logits, dim=-1)

            # Sample next char from probability distribution
            next_id = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_char[next_id]
            generated += next_char

            # Next input becomes the last predicted char
            input_tensor = torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)

            # Optional: stop early if art seems complete
            if generated.endswith("\n\n") or len(generated.strip()) > max_length:
                break

    print("Generated ASCII Art:\n")
    print(generated)
    return generated

# MAIN ENTRY
if __name__ == "__main__":
    user_input = input("Enter a starting character or phrase: ") or "A"
    generate_ascii(user_input)
