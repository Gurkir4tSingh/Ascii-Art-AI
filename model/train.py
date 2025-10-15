import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm

# CONFIG
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIRS = [
    os.path.join(ROOT_DIR, "data", "ascii_data"),
    os.path.join(ROOT_DIR, "data", "processed"),
]
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "model", "ascii_model.pth")
VOCAB_SAVE_PATH = os.path.join(ROOT_DIR, "model", "vocab.json")

BATCH_SIZE = 1
EPOCHS = 60
LR = 0.0005
HIDDEN_DIM = 256
EMBED_DIM = 64
MAX_SEQ_LEN = 500
MIN_LEN = 10  # keep small ASCII blocks
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def collect_ascii_data():
    """Collect all ASCII text from data/ascii_data and data/processed."""
    ascii_pieces = []
    for folder in DATA_DIRS:
        if not os.path.exists(folder):
            continue
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.endswith(".txt"):
                    path = os.path.join(root, fname)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read().strip()
                            if len(content) >= MIN_LEN:
                                # limit overly large ASCII art
                                ascii_pieces.append(content[:MAX_SEQ_LEN])
                    except Exception as e:
                        print(f"Skipping {fname}: {e}")
    print(f"[INFO] Collected {len(ascii_pieces)} ASCII pieces from all data sources.")
    return ascii_pieces


class ASCIIDataset(Dataset):
    def __init__(self, ascii_pieces):
        if not ascii_pieces:
            raise ValueError("No ASCII art samples found.")

        # Build vocab
        all_text = "".join(ascii_pieces)
        chars = sorted(set(all_text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        self.samples = ascii_pieces

        avg_len = np.mean([len(s) for s in self.samples])
        print(f"[INFO] Loaded {len(self.samples)} samples | Vocab size = {self.vocab_size} | Avg len = {avg_len:.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        x = torch.tensor([self.char_to_idx[c] for c in text[:-1]], dtype=torch.long)
        y = torch.tensor([self.char_to_idx[c] for c in text[1:]], dtype=torch.long)
        return x, y


class ASCIIGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


def collate_batch(batch, pad_token=0):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    padded_x = torch.full((len(xs), max_len), pad_token, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_token, dtype=torch.long)

    for i in range(len(xs)):
        padded_x[i, :len(xs[i])] = xs[i]
        padded_y[i, :len(ys[i])] = ys[i]

    return padded_x, padded_y


def train_model():
    ascii_pieces = collect_ascii_data()
    dataset = ASCIIDataset(ascii_pieces)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=lambda b: collate_batch(b, pad_token=0))

    model = ASCIIGenerator(dataset.vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n[TRAINING] Started on {DEVICE}\n")
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

        for x, y in progress:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.view(-1, dataset.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

    # Save model and vocab
    os.makedirs(os.path.join(ROOT_DIR, "model"), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    vocab_data = {
        "char_to_idx": dataset.char_to_idx,
        "idx_to_char": dataset.idx_to_char,
    }
    with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2)

    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print(f"Vocabulary saved to {VOCAB_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
