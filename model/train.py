import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm

# CONFIGURATION
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(ROOT_DIR, "data", "dataset.json")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "model", "ascii_model.pth")
VOCAB_SAVE_PATH = os.path.join(ROOT_DIR, "model", "vocab.json")

BATCH_SIZE = 2
EPOCHS = 30
LR = 0.001
HIDDEN_DIM = 256
EMBED_DIM = 64
MAX_SEQ_LEN = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# DATASET CLASS
class ASCIIDataset(Dataset):
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
        for item in data:
            art = item.get("ascii", "").strip()
            label = item.get("label", "unknown")
            if art:
                art = art[:MAX_SEQ_LEN]
                # Skip too-short samples (like 1 char)
                if len(art) > 1:
                    self.samples.append({"ascii": art, "label": label})

        if not self.samples:
            raise ValueError("No valid ASCII samples found in dataset.")

        all_text = "".join([s["ascii"] for s in self.samples])
        chars = sorted(set(all_text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

        avg_len = np.mean([len(s["ascii"]) for s in self.samples])
        print(f"[INFO] Loaded {len(self.samples)} samples | "
              f"Vocab size: {self.vocab_size} | Avg length: {avg_len:.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]["ascii"]
        if len(text) <= 1:
            # fallback, skip tiny sample
            text = "??"
        x = torch.tensor([self.char_to_idx[c] for c in text[:-1]], dtype=torch.long)
        y = torch.tensor([self.char_to_idx[c] for c in text[1:]], dtype=torch.long)
        return x, y

# MODEL CLASS
class ASCIIGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        if x.size(1) == 0:
            raise ValueError("Received empty sequence for LSTM forward pass.")
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# COLLATE FUNCTION
def collate_batch(batch, pad_token=0):
    # Filter out empty sequences if any slipped through
    batch = [(x, y) for x, y in batch if len(x) > 0 and len(y) > 0]
    if not batch:
        return torch.zeros((0, 1), dtype=torch.long), torch.zeros((0, 1), dtype=torch.long)

    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    padded_x = torch.full((len(xs), max_len), pad_token, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), pad_token, dtype=torch.long)

    for i in range(len(xs)):
        padded_x[i, :len(xs[i])] = xs[i]
        padded_y[i, :len(ys[i])] = ys[i]

    return padded_x, padded_y

# TRAINING FUNCTION
def train_model():
    dataset = ASCIIDataset(DATASET_PATH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_token=0),
        drop_last=False
    )

    model = ASCIIGenerator(dataset.vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n[TRAINING] Started on {DEVICE}\n")
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

        for x, y in progress:
            if x.size(1) == 0:
                continue  # skip zero-length batch

            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            output, _ = model(x)
            loss = criterion(output.reshape(-1, dataset.vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

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
