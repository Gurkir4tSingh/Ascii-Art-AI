import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm  # progress bar

# CONFIG
DATASET_PATH = os.path.join("data", "dataset.json")
MODEL_SAVE_PATH = os.path.join("model", "ascii_model.pth")
VOCAB_SAVE_PATH = os.path.join("model", "vocab.json")

BATCH_SIZE = 4
EPOCHS = 10
LR = 0.001
HIDDEN_DIM = 128
EMBED_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SEED FOR REPRODUCIBILITY
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

        # Extract and clean ASCII art samples
        self.samples = [item["ascii"] for item in data if item.get("ascii", "").strip()]
        if not self.samples:
            raise ValueError("No valid ASCII samples found in dataset.")

        # Build vocabulary
        chars = sorted(list(set("".join(self.samples))))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

        print(f"[INFO] Loaded {len(self.samples)} samples | Vocab size: {self.vocab_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
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
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# BATCH COLLATION
def collate_batch(batch, pad_token=0):
    """Pads variable-length sequences in a batch to uniform length."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    padded_x = torch.zeros((len(xs), max_len), dtype=torch.long) + pad_token
    padded_y = torch.zeros((len(ys), max_len), dtype=torch.long) + pad_token

    for i in range(len(xs)):
        padded_x[i, :len(xs[i])] = xs[i]
        padded_y[i, :len(ys[i])] = ys[i]

    return padded_x, padded_y

# TRAINING LOOP
def train_model():
    dataset = ASCIIDataset(DATASET_PATH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_token=0),
    )

    model = ASCIIGenerator(dataset.vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining started on {DEVICE}")
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

    # Save model + vocab
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    vocab_data = {
        "char_to_idx": dataset.char_to_idx,
        "idx_to_char": dataset.idx_to_char,
    }
    with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2)

    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print(f"Vocabulary saved to {VOCAB_SAVE_PATH}")

# ENTRY POINT
if __name__ == "__main__":
    train_model()
