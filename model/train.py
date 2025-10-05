import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

DATASET_PATH = os.path.join("data", "dataset.json")
MODEL_SAVE_PATH = os.path.join("model", "ascii_model.pth")

BATCH_SIZE = 4
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ASCIIDataset(Dataset):
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f" Dataset file not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract ASCII samples
        self.samples = [item["ascii"] for item in data if "ascii" in item and item["ascii"].strip()]

        # Build character vocabulary
        chars = sorted(list(set("".join(self.samples))))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        # Convert text to integer tensor sequences
        x = torch.tensor([self.char_to_idx[c] for c in text[:-1]], dtype=torch.long)
        y = torch.tensor([self.char_to_idx[c] for c in text[1:]], dtype=torch.long)
        return x, y


class ASCIIGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


def train_model():
    dataset = ASCIIDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: collate_batch(batch, dataset.vocab_size))

    model = ASCIIGenerator(dataset.vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {len(dataset)} samples | Vocab size = {dataset.vocab_size} | Device = {DEVICE}")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            output, _ = model(x)
            loss = criterion(output.view(-1, dataset.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] → Loss: {avg_loss:.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


def collate_batch(batch, vocab_size):
    """
    Pads variable-length ASCII sequences so they can form a batch tensor.
    """
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)

    padded_x = torch.full((len(xs), max_len), fill_value=vocab_size - 1, dtype=torch.long)
    padded_y = torch.full((len(ys), max_len), fill_value=vocab_size - 1, dtype=torch.long)

    for i in range(len(xs)):
        padded_x[i, :len(xs[i])] = xs[i]
        padded_y[i, :len(ys[i])] = ys[i]

    return padded_x, padded_y


if __name__ == "__main__":
    train_model()
