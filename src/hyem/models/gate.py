from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_KEYWORDS = [
    "parent",
    "parents",
    "ancestor",
    "ancestors",
    "superclass",
    "superclasses",
    "broader",
    "category",
    "taxonomy",
    "is-a",
    "type of",
    "subtype",
    "subtypes",
    "belongs to",
]


def rule_based_score(texts: Sequence[str], keywords: Sequence[str] = DEFAULT_KEYWORDS) -> np.ndarray:
    """Return a score in {0,1} based on keyword matches."""
    out = []
    for t in texts:
        tl = t.lower()
        hit = any(k in tl for k in keywords)
        out.append(1.0 if hit else 0.0)
    return np.array(out, dtype=np.float32)


class GateDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2
        assert y.ndim == 1 and y.shape[0] == X.shape[0]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LinearGate(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLPGate(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class GateTrainConfig:
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 256
    weight_decay: float = 1e-4
    device: str = "cpu"


def train_gate(model: nn.Module, X: np.ndarray, y: np.ndarray, cfg: GateTrainConfig, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    ds = GateDataset(X, y)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for xb, yb in dl:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        print(f"[gate] epoch {ep+1}/{cfg.epochs} loss={float(np.mean(losses)):.4f}")
    return model


def predict_gate(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, device=device, dtype=torch.float32)
        logits = model(xb).detach().cpu().numpy()
        score = 1.0 / (1.0 + np.exp(-logits))
    return score.astype(np.float32)
