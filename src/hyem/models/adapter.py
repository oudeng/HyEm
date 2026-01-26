from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .hyperbolic import lorentz_distance


class QueryDataset(Dataset):
    def __init__(self, q_emb: np.ndarray, pos_lists: List[List[int]]):
        assert q_emb.ndim == 2
        self.q_emb = q_emb.astype(np.float32)
        self.pos_lists = pos_lists
        assert len(self.pos_lists) == self.q_emb.shape[0]

    def __len__(self) -> int:
        return self.q_emb.shape[0]

    def __getitem__(self, idx: int):
        return self.q_emb[idx], self.pos_lists[idx]


def _collate_fn(batch: List[Tuple[np.ndarray, List[int]]]) -> Tuple[torch.Tensor, List[List[int]]]:
    """Custom collate function to handle variable-length pos_lists."""
    embs = torch.stack([torch.from_numpy(b[0]) for b in batch])
    pos_lists = [b[1] for b in batch]
    return embs, pos_lists


class LinearAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPAdapter(nn.Module):
    """A small non-linear adapter used for the adapter-expressivity ablation."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_adapter(
    adapter_type: str,
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.0,
) -> nn.Module:
    adapter_type = (adapter_type or "linear").lower()
    if adapter_type in {"linear", "affine"}:
        return LinearAdapter(in_dim, out_dim)
    if adapter_type in {"mlp", "2layer", "2-layer"}:
        return MLPAdapter(in_dim, out_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown adapter_type={adapter_type}")


@dataclass
class AdapterTrainConfig:
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 256
    neg_k: int = 50
    temperature: float = 1.0
    adapter_type: str = "linear"
    mlp_hidden_dim: int = 128
    mlp_dropout: float = 0.0
    radius_budget: float = 3.0
    weight_radius: float = 0.1
    device: str = "cpu"


def _sample_negatives(num_nodes: int, batch: int, neg_k: int, device: str) -> torch.Tensor:
    return torch.randint(0, num_nodes, size=(batch, neg_k), device=device)


def train_hyperbolic_adapter(
    node_u: np.ndarray,
    q_emb: np.ndarray,
    pos_lists: List[List[int]],
    cfg: AdapterTrainConfig,
    seed: int = 0,
) -> nn.Module:
    """Train an adapter e -> u for hyperbolic retrieval (node_u fixed)."""
    torch.manual_seed(seed)
    ds = QueryDataset(q_emb, pos_lists)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=_collate_fn)

    in_dim = q_emb.shape[1]
    out_dim = node_u.shape[1]

    adapter = build_adapter(cfg.adapter_type, in_dim, out_dim, hidden_dim=cfg.mlp_hidden_dim, dropout=cfg.mlp_dropout).to(cfg.device)
    opt = torch.optim.Adam(adapter.parameters(), lr=cfg.lr)

    node_u_t = torch.tensor(node_u, device=cfg.device)  # (N,d)

    for ep in range(cfg.epochs):
        adapter.train()
        losses = []
        for batch_emb, batch_pos in dl:
            batch_emb = batch_emb.to(cfg.device)
            # choose one positive per query in batch (random among list)
            pos_idx = []
            for plist in batch_pos:
                plist = list(plist)
                if len(plist) == 0:
                    pos_idx.append(0)
                else:
                    pos_idx.append(plist[np.random.randint(0, len(plist))])
            pos_idx = torch.tensor(pos_idx, device=cfg.device, dtype=torch.long)

            u_q = adapter(batch_emb)  # (b,d)
            u_pos = node_u_t[pos_idx]  # (b,d)
            d_pos = lorentz_distance(u_q, u_pos)  # (b,)

            neg_idx = _sample_negatives(node_u_t.shape[0], u_q.shape[0], cfg.neg_k, cfg.device)  # (b,k)
            u_neg = node_u_t[neg_idx]  # (b,k,d)

            u_q_exp = u_q[:, None, :].expand_as(u_neg)  # (b,k,d)
            d_neg = lorentz_distance(u_q_exp.reshape(-1, out_dim), u_neg.reshape(-1, out_dim)).reshape(-1, cfg.neg_k)  # (b,k)

            # logits: higher is better
            logits_pos = (-d_pos / cfg.temperature).unsqueeze(1)  # (b,1)
            logits_neg = (-d_neg / cfg.temperature)  # (b,k)
            logits = torch.cat([logits_pos, logits_neg], dim=1)  # (b,1+k)
            labels = torch.zeros(u_q.shape[0], dtype=torch.long, device=cfg.device)
            loss_ce = nn.CrossEntropyLoss()(logits, labels)

            if cfg.weight_radius > 0:
                r_q = torch.linalg.norm(u_q, dim=-1)
                loss_R = torch.relu(r_q - cfg.radius_budget).pow(2).mean()
            else:
                loss_R = torch.tensor(0.0, device=cfg.device)

            loss = loss_ce + cfg.weight_radius * loss_R

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[hyperbolic-adapter] epoch {ep+1}/{cfg.epochs} loss={mean_loss:.4f}")

    return adapter


def train_euclidean_adapter(
    node_z: np.ndarray,
    q_emb: np.ndarray,
    pos_lists: List[List[int]],
    cfg: AdapterTrainConfig,
    seed: int = 0,
) -> nn.Module:
    """Train an adapter e -> z for Euclidean graph-embedding baseline (node_z fixed)."""
    torch.manual_seed(seed)
    ds = QueryDataset(q_emb, pos_lists)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=_collate_fn)

    in_dim = q_emb.shape[1]
    out_dim = node_z.shape[1]
    adapter = build_adapter(cfg.adapter_type, in_dim, out_dim, hidden_dim=cfg.mlp_hidden_dim, dropout=cfg.mlp_dropout).to(cfg.device)
    opt = torch.optim.Adam(adapter.parameters(), lr=cfg.lr)

    node_z_t = torch.tensor(node_z, device=cfg.device)

    for ep in range(cfg.epochs):
        adapter.train()
        losses = []
        for batch_emb, batch_pos in dl:
            batch_emb = batch_emb.to(cfg.device)
            pos_idx = []
            for plist in batch_pos:
                plist = list(plist)
                if len(plist) == 0:
                    pos_idx.append(0)
                else:
                    pos_idx.append(plist[np.random.randint(0, len(plist))])
            pos_idx = torch.tensor(pos_idx, device=cfg.device, dtype=torch.long)

            z_q = adapter(batch_emb)
            z_pos = node_z_t[pos_idx]
            d_pos = torch.linalg.norm(z_q - z_pos, dim=-1)

            neg_idx = _sample_negatives(node_z_t.shape[0], z_q.shape[0], cfg.neg_k, cfg.device)
            z_neg = node_z_t[neg_idx]  # (b,k,d)
            z_q_exp = z_q[:, None, :].expand_as(z_neg)
            d_neg = torch.linalg.norm(z_q_exp - z_neg, dim=-1)  # (b,k)

            logits_pos = (-d_pos / cfg.temperature).unsqueeze(1)
            logits_neg = (-d_neg / cfg.temperature)
            logits = torch.cat([logits_pos, logits_neg], dim=1)
            labels = torch.zeros(z_q.shape[0], dtype=torch.long, device=cfg.device)
            loss = nn.CrossEntropyLoss()(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[euclidean-adapter] epoch {ep+1}/{cfg.epochs} loss={mean_loss:.4f}")

    return adapter