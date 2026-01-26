from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .hyperbolic import lorentz_distance


class EdgeDataset(Dataset):
    def __init__(self, edges: np.ndarray):
        assert edges.ndim == 2 and edges.shape[1] == 2
        self.edges = edges.astype(np.int64)

    def __len__(self) -> int:
        return self.edges.shape[0]

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        p, c = self.edges[idx]
        return int(p), int(c)


@dataclass
class GraphTrainConfig:
    dim: int = 32
    lr: float = 1e-2
    epochs: int = 10
    batch_size: int = 1024
    neg_k: int = 10
    margin: float = 1.0
    radial_margin: float = 0.1
    weight_radial: float = 0.1
    radius_budget: float = 3.0
    weight_radius: float = 0.1
    device: str = "cpu"


def _sample_negatives(num_nodes: int, batch_size: int, neg_k: int, device: str) -> torch.Tensor:
    # Uniform negative sampling. Shape: (batch, neg_k)
    return torch.randint(0, num_nodes, size=(batch_size, neg_k), device=device)


def train_hyperbolic_graph_embeddings(
    num_nodes: int,
    edges: np.ndarray,
    cfg: GraphTrainConfig,
) -> np.ndarray:
    """Train hyperbolic (Lorentz) embeddings using a lightweight ranking loss."""
    ds = EdgeDataset(edges)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    emb = nn.Embedding(num_nodes, cfg.dim)
    # small init helps avoid huge radii early
    nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    emb.to(cfg.device)
    opt = torch.optim.Adam(emb.parameters(), lr=cfg.lr)

    for ep in range(cfg.epochs):
        emb.train()
        losses = []
        for p_idx, c_idx in dl:
            p_idx = p_idx.to(cfg.device)
            c_idx = c_idx.to(cfg.device)

            u_p = emb(p_idx)  # (b,d)
            u_c = emb(c_idx)  # (b,d)

            # positive distances
            d_pos = lorentz_distance(u_p, u_c)  # (b,)

            # negatives: sample nodes for each parent
            neg_idx = _sample_negatives(num_nodes, u_p.shape[0], cfg.neg_k, cfg.device)  # (b,k)
            u_n = emb(neg_idx)  # (b,k,d)

            # expand u_p to match
            u_p_exp = u_p[:, None, :].expand_as(u_n)  # (b,k,d)
            d_neg = lorentz_distance(u_p_exp.reshape(-1, cfg.dim), u_n.reshape(-1, cfg.dim)).reshape(-1, cfg.neg_k)  # (b,k)

            # ranking loss: want d_pos + margin <= d_neg
            loss_rank = torch.relu(cfg.margin + d_pos[:, None] - d_neg).mean()

            # radial order: r_c >= r_p + radial_margin
            r_p = torch.linalg.norm(u_p, dim=-1)
            r_c = torch.linalg.norm(u_c, dim=-1)
            loss_radial = torch.relu(cfg.radial_margin + r_p - r_c).mean()

            # radius budget penalty
            if cfg.weight_radius > 0:
                r_all = torch.linalg.norm(torch.cat([u_p, u_c], dim=0), dim=-1)
                loss_R = torch.relu(r_all - cfg.radius_budget).pow(2).mean()
            else:
                loss_R = torch.tensor(0.0, device=cfg.device)

            loss = loss_rank + cfg.weight_radial * loss_radial + cfg.weight_radius * loss_R

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu().item()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[hyperbolic-graph] epoch {ep+1}/{cfg.epochs} loss={mean_loss:.4f}")

    return emb.weight.detach().cpu().numpy().astype(np.float32)


def train_euclidean_graph_embeddings(
    num_nodes: int,
    edges: np.ndarray,
    cfg: GraphTrainConfig,
) -> np.ndarray:
    """Train Euclidean graph embeddings with the same ranking loss for a baseline."""
    ds = EdgeDataset(edges)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    emb = nn.Embedding(num_nodes, cfg.dim)
    nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    emb.to(cfg.device)
    opt = torch.optim.Adam(emb.parameters(), lr=cfg.lr)

    for ep in range(cfg.epochs):
        emb.train()
        losses = []
        for p_idx, c_idx in dl:
            p_idx = p_idx.to(cfg.device)
            c_idx = c_idx.to(cfg.device)
            z_p = emb(p_idx)
            z_c = emb(c_idx)
            d_pos = torch.linalg.norm(z_p - z_c, dim=-1)  # (b,)

            neg_idx = _sample_negatives(num_nodes, z_p.shape[0], cfg.neg_k, cfg.device)
            z_n = emb(neg_idx)  # (b,k,d)
            z_p_exp = z_p[:, None, :].expand_as(z_n)
            d_neg = torch.linalg.norm(z_p_exp - z_n, dim=-1)  # (b,k)

            loss_rank = torch.relu(cfg.margin + d_pos[:, None] - d_neg).mean()
            loss = loss_rank

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"[euclidean-graph] epoch {ep+1}/{cfg.epochs} loss={mean_loss:.4f}")

    return emb.weight.detach().cpu().numpy().astype(np.float32)
