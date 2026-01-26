from __future__ import annotations

"""Lightweight tangent-space HGCN baseline.

This module implements a simple hyperbolic GCN-style encoder in *origin normal
coordinates* (tangent space at the origin).

Rationale:
  - HyEm's indexing layer only requires hyperbolic embeddings in a bounded-radius
    regime.
  - For a lightweight reproducibility package, we implement a minimal message-
    passing encoder without external manifold libraries.
  - We follow the common pattern: aggregate neighbor features in a normalized
    adjacency, apply linear transforms + nonlinearity, and train the resulting
    tangent vectors with the same hyperbolic ranking loss used by the base KG
    embedding.

Note:
  - This is intended as a pragmatic baseline for reviewer-motivated comparisons,
    not a full-featured reproduction of any single library's HGCN implementation.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from .hyperbolic import lorentz_distance


def _build_norm_adj(num_nodes: int, edges: np.ndarray, device: str) -> torch.Tensor:
    """Build a symmetric normalized adjacency matrix with self-loops.

    Parameters
    ----------
    edges: (E,2) array of (parent, child) node indices.
    """
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (E,2), got {edges.shape}")

    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    # undirected edges + self loops
    i = np.concatenate([src, dst, np.arange(num_nodes, dtype=np.int64)])
    j = np.concatenate([dst, src, np.arange(num_nodes, dtype=np.int64)])

    idx = torch.tensor(np.stack([i, j], axis=0), dtype=torch.long, device=device)
    val = torch.ones(idx.shape[1], dtype=torch.float32, device=device)

    deg = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    deg.scatter_add_(0, idx[0], val)
    deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1.0), -0.5)
    norm_val = deg_inv_sqrt[idx[0]] * val * deg_inv_sqrt[idx[1]]

    adj = torch.sparse_coo_tensor(idx, norm_val, size=(num_nodes, num_nodes))
    return adj.coalesce()


class TangentGCN(nn.Module):
    def __init__(self, num_nodes: int, dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.01)

        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        # H in tangent space at origin
        h = self.emb.weight
        for lin in self.layers:
            h = torch.sparse.mm(adj, h)
            h = lin(h)
            h = self.act(h)
            h = self.drop(h)
        return h


@dataclass
class HGCNTrainConfig:
    dim: int = 32
    num_layers: int = 2
    dropout: float = 0.0
    lr: float = 1e-2
    epochs: int = 50
    neg_k: int = 10
    margin: float = 1.0
    radial_margin: float = 0.1
    weight_radial: float = 0.1
    radius_budget: float = 3.0
    weight_radius: float = 0.1
    device: str = "cpu"


def _sample_negatives(num_nodes: int, num_edges: int, neg_k: int, device: str) -> torch.Tensor:
    return torch.randint(0, num_nodes, size=(num_edges, neg_k), device=device)


def train_hgcn_hyperbolic_embeddings(
    num_nodes: int,
    edges: np.ndarray,
    cfg: HGCNTrainConfig,
    seed: int = 0,
) -> np.ndarray:
    """Train a tangent-space GCN encoder with the same hyperbolic ranking loss.

    Returns
    -------
    u: (N, dim) tangent vectors (normal coordinates at the origin).
    """
    torch.manual_seed(seed)
    device = cfg.device

    adj = _build_norm_adj(num_nodes, edges, device=device)
    edge_idx = torch.tensor(edges.astype(np.int64), dtype=torch.long, device=device)

    model = TangentGCN(num_nodes=num_nodes, dim=cfg.dim, num_layers=cfg.num_layers, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for ep in range(cfg.epochs):
        model.train()
        u_all = model(adj)  # (N, d)
        u_p = u_all[edge_idx[:, 0]]
        u_c = u_all[edge_idx[:, 1]]

        d_pos = lorentz_distance(u_p, u_c)  # (E,)
        neg_idx = _sample_negatives(num_nodes, edge_idx.shape[0], cfg.neg_k, device)
        u_n = u_all[neg_idx]  # (E, k, d)
        u_p_exp = u_p[:, None, :].expand_as(u_n)
        d_neg = lorentz_distance(u_p_exp.reshape(-1, cfg.dim), u_n.reshape(-1, cfg.dim)).reshape(-1, cfg.neg_k)

        loss_rank = torch.relu(cfg.margin + d_pos[:, None] - d_neg).mean()

        # radial order (child should be further out than parent)
        r_p = torch.linalg.norm(u_p, dim=-1)
        r_c = torch.linalg.norm(u_c, dim=-1)
        loss_radial = torch.relu(cfg.radial_margin + r_p - r_c).mean()

        if cfg.weight_radius > 0:
            r_all = torch.linalg.norm(u_all, dim=-1)
            loss_R = torch.relu(r_all - cfg.radius_budget).pow(2).mean()
        else:
            loss_R = torch.tensor(0.0, device=device)

        loss = loss_rank + cfg.weight_radial * loss_radial + cfg.weight_radius * loss_R

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[hgcn-hyp] epoch {ep+1}/{cfg.epochs} loss={float(loss.detach().cpu().item()):.4f}")

    model.eval()
    with torch.no_grad():
        u = model(adj).detach().cpu().numpy().astype(np.float32)
    return u
