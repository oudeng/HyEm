from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from ..indexing.hnsw import query_hnsw
from ..models.hyperbolic import lorentz_distance_to_many


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return (x - mu) / (sd + eps)


@dataclass
class RetrievalConfig:
    k: int = 10
    L_h: int = 200   # hyperbolic candidate oversampling
    L_e: int = 200   # euclidean candidate oversampling for pooling
    anc_k: int = 50  # for ancestor F1 evaluation
    device: str = "cpu"


def retrieve_euclidean_text(
    e_q: np.ndarray,
    e_nodes: np.ndarray,
    idx_E,
    node_ids: Sequence[str],
    cfg: RetrievalConfig,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    labels, distances = query_hnsw(idx_E, e_q, k=cfg.L_e)
    cand = [int(i) for i in labels]
    # cosine space: distance = 1 - cosine_sim
    sims = 1.0 - distances
    order = np.argsort(-sims)  # descending similarity
    ranked = []
    for j in order:
        nid = node_ids[cand[j]]
        if exclude is not None and nid in exclude:
            continue
        ranked.append(nid)
        if len(ranked) >= cfg.k:
            break
    return ranked


def retrieve_euclidean_kg(
    e_q: np.ndarray,
    adapter_euc,
    z_nodes: np.ndarray,
    idx_Z,
    node_ids: Sequence[str],
    cfg: RetrievalConfig,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    with torch.no_grad():
        device = next(adapter_euc.parameters()).device
        z_q = adapter_euc(torch.tensor(e_q, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
    labels, distances = query_hnsw(idx_Z, z_q, k=cfg.L_e)
    cand = [int(i) for i in labels]
    # l2: smaller is better
    order = np.argsort(distances)
    ranked = []
    for j in order:
        nid = node_ids[cand[j]]
        if exclude is not None and nid in exclude:
            continue
        ranked.append(nid)
        if len(ranked) >= cfg.k:
            break
    return ranked


def retrieve_hyem_hyperbolic(
    e_q: np.ndarray,
    adapter_hyp,
    u_nodes: np.ndarray,
    idx_H,
    node_ids: Sequence[str],
    cfg: RetrievalConfig,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    # query in tangent via adapter
    with torch.no_grad():
        device = next(adapter_hyp.parameters()).device
        u_q = adapter_hyp(torch.tensor(e_q, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
    labels, _ = query_hnsw(idx_H, u_q, k=cfg.L_h)
    cand = [int(i) for i in labels]
    u_cand = u_nodes[cand]  # (L,d)

    # rerank by true hyperbolic distance (Lorentz)
    u_q_t = torch.tensor(u_q, dtype=torch.float32)
    u_c_t = torch.tensor(u_cand, dtype=torch.float32)
    d = lorentz_distance_to_many(u_q_t, u_c_t).detach().cpu().numpy()  # (L,)
    order = np.argsort(d)  # ascending distance
    ranked = []
    for j in order:
        nid = node_ids[cand[j]]
        if exclude is not None and nid in exclude:
            continue
        ranked.append(nid)
        if len(ranked) >= cfg.k:
            break
    return ranked


def retrieve_soft_mix(
    text: str,
    e_q: np.ndarray,
    adapter_hyp,
    gate_score: float,
    e_nodes: np.ndarray,
    u_nodes: np.ndarray,
    idx_E,
    idx_H,
    node_ids: Sequence[str],
    cfg: RetrievalConfig,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    """Soft mixing over a pooled candidate set C = C_E ∪ C_H.

    Scores:
      s_E = cosine(e_q, e_v)
      s_H = -d_H(u_q, u_v)
      score = α * z(s_H) + (1-α) * z(s_E)
    where z() is per-query z-score normalization over the candidate pool.
    """
    with torch.no_grad():
        device = next(adapter_hyp.parameters()).device
        u_q = adapter_hyp(torch.tensor(e_q, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)

    lab_E, dist_E = query_hnsw(idx_E, e_q, k=cfg.L_e)
    lab_H, _ = query_hnsw(idx_H, u_q, k=cfg.L_h)

    cand = sorted(set([int(i) for i in lab_E] + [int(i) for i in lab_H]))
    cand_ids = [node_ids[i] for i in cand]
    if exclude is not None:
        keep = [i for i, nid in zip(cand, cand_ids) if nid not in exclude]
        cand = keep

    if len(cand) == 0:
        return []

    # Euclidean cosine
    e_c = e_nodes[cand]  # (C,d)
    s_E = np.dot(e_c, e_q)  # since embeddings are normalized

    # Hyperbolic similarity
    u_c = u_nodes[cand]
    u_q_t = torch.tensor(u_q, dtype=torch.float32)
    u_c_t = torch.tensor(u_c, dtype=torch.float32)
    d = lorentz_distance_to_many(u_q_t, u_c_t).detach().cpu().numpy()
    s_H = -d

    sE_n = _zscore(s_E)
    sH_n = _zscore(s_H)
    alpha = float(gate_score)
    score = alpha * sH_n + (1.0 - alpha) * sE_n
    order = np.argsort(-score)

    ranked = [node_ids[cand[j]] for j in order[: cfg.k]]
    return ranked


def retrieve_hard_route(
    e_q: np.ndarray,
    adapter_hyp,
    adapter_euc,
    gate_score: float,
    e_nodes: np.ndarray,
    u_nodes: np.ndarray,
    z_nodes: np.ndarray,
    idx_E,
    idx_H,
    idx_Z,
    node_ids: Sequence[str],
    cfg: RetrievalConfig,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    """Hard routing: choose hyperbolic if α>=0.5 else Euclidean text."""
    if gate_score >= 0.5:
        return retrieve_hyem_hyperbolic(e_q, adapter_hyp, u_nodes, idx_H, node_ids, cfg, exclude=exclude)
    return retrieve_euclidean_text(e_q, e_nodes, idx_E, node_ids, cfg, exclude=exclude)
