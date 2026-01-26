from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import hnswlib
import numpy as np


@dataclass
class HNSWConfig:
    space: str = "l2"  # "l2" or "cosine"
    ef_construction: int = 200
    M: int = 16
    ef_search: int = 100


def build_hnsw_index(vectors: np.ndarray, cfg: HNSWConfig) -> hnswlib.Index:
    assert vectors.ndim == 2
    dim = vectors.shape[1]
    index = hnswlib.Index(space=cfg.space, dim=dim)
    index.init_index(max_elements=vectors.shape[0], ef_construction=cfg.ef_construction, M=cfg.M)
    index.add_items(vectors, np.arange(vectors.shape[0]))
    index.set_ef(cfg.ef_search)
    return index


def save_hnsw(index: hnswlib.Index, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(path))


def load_hnsw(path: str | Path, dim: int, space: str = "l2", ef_search: int = 100) -> hnswlib.Index:
    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(str(path))
    index.set_ef(ef_search)
    return index


def query_hnsw(index: hnswlib.Index, vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    labels, distances = index.knn_query(vec.reshape(1, -1), k=k)
    return labels[0], distances[0]
