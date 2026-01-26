from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class TextEmbedderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 64
    normalize: bool = True


class TextEmbedder:
    def __init__(self, cfg: TextEmbedderConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
        )
        return emb.astype(np.float32)
