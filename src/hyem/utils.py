from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set seeds for python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Deterministic flags can reduce speed but improve reproducibility.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


@contextmanager
def timed(msg: str, logger: Optional[callable] = None) -> Iterator[None]:
    """Simple timing context manager."""
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    out = f"{msg}: {t1 - t0:.3f}s"
    if logger is None:
        print(out)
    else:
        logger(out)


@dataclass
class RunningStat:
    """Online mean/std tracker."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))
