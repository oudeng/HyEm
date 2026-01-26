from __future__ import annotations

from typing import Optional

import torch


def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    return torch.clamp(torch.linalg.norm(x, dim=dim), min=eps)


def acosh(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Numerically stable arcosh with clamping."""
    x = torch.clamp(x, min=1.0 + eps)
    return torch.log(x + torch.sqrt(x * x - 1.0))


def lorentz_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Hyperbolic distance on the hyperboloid (curvature -1) in normal coordinates.

    Parameters
    ----------
    u, v: (..., d) tangent vectors at the origin.

    Returns
    -------
    d: (...) hyperbolic distance between exp0(u) and exp0(v).
    """
    # r_u, r_v: (..., 1)
    r_u = torch.linalg.norm(u, dim=-1)
    r_v = torch.linalg.norm(v, dim=-1)

    # cosine between directions
    # handle r=0 safely
    dot = torch.sum(u * v, dim=-1)
    denom = torch.clamp(r_u * r_v, min=1e-9)
    cos_theta = dot / denom

    # For r_u==0 or r_v==0, define cos_theta=1 so that formula reduces correctly.
    cos_theta = torch.where((r_u < 1e-9) | (r_v < 1e-9), torch.ones_like(cos_theta), cos_theta)
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)

    # alpha = cosh(r_u)cosh(r_v) - sinh(r_u)sinh(r_v)cos(theta)
    alpha = torch.cosh(r_u) * torch.cosh(r_v) - torch.sinh(r_u) * torch.sinh(r_v) * cos_theta
    return acosh(alpha)


def lorentz_distance_to_many(u: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Distance between one vector u (d,) and many vectors V (n,d)."""
    if u.ndim != 1:
        raise ValueError(f"Expected u to have shape (d,), got {tuple(u.shape)}")
    if V.ndim != 2:
        raise ValueError(f"Expected V to have shape (n,d), got {tuple(V.shape)}")

    r_u = torch.linalg.norm(u)
    r_V = torch.linalg.norm(V, dim=-1)  # (n,)
    dot = torch.mv(V, u)  # (n,)

    denom = torch.clamp(r_u * r_V, min=1e-9)
    cos_theta = dot / denom
    cos_theta = torch.where((r_u < 1e-9) | (r_V < 1e-9), torch.ones_like(cos_theta), cos_theta)
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)

    alpha = torch.cosh(r_u) * torch.cosh(r_V) - torch.sinh(r_u) * torch.sinh(r_V) * cos_theta
    return acosh(alpha)


def kappa(R: float) -> float:
    """Distortion factor kappa(R) = sinh(R)/R (with kappa(0)=1)."""
    import math

    if R <= 1e-9:
        return 1.0
    return math.sinh(R) / R
