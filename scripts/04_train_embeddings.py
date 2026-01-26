#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hyem.models.train_graph import GraphTrainConfig, train_euclidean_graph_embeddings, train_hyperbolic_graph_embeddings
from hyem.utils import ensure_dir, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")

    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--neg_k", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--margin", type=float, default=1.0)

    ap.add_argument("--radius_budget", type=float, default=3.0)
    ap.add_argument("--weight_radius", type=float, default=0.1)

    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    ensure_dir(root)
    edges_path = root / "edges.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing {edges_path}. Run 01_build_subset.py first.")

    edges = pd.read_csv(edges_path)[["parent", "child"]].values.astype(np.int64)
    num_nodes = sum(1 for _ in open(root / "node_ids.txt", "r", encoding="utf-8"))

    base_cfg = GraphTrainConfig(
        dim=args.dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        neg_k=args.neg_k,
        margin=args.margin,
        radius_budget=args.radius_budget,
        weight_radius=args.weight_radius,
        device=args.device,
    )

    print("\n== Train hyperbolic graph embeddings (HYEM, with radius budget) ==")
    u_hyem = train_hyperbolic_graph_embeddings(num_nodes, edges, base_cfg)
    np.save(root / "u_hyem.npy", u_hyem)

    print("\n== Train hyperbolic graph embeddings (no radius control baseline) ==")
    cfg_noR = GraphTrainConfig(**{**base_cfg.__dict__})
    cfg_noR.weight_radius = 0.0
    cfg_noR.radius_budget = 1e9
    u_noR = train_hyperbolic_graph_embeddings(num_nodes, edges, cfg_noR)
    np.save(root / "u_noR.npy", u_noR)

    print("\n== Train Euclidean graph embeddings baseline ==")
    z_euc = train_euclidean_graph_embeddings(num_nodes, edges, base_cfg)
    np.save(root / "z_euckg.npy", z_euc)

    with open(root / "train_embeddings_config.json", "w", encoding="utf-8") as f:
        json.dump({"base": base_cfg.__dict__, "noR": cfg_noR.__dict__}, f, indent=2)

    print(f"[done] saved u_hyem.npy, u_noR.npy, z_euckg.npy under {root}")


if __name__ == "__main__":
    main()
