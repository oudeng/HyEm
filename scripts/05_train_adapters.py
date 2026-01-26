#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hyem.io import read_jsonl, load_npy
from hyem.models.adapter import AdapterTrainConfig, train_euclidean_adapter, train_hyperbolic_adapter
from hyem.utils import ensure_dir, set_seed


def _load_query_embeddings(root: Path, split: str):
    q_items = read_jsonl(root / f"queries_{split}.jsonl")
    q_emb = load_npy(root / f"emb_queries_{split}.npy")
    with open(root / f"query_ids_{split}.txt", "r", encoding="utf-8") as f:
        qids = [line.strip() for line in f if line.strip()]
    assert len(q_items) == q_emb.shape[0] == len(qids)
    # attach qid order check
    for item, qid in zip(q_items, qids):
        item["qid"] = qid
    return q_items, q_emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--neg_k", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--radius_budget", type=float, default=3.0)
    ap.add_argument("--weight_radius", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--train_types", nargs="+", default=["QE", "QH"], help="Which query types to use for adapter training.")
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    ensure_dir(root)

    with open(root / "node2idx.json", "r", encoding="utf-8") as f:
        node2idx = json.load(f)

    q_items, q_emb = _load_query_embeddings(root, "train")

    # build pos lists
    pos_lists = []
    q_emb_keep = []
    keep_items = []
    for item, emb in zip(q_items, q_emb):
        if item["type"] not in args.train_types:
            continue
        pos = []
        for nid in item.get("pos_ids", []):
            if nid in node2idx:
                pos.append(int(node2idx[nid]))
        if len(pos) == 0:
            continue
        pos_lists.append(pos)
        q_emb_keep.append(emb)
        keep_items.append(item)

    q_emb_keep = np.stack(q_emb_keep, axis=0).astype(np.float32)
    print(f"[train] adapter training queries kept: {len(keep_items)} / {len(q_items)}")

    cfg = AdapterTrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        neg_k=args.neg_k,
        temperature=args.temperature,
        radius_budget=args.radius_budget,
        weight_radius=args.weight_radius,
        device=args.device,
    )

    # load node embeddings
    u_hyem = np.load(root / "u_hyem.npy").astype(np.float32)
    u_noR = np.load(root / "u_noR.npy").astype(np.float32)
    z_euckg = np.load(root / "z_euckg.npy").astype(np.float32)

    print("\n== Train hyperbolic adapter (HYEM) ==")
    adapter_hyem = train_hyperbolic_adapter(u_hyem, q_emb_keep, pos_lists, cfg, seed=args.seed)
    torch.save(adapter_hyem.state_dict(), root / "adapter_hyem.pt")

    print("\n== Train hyperbolic adapter (no radius ctrl baseline) ==")
    adapter_noR = train_hyperbolic_adapter(u_noR, q_emb_keep, pos_lists, cfg, seed=args.seed)
    torch.save(adapter_noR.state_dict(), root / "adapter_noR.pt")

    print("\n== Train Euclidean adapter (KG baseline) ==")
    adapter_euc = train_euclidean_adapter(z_euckg, q_emb_keep, pos_lists, cfg, seed=args.seed)
    torch.save(adapter_euc.state_dict(), root / "adapter_euckg.pt")

    with open(root / "train_adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print(f"[done] saved adapters under {root}")


if __name__ == "__main__":
    main()
