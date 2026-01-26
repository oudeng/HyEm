#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hyem.indexing.hnsw import load_hnsw
from hyem.io import read_jsonl, load_npy, save_csv
from hyem.models.adapter import LinearAdapter
from hyem.models.gate import LinearGate, predict_gate
from hyem.retrieval.retriever import RetrievalConfig, retrieve_euclidean_text, retrieve_hyem_hyperbolic, retrieve_soft_mix
from hyem.utils import ensure_dir


def _attach_embeddings(root: Path, split: str) -> list:
    items = read_jsonl(root / f"queries_{split}.jsonl")
    X = load_npy(root / f"emb_queries_{split}.npy")
    with open(root / f"query_ids_{split}.txt", "r", encoding="utf-8") as f:
        qids = [line.strip() for line in f if line.strip()]
    out = []
    for it, qid, x in zip(items, qids, X):
        it = dict(it)
        it["qid"] = qid
        it["e_q"] = x
        out.append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["hpo", "do", "mesh"])
    ap.add_argument("--subset_size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", type=str, default="data/processed")
    ap.add_argument("--num_queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--L_h", type=int, default=200)
    ap.add_argument("--L_e", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    root = Path(args.data_dir) / args.dataset / f"{args.subset_size}_seed{args.seed}"
    out_dir = ensure_dir(root / "analysis")

    with open(root / "node_ids.txt", "r", encoding="utf-8") as f:
        node_ids = [line.strip() for line in f if line.strip()]

    e_nodes = np.load(root / "emb_nodes.npy").astype(np.float32)
    u_hyem = np.load(root / "u_hyem.npy").astype(np.float32)

    idx_dir = root / "indexes"
    idx_E_path = idx_dir / "index_text.bin"
    idx_H_path = idx_dir / "index_hyem.bin"

    idx_E = load_hnsw(idx_E_path, dim=e_nodes.shape[1], space="cosine", ef_search=200)
    idx_H = load_hnsw(idx_H_path, dim=u_hyem.shape[1], space="l2", ef_search=200)

    # adapter
    adapter = LinearAdapter(e_nodes.shape[1], u_hyem.shape[1]).to(args.device)
    adapter.load_state_dict(torch.load(root / "adapter_hyem.pt", map_location=args.device))
    adapter.eval()

    # gate (linear)
    gate_scores = None
    gate_path = root / "gate_linear.pt"
    if gate_path.exists():
        gate = LinearGate(in_dim=e_nodes.shape[1]).to(args.device)
        gate.load_state_dict(torch.load(gate_path, map_location=args.device))
        gate.eval()
    else:
        gate = None

    queries = _attach_embeddings(root, "test")[: args.num_queries]

    cfg = RetrievalConfig(k=args.k, L_h=args.L_h, L_e=args.L_e, device=args.device)

    def bench(fn, name: str):
        t0 = time.perf_counter()
        for q in queries:
            e_q = q["e_q"]
            exclude = {q.get("focus_id")} if q.get("focus_id") else None
            _ = fn(e_q, exclude)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0 / max(1, len(queries))
        return ms

    def fn_text(e_q, exclude):
        return retrieve_euclidean_text(e_q, e_nodes, idx_E, node_ids, cfg, exclude=exclude)

    def fn_hyem(e_q, exclude):
        return retrieve_hyem_hyperbolic(e_q, adapter, u_hyem, idx_H, node_ids, cfg, exclude=exclude)

    def fn_soft(e_q, exclude):
        if gate is None:
            alpha = 0.5
        else:
            alpha = float(predict_gate(gate, e_q.reshape(1, -1), device=args.device)[0])
        return retrieve_soft_mix(
            text="",
            e_q=e_q,
            adapter_hyp=adapter,
            gate_score=alpha,
            e_nodes=e_nodes,
            u_nodes=u_hyem,
            idx_E=idx_E,
            idx_H=idx_H,
            node_ids=node_ids,
            cfg=cfg,
            exclude=exclude,
        )

    rows = []
    rows.append(
        {
            "method": "euclid_text",
            "index_size_mb": idx_E_path.stat().st_size / (1024 * 1024),
            "latency_ms": bench(fn_text, "euclid_text"),
            "extra_ops": "--",
        }
    )
    rows.append(
        {
            "method": "hyem_no_gate",
            "index_size_mb": idx_H_path.stat().st_size / (1024 * 1024),
            "latency_ms": bench(fn_hyem, "hyem_no_gate"),
            "extra_ops": "adapter + rerank",
        }
    )
    # soft mixing uses both indexes
    rows.append(
        {
            "method": "hyem_soft",
            "index_size_mb": (idx_E_path.stat().st_size + idx_H_path.stat().st_size) / (1024 * 1024),
            "latency_ms": bench(fn_soft, "hyem_soft"),
            "extra_ops": "adapter + rerank + optional 2nd ANN",
        }
    )

    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "efficiency.csv")
    print(df)
    print(f"[done] saved {out_dir / 'efficiency.csv'}")


if __name__ == "__main__":
    main()
